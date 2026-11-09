"""Seekable HTTP ranges for reading a stored ZIP member without its full archive."""
import io
import re
import struct
from collections import OrderedDict
from zipfile import ZIP_STORED

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        response = self.session.head(url, timeout=60)
        response.raise_for_status()
        self.length = int(response.headers["Content-Length"])
        self.etag = response.headers.get("ETag")
        self.position = 0
        self.cache = OrderedDict()
        self.block_size = 8 * 1024 * 1024
        self.transferred = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        if target < 0:
            raise ValueError("Negative seek")
        self.position = target
        return target

    def read(self, size=-1):
        size = min(self.length - self.position, size if size >= 0 else self.length)
        if size <= 0:
            return b""
        result = bytearray()
        while size:
            start = self.position // self.block_size * self.block_size
            if start not in self.cache:
                end = min(start + self.block_size, self.length) - 1
                headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
                if self.etag:
                    headers["If-Match"] = self.etag
                with self.session.get(self.url, headers=headers, timeout=120, stream=True) as response:
                    response.raise_for_status()
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if response.status_code != 206 or not match or tuple(map(int, match.groups())) != (start, end, self.length):
                        raise RuntimeError("Server did not honor exact byte range; refusing full archive download")
                    block = response.content
                if len(block) != end - start + 1:
                    raise IOError("Truncated range")
                self.transferred += len(block)
                self.cache[start] = block
                if len(self.cache) > 4:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(start)
            block = self.cache[start]
            offset = self.position - start
            take = min(size, len(block) - offset)
            result.extend(block[offset:offset + take])
            self.position += take
            size -= take
        return bytes(result)


class StoredMemberReader(io.RawIOBase):
    def __init__(self, parent, info):
        if info.compress_type != ZIP_STORED:
            raise RuntimeError("Nested ZIP is compressed; range extraction is not possible")
        parent.seek(info.header_offset)
        header = parent.read(30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("Invalid local ZIP header")
        name_len, extra_len = struct.unpack_from("<HH", header, 26)
        self.start = info.header_offset + 30 + name_len + extra_len
        self.length = info.file_size
        self.parent = parent
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        if target < 0:
            raise ValueError("Negative seek")
        self.position = target
        return target

    def read(self, size=-1):
        size = max(0, min(self.length - self.position, size if size >= 0 else self.length))
        self.parent.seek(self.start + self.position)
        result = self.parent.read(size)
        self.position += len(result)
        return result
