from typing import Protocol


class Adapter(Protocol):
    patch_size: int
    special_count: int
    frame_blocks: list
    global_blocks: list
    head_layers: tuple

    def embed(self, images): ...
    def run_heads(self, cache, grids): ...
