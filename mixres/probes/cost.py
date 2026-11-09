import statistics
import time

import torch


def timed_inference(forward, warmup=1, repeats=3):
    if warmup < 1 or repeats < 3:
        raise ValueError("Timing requires at least 1 warm-up and 3 measured passes")
    for _ in range(warmup):
        temporary = forward()
        del temporary
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = forward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        if i != repeats - 1:
            del result
    return result, {"latency_s": statistics.median(times), "timed_passes_s": times,
                    "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3}
