
import torch
import time

def gpu_info():
    if not torch.cuda.is_available():
        print("No CUDA GPU available")
        return

    properties = torch.cuda.get_device_properties(0)

    print("GPU:", properties.name)
    print("VRAM (GiB):", round(
        properties.total_memory / 1024**3, 2
    ))
    print("CUDA:", torch.version.cuda)

def start_measurement():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()

def finish_measurement(start):
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    allocated = (
        torch.cuda.max_memory_allocated() / 1024**3
    )
    reserved = (
        torch.cuda.max_memory_reserved() / 1024**3
    )

    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Peak allocated: {allocated:.2f} GiB")
    print(f"Peak reserved: {reserved:.2f} GiB")
    return elapsed, allocated, reserved
