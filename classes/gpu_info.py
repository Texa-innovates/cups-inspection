# gpu_info.py
import torch

def safe_get(obj, *candidates, default=None):
    for name in candidates:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available :", torch.cuda.is_available())
    print("CUDA version   :", getattr(torch.version, "cuda", None))
    try:
        print("cuDNN version  :", torch.backends.cudnn.version())
    except Exception:
        print("cuDNN version  : None/unknown")

    if not torch.cuda.is_available():
        return

    n = torch.cuda.device_count()
    print("GPU count      :", n)

    for i in range(n):
        name = torch.cuda.get_device_name(i)
        prop = torch.cuda.get_device_properties(i)
        print(f"\n--- GPU {i} ---")
        print("Name                 :", name)
        print("Capability           : %d.%d" % (prop.major, prop.minor))
        print("Total Memory (GB)    : %.2f" % (prop.total_memory / (1024**3)))
        print("Multiprocessors      :", safe_get(prop, "multi_processor_count", "multiprocessor_count"))
        print("Max Threads/SM       :", safe_get(prop, "max_threads_per_multiprocessor"))
        print("Max Threads/Block    :", safe_get(prop, "max_threads_per_block"))
        print("Warp Size            :", safe_get(prop, "warp_size"))
        # Clock fields vary across versions; print if present
        core_clock = safe_get(prop, "core_clock_rate", "clock_rate", "coreClockRate")
        mem_clock  = safe_get(prop, "memory_clock_rate", "memoryClockRate")
        if core_clock is not None:
            print("Core Clock (kHz)     :", core_clock)
        if mem_clock is not None:
            print("Memory Clock (kHz)   :", mem_clock)
        # PCI info if present
        pci_bus   = safe_get(prop, "pci_bus_id")
        pci_dev   = safe_get(prop, "pci_device_id")
        if pci_bus is not None and pci_dev is not None:
            print("PCI Bus/Device       :", f"{pci_bus}:{pci_dev}")

        # Free/total memory (works on current device; set device first)
        torch.cuda.set_device(i)
        try:
            free, total = torch.cuda.mem_get_info()
            print("Free/Total Mem (GB)  : %.2f / %.2f" % (free/(1024**3), total/(1024**3)))
        except Exception:
            pass

    # AMP capabilities & sanity matmul
    bf16_ok = torch.cuda.is_bf16_supported()
    print("\nAMP bf16 supported:", bf16_ok)
    try:
        torch.set_float32_matmul_precision("high")
        print("TF32 (matmul) precision set to 'high'")
    except Exception:
        pass

    device = torch.device("cuda:0")
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    torch.cuda.synchronize()
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16 if bf16_ok else torch.float16):
        c = a @ b
    torch.cuda.synchronize()
    print("\nSanity matmul done on:", c.device, "dtype:", c.dtype)

if __name__ == "__main__":
    main()
