"""Verify GPU, PyTorch, AMP, and key libraries."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.utils import load_config, set_seed, get_device

ok = True

print(f"Python        : {sys.version.split()[0]}")
print(f"PyTorch       : {torch.__version__}")
print(f"CUDA (torch)  : {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

cfg = load_config()
set_seed(cfg["seed"])
dev = get_device()

print(f"Device        : {dev}")

if dev.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()

    print(f"GPU           : {props.name}")
    print(f"VRAM          : {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")

    # AMP test
    a = torch.randn(4096, 4096, device=dev)
    b = torch.randn(4096, 4096, device=dev)

    with torch.autocast("cuda", dtype=torch.float16):
        c = a @ b

    if c.dtype == torch.float16:
        print("AMP OK")
    else:
        print("AMP FAILED")
        ok = False

    # Matmul benchmark
    ah, bh = a.half(), b.half()

    torch.cuda.synchronize()
    t = time.time()

    for _ in range(20):
        _ = ah @ bh

    torch.cuda.synchronize()
    dt = (time.time() - t) / 20

    print(
        f"fp16 4096^2 matmul: "
        f"{dt*1000:.2f} ms  "
        f"(~{2*4096**3/dt/1e12:.1f} TFLOPS)"
    )

else:
    print("WARNING: no GPU visible to PyTorch. Training will be very slow.")
    ok = False


for mod in [
    "transformers",
    "sentencepiece",
    "yaml",
    "numpy",
    "PIL",
    "matplotlib",
    "tqdm",
    "pycocotools",
    "scipy",
    "pytest",
]:
    try:
        m = __import__(mod)
        print(f"{mod:14s}: OK ({getattr(m, '__version__', '?')})")
    except Exception as e:
        print(f"{mod:14s}: MISSING ({e})")
        ok = False


print("ENV CHECK PASSED" if ok else "ENV CHECK FAILED")
sys.exit(0 if ok else 1)
