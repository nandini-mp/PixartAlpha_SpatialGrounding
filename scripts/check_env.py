import sys

print("=" * 60)
print("Spatial Grounding Environment Check")
print("=" * 60)

# Python
print(f"\nPython: {sys.version}")

# PyTorch
try:
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

        # Small GPU computation
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()

        print("GPU computation: PASS")
        del x, y
        torch.cuda.empty_cache()
    else:
        print("GPU computation: SKIPPED")

except Exception as e:
    print(f"PyTorch check failed: {e}")

# Other packages
packages = [
    "numpy",
    "scipy",
    "transformers",
    "sentencepiece",
    "PIL",
    "pycocotools",
    "yaml",
    "matplotlib",
    "tqdm",
    "accelerate",
    "safetensors",
]

print("\nPackage checks:")

for package in packages:
    try:
        __import__(package)
        print(f"  {package}: OK")
    except Exception as e:
        print(f"  {package}: FAILED ({e})")

print("\n" + "=" * 60)
print("Environment check complete")
print("=" * 60)
