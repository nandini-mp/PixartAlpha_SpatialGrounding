"""Stage 14 probe: load PixArt-Alpha, print the transformer structure, generate one image."""
import torch, diffusers
from diffusers import PixArtAlphaPipeline

pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", torch_dtype=torch.float16).to("cuda")
tr = pipe.transformer
print("diffusers", diffusers.__version__, "| transformer:", type(tr).__name__)
print("config:", {k: tr.config[k] for k in ("num_layers", "num_attention_heads", "attention_head_dim", "cross_attention_dim", "sample_size", "in_channels") if k in tr.config})
print("params (M):", round(sum(p.numel() for p in tr.parameters()) / 1e6, 1))
blk = tr.transformer_blocks[0]
print("block type:", type(blk).__name__)
for n, m in blk.named_children():
    print("  ", n, type(m).__name__)
print("forward args:", blk.forward.__code__.co_varnames[:blk.forward.__code__.co_argcount])
img = pipe("A small dog is to the left of a large cat.", num_inference_steps=20, generator=torch.Generator("cuda").manual_seed(0)).images[0]
img.save("outputs/pixart_probe.png")
print("saved outputs/pixart_probe.png | peak VRAM GB:", round(torch.cuda.max_memory_allocated() / 2**30, 1))
