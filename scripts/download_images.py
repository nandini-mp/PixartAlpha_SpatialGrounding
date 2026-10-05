"""Stage 14: prepare the first N images of a split as 512x512 (squash, no crop).
Uses local data/coco/train2017 if present, else downloads from coco_url."""
import argparse
import io
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from src.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train")
ap.add_argument("--n", type=int, default=8000)
ap.add_argument("--size", type=int, default=512)
a = ap.parse_args()

cfg = load_config()
data = json.load(open(Path(cfg["paths"]["processed_dir"]) / f"{a.split}_prompts.json"))[:a.n]
out = Path("data/images512"); out.mkdir(parents=True, exist_ok=True)
local = Path("data/coco/train2017")
n_local = 0


def fetch(s):
    global n_local
    dst = out / s["file_name"]
    if dst.exists():
        return True
    src = local / s["file_name"]
    try:
        if src.exists():
            img = Image.open(src); n_local += 1
        else:
            img = Image.open(io.BytesIO(urllib.request.urlopen(s["coco_url"], timeout=30).read()))
        img.convert("RGB").resize((a.size, a.size), Image.BICUBIC).save(dst, quality=95)
        return True
    except Exception:
        return False


with ThreadPoolExecutor(16) as ex:
    ok = list(ex.map(fetch, data))
print(f"{sum(ok)}/{len(data)} images ready in {out} | from local: {n_local} | failed: {len(ok) - sum(ok)}")
