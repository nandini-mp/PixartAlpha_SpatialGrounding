"""Stage 1: download COCO annotations, filter, select 25k images, split 20k/2.5k/2.5k.

All boxes are stored in normalized (cx, cy, w, h) in [0,1] relative to the image.
"""
import hashlib
import json
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

ANN_MEMBER = "annotations/instances_train2017.json"


def _progress_hook(pbar):
    def hook(block_num, block_size, total_size):
        if pbar.total is None and total_size > 0:
            pbar.total = total_size
        pbar.update(block_size)
    return hook


def download_annotations(cfg):
    """Download + extract instances_train2017.json if missing. Returns its path."""
    dl_dir = Path(cfg["paths"]["download_dir"]); dl_dir.mkdir(parents=True, exist_ok=True)
    coco_dir = Path(cfg["paths"]["coco_dir"]); coco_dir.mkdir(parents=True, exist_ok=True)
    json_path = coco_dir / ANN_MEMBER
    if json_path.exists():
        print(f"[skip] found {json_path}")
        return json_path

    zip_path = dl_dir / "annotations_trainval2017.zip"
    if not zip_path.exists():
        url = cfg["data"]["annotations_url"]
        print(f"Downloading {url}")
        part = zip_path.with_suffix(".zip.part")
        with tqdm(unit="B", unit_scale=True, desc="annotations zip") as pbar:
            urllib.request.urlretrieve(url, part, _progress_hook(pbar))
        part.rename(zip_path)
    if not zipfile.is_zipfile(zip_path):
        zip_path.unlink()
        raise RuntimeError("Downloaded file is not a valid zip (deleted). Re-run the script.")
    print(f"Extracting {ANN_MEMBER}")
    with zipfile.ZipFile(zip_path) as z:
        z.extract(ANN_MEMBER, coco_dir)
    return json_path


def build_candidates(coco, dcfg):
    """Return {image_id: image_record} for images with >= min_objects valid objects."""
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {im["id"]: im for im in coco["images"]}
    per_img = defaultdict(list)
    for a in coco["annotations"]:
        if dcfg["skip_crowd"] and a.get("iscrowd", 0):
            continue
        im = imgs[a["image_id"]]
        W, H = im["width"], im["height"]
        x, y, w, h = a["bbox"]
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(W), x + w), min(float(H), y + h)   # clip to image
        bw, bh = x1 - x0, y1 - y0
        if bw <= 1 or bh <= 1:
            continue
        area_frac = (bw * bh) / (W * H)
        if area_frac < dcfg["min_box_area_frac"]:
            continue
        per_img[a["image_id"]].append({
            "ann_id": a["id"],
            "category_id": a["category_id"],
            "category": cats[a["category_id"]],
            "box": [round((x0 + bw / 2) / W, 6), round((y0 + bh / 2) / H, 6),
                    round(bw / W, 6), round(bh / H, 6)],       # normalized cx,cy,w,h
            "area_frac": round(area_frac, 6),
        })

    out = {}
    for iid, objs in per_img.items():
        objs.sort(key=lambda o: (-o["area_frac"], o["ann_id"]))   # largest first, deterministic
        objs = objs[: dcfg["max_objects"]]
        if len(objs) < dcfg["min_objects"]:
            continue
        im = imgs[iid]
        out[iid] = {"image_id": iid, "file_name": im["file_name"], "coco_url": im["coco_url"],
                    "width": im["width"], "height": im["height"], "objects": objs}
    return out


def select_and_split(candidates, dcfg, seed):
    """Deterministically pick num_samples images and split them. No overlap by construction."""
    n_tr, n_va, n_te = dcfg["split"]["train"], dcfg["split"]["val"], dcfg["split"]["test"]
    n = dcfg["num_samples"]
    if n_tr + n_va + n_te != n:
        raise ValueError(f"split sizes {n_tr}+{n_va}+{n_te} != num_samples {n}")
    ids = sorted(candidates.keys())
    if len(ids) < n:
        raise RuntimeError(f"Only {len(ids)} eligible images, need {n}. Relax filters in config.")
    perm = np.random.RandomState(seed).permutation(len(ids))[:n]
    chosen = [ids[i] for i in perm]          # order defined by the seeded permutation
    splits = {"train": chosen[:n_tr],
              "val": chosen[n_tr:n_tr + n_va],
              "test": chosen[n_tr + n_va:]}
    return {k: [candidates[i] for i in v] for k, v in splits.items()}, chosen


def run(cfg):
    json_path = download_annotations(cfg)
    print("Loading annotations (~30-60 s, a few GB RAM)...")
    with open(json_path, "r") as f:
        coco = json.load(f)
    cands = build_candidates(coco, cfg["data"])
    print(f"Eligible images: {len(cands)}")
    splits, chosen = select_and_split(cands, cfg["data"], cfg["seed"])

    out_dir = Path(cfg["paths"]["splits_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump(recs, f)
    sha = hashlib.sha256(json.dumps(chosen).encode()).hexdigest()
    cat_counts = Counter(o["category"] for r in splits["train"] for o in r["objects"])
    manifest = {"coco_version": "COCO 2017 (train2017 images)", "seed": cfg["seed"],
                "eligible_images": len(cands), "sizes": {k: len(v) for k, v in splits.items()},
                "selected_ids_sha256": sha, "filters": cfg["data"],
                "train_category_counts": dict(cat_counts.most_common())}
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("Split sizes   :", manifest["sizes"])
    print("Objects/image : mean %.2f" % np.mean([len(r["objects"]) for r in splits["train"]]))
    print("Top categories:", list(cat_counts.most_common(5)))
    print("Selected-IDs SHA256:", sha)
    return manifest
