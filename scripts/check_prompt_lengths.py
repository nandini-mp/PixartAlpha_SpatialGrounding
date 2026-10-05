"""Check T5-small tokenization: lengths vs config limits, unknown tokens, exact decode round-trip."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from transformers import T5TokenizerFast
from src.utils import load_config

cfg = load_config()
tok = T5TokenizerFast.from_pretrained(cfg["parser"]["model_name"])
proc = Path(cfg["paths"]["processed_dir"])
ms, mt = cfg["parser"]["max_source_len"], cfg["parser"]["max_target_len"]
ok = True
for split in ("train", "val", "test"):
    data = json.load(open(proc / f"{split}_prompts.json"))
    src = tok([d["prompt"] for d in data])["input_ids"]
    tgt = tok([d["target"] for d in data])["input_ids"]
    ls, lt = np.array([len(x) for x in src]), np.array([len(x) for x in tgt])
    unk_s = sum(x.count(tok.unk_token_id) for x in src)
    unk_t = sum(x.count(tok.unk_token_id) for x in tgt)
    rt = np.mean([tok.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) == d["target"]
                  for t, d in zip(tgt, data)])
    print(f"[{split}] src tokens mean {ls.mean():.1f} max {ls.max()} | tgt tokens mean {lt.mean():.1f} "
          f"max {lt.max()} | over limit: src {int((ls > ms).sum())} tgt {int((lt > mt).sum())} | "
          f"unk: src {unk_s} tgt {unk_t} | target decode round-trip exact: {rt * 100:.2f}%")
    ok &= (ls.max() <= ms and lt.max() <= mt and unk_t == 0 and rt == 1.0)
print("TOKENIZER CHECK PASSED" if ok else "TOKENIZER CHECK FAILED")
