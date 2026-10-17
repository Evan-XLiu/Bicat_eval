"""Build base_lite/model.safetensors from the fine-tuned checkpoint so the 22 GB
allenai/MolmoAct2-BimanualYAM weights need not be downloaded. The policy constructor
only uses the base weights as an initial value: MolmoAct2Policy.from_pretrained then
strict-loads the fine-tuned checkpoint over every tensor.

Usage: python make_molmoact2_base_lite.py <dir containing yam_fold_70/ and base_lite/>"""
import json
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

here = Path(sys.argv[1])
index_path = here / "base_lite" / "model.safetensors.index.json"
base_keys = set(json.loads(index_path.read_text())["weight_map"])
ft = load_file(here / "yam_fold_70" / "model.safetensors")
out = {k.removeprefix("model."): v for k, v in ft.items() if k.startswith("model.")}
missing, extra = sorted(base_keys - set(out)), sorted(set(out) - base_keys)
print(f"finetuned={len(ft)} base={len(base_keys)} missing={missing[:5]} extra={extra[:5]}")
assert not missing and not extra and len(out) == len(ft), "key spaces differ -- download the real base model"
save_file(out, here / "base_lite" / "model.safetensors")
index_path.rename(index_path.with_suffix(".json.unused"))
print("ok")
