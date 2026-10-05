"""Train the layout model.  Smoke test: --limit 256 --epochs 3 --tag smoke"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.layout_trainer import train
from src.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--tag", default="layout")
a = ap.parse_args()
train(load_config(), limit=a.limit, epochs=a.epochs, resume=a.resume, tag=a.tag)
