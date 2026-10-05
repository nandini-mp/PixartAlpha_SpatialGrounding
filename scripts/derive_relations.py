"""Stage 2 entry point: python scripts/derive_relations.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config
from src.data.derive_relations import run

if __name__ == "__main__":
    run(load_config())
