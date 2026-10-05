"""Stage 3 entry point: python scripts/make_prompts.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config
from src.data.prompt_gen import run

if __name__ == "__main__":
    run(load_config())
