import tomllib
from pathlib import Path
from types import SimpleNamespace

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


def _to_namespace(d):
    return SimpleNamespace(**{
        k: _to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()
    })


with open(CONFIG_PATH, "rb") as f:
    cfg = _to_namespace(tomllib.load(f))
