# -*- coding: utf-8 -*-
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

TC_RE = re.compile(r"<tool_call>\n(\{.*?\})\n</tool_call>", re.S)


def _load(name):
    p = DATA / name
    if not p.exists():
        pytest.skip(f"yok: {name} (önce scripts/generate_dataset.py)")
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


@pytest.fixture(scope="session")
def train():
    return _load("tool_calling_train.jsonl")


@pytest.fixture(scope="session")
def train_meta():
    return _load("tool_calling_train.meta.jsonl")


@pytest.fixture(scope="session")
def val():
    return _load("tool_calling_val.jsonl")


@pytest.fixture(scope="session")
def val_meta():
    return _load("tool_calling_val.meta.jsonl")


@pytest.fixture(scope="session")
def hard_eval():
    return _load("tool_calling_hard_eval.jsonl")


@pytest.fixture(scope="session")
def hard_eval_meta():
    return _load("tool_calling_hard_eval.meta.jsonl")


@pytest.fixture(scope="session")
def catalog():
    from catalog import TOOLS, by_name  # noqa
    return TOOLS


@pytest.fixture(scope="session")
def calls_of():
    def _f(rec):
        out = []
        for m in rec["messages"]:
            if m["role"] == "assistant":
                for b in TC_RE.findall(m["content"]):
                    out.append(json.loads(b))
        return out
    return _f


def fold(s):
    import unicodedata
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    s = s.replace("İ", "i").replace("I", "ı").lower()
    for a, b in (("ı", "i"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("ç", "c")):
        s = s.replace(a, b)
    return s
