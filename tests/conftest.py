# -*- coding: utf-8 -*-
"""
Büyük İK dataset test paketi — paylaşılan fixture'lar ve yardımcılar
===================================================================

Bu paket, `data/` altındaki üretilmiş veri setini **artefakt olarak** doğrular.
Testlerin çoğu üreticiden bağımsızdır; yalnızca:

  * `test_no_hallucination.py`  (yüzey → kanonik değer haritaları)
  * `test_generator_reproducibility.py`  (üreticiyi alt süreçte koşar)

`scripts/generate_dataset.py`'ye bilerek bağlıdır — tıpkı `validate_dataset.py`
gibi (When2Call §31).

Çalıştırma:
    pip install -r tests/requirements-test.txt
    pytest tests/                    # tüm suite
    pytest tests/ -m "not slow"      # üretici alt süreçlerini atla
    pytest tests/test_tool_calls.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# --- Depo yolları -----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SCRIPTS_DIR = REPO_ROOT / "scripts"
DOCS_DIR = REPO_ROOT / "docs"
PREVIEW_DIR = REPO_ROOT / "preview"
PREFIX = "buyuk_ik_tool_calling"

# generate_dataset / validate_dataset import edilebilsin
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# --- Düşük seviye yükleyiciler --------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Bir JSONL dosyasını satır satır ayrıştırır; her satır geçerli JSON olmalı."""
    rows: list[dict] = []
    text = path.read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:  # pragma: no cover - test hatası olarak yükselir
            raise AssertionError(f"{path.name}:{i} geçersiz JSON — {e}") from e
    return rows


def _require(path: Path):
    if not path.exists():
        pytest.skip(
            f"{path.relative_to(REPO_ROOT)} yok — önce `python scripts/generate_dataset.py` çalıştırın",
            allow_module_level=False,
        )


# --- Fixture'lar (session kapsamında, bir kez yüklenir) -------------------

@pytest.fixture(scope="session")
def data_dir() -> Path:
    if not DATA_DIR.is_dir():
        pytest.skip("data/ klasörü yok — önce üreticiyi çalıştırın")
    return DATA_DIR


@pytest.fixture(scope="session")
def train_records(data_dir) -> list[dict]:
    p = data_dir / f"{PREFIX}_train.jsonl"
    _require(p)
    return load_jsonl(p)


@pytest.fixture(scope="session")
def val_records(data_dir) -> list[dict]:
    p = data_dir / f"{PREFIX}_val.jsonl"
    _require(p)
    return load_jsonl(p)


@pytest.fixture(scope="session")
def all_records(train_records, val_records) -> list[dict]:
    return [*train_records, *val_records]


@pytest.fixture(scope="session")
def train_meta(data_dir) -> list[dict]:
    p = data_dir / f"{PREFIX}_train.meta.jsonl"
    _require(p)
    return load_jsonl(p)


@pytest.fixture(scope="session")
def val_meta(data_dir) -> list[dict]:
    p = data_dir / f"{PREFIX}_val.meta.jsonl"
    _require(p)
    return load_jsonl(p)


@pytest.fixture(scope="session")
def all_meta(train_meta, val_meta) -> list[dict]:
    return [*train_meta, *val_meta]


@pytest.fixture(scope="session")
def paired(train_records, train_meta, val_records, val_meta):
    """(record, meta) çiftleri — satır sırası eğitim dosyasıyla AYNI olmalı (§23).

    Meta satırları `tools`+`messages` alanlarını da taşır; bunların `record` ile
    birebir aynı olduğunu burada doğrular ve eşleştiririz.
    """
    pairs: list[tuple[dict, dict]] = []
    for split, recs, metas in (("train", train_records, train_meta), ("val", val_records, val_meta)):
        assert len(recs) == len(metas), (
            f"{split}: data ({len(recs)}) ile meta ({len(metas)}) satır sayısı farklı"
        )
        for i, (rec, meta) in enumerate(zip(recs, metas)):
            assert rec["messages"] == meta.get("messages"), (
                f"{split}[{i}]: meta.messages, record.messages ile aynı değil (sıra bozulmuş olabilir)"
            )
            pairs.append((rec, meta))
    return pairs


@pytest.fixture(scope="session")
def tools_inventory(data_dir) -> list[dict]:
    p = data_dir / f"{PREFIX}_tools.json"
    _require(p)
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def gen():
    """`scripts/generate_dataset.py` modülü — yüzey haritaları ve sabitler için."""
    try:
        import generate_dataset as G  # noqa: WPS433 (test amaçlı geç import)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"generate_dataset import edilemedi: {e}")
    return G


# --- Yardımcı işlevler (test dosyalarından import edilir) ----------------

import re  # noqa: E402  (yardımcılar aşağıda)

TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
EMP_RE = re.compile(r"EMP-\d+", re.IGNORECASE)
LV_RE = re.compile(r"LV-\d[\d-]*", re.IGNORECASE)
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
ISO_PERIOD_RE = re.compile(r"\b\d{4}-\d{2}\b")

DECISIONS = ("direct", "tool_call", "request_for_info", "cannot_answer")


def user_turns(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "user"]


def assistant_turns(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "assistant"]


def user_blob(messages: list[dict]) -> str:
    return "\n".join(user_turns(messages))


def strip_tool_calls(text: str) -> str:
    """Assistant metninden `<tool_call>…</tool_call>` bloklarını çıkarır (düz metin kalır)."""
    return TOOLCALL_RE.sub(" ", text)


def iter_tool_calls(messages: list[dict]):
    """(msg_index, parsed_obj) — tüm assistant turlarındaki tool_call blokları."""
    for idx, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        for raw in TOOLCALL_RE.findall(m["content"]):
            yield idx, json.loads(raw)


def has_tool_call(text: str) -> bool:
    return bool(TOOLCALL_RE.search(text))


def fold(s: str) -> str:
    """Basit Türkçe katlama — diyakritikler + küçük harf (generate_dataset.tr_fold ile uyumlu)."""
    s = s.replace("İ", "i").replace("I", "ı").lower()
    for a, b in (("ı", "i"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("ç", "c")):
        s = s.replace(a, b)
    return s


def loose(s: str) -> str:
    """fold + alfasayısal-dışı → boşluk (gevşek altdizi eşleşmesi için)."""
    return re.sub(r"[^a-z0-9]+", " ", fold(s)).strip()


# --- pytest kancaları ----------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: üreticiyi alt süreçte koşan yavaş testler")
    config.addinivalue_line("markers", "statistical: dağılım/oran temelli testler (üretim parametrelerine duyarlı)")
