# -*- coding: utf-8 -*-
"""Belirlenimcilik: aynı seed -> byte-aynı çıktı."""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_regen_byte_identical():
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_dataset.py"),
             "--n", "1500", "--val-seen", "120", "--val-unseen", "120", "--out", d],
            capture_output=True, text=True, cwd=str(ROOT))
        assert r.returncode == 0, r.stderr
        h1 = {}
        for f in Path(d).glob("*.jsonl"):
            h1[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        r2 = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_dataset.py"),
             "--n", "1500", "--val-seen", "120", "--val-unseen", "120", "--out", d],
            capture_output=True, text=True, cwd=str(ROOT))
        assert r2.returncode == 0
        for f in Path(d).glob("*.jsonl"):
            assert hashlib.sha256(f.read_bytes()).hexdigest() == h1[f.name], f.name


@pytest.mark.slow
def test_validator_passes():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_dataset.py")],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout[-2000:]


def test_catalog_scales_without_new_templates():
    """Yeni tool eklemek yalnız katalog satırı gerektirir: üretici tool'a özel
    CÜMLE ŞABLONU / cevap havuzu içermez. Tek istisna: `CHAINS` ilişki tablosu
    (sıralı çoklu-tool; sayısı sub-linear, item 11 gereği)."""
    src = (ROOT / "scripts" / "gen" / "scenarios.py").read_text(encoding="utf-8")
    from catalog import TOOLS
    m = __import__("re").search(r"CHAINS\s*=\s*\[(.*?)\n\]", src, __import__("re").S)
    chains_block = m.group(1) if m else ""
    outside = src.replace(chains_block, "")
    stray = [t.name for t in TOOLS if t.name in outside]
    assert not stray, f"scenarios.py CHAINS dışında tool adına özel kod: {stray}"
    # frames.py hiç tool adı içermemeli
    fsrc = (ROOT / "scripts" / "gen" / "frames.py").read_text(encoding="utf-8")
    assert not [t.name for t in TOOLS if t.name in fsrc]
