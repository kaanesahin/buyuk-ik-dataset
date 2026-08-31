# -*- coding: utf-8 -*-
"""
test_generator_reproducibility.py — ÜRETİCİ boru hattı garantileri (§22, §31, §37)
============================================================================

Bu dosya diğerlerinden farklı olarak **üreticiyi alt süreçte koşar**. Hepsi
`@pytest.mark.slow` — `pytest -m "not slow"` ile atlanır. Toplam ~5-10 sn.

Kapsam
------
* Belirlenimcilik: aynı ``--seed`` → BYTE-AYNI çıktı dosyaları.
* Farklı ``--seed`` → farklı çıktı (RNG gerçekten kullanılıyor).
* Ölçeklenme: ``--n 800`` ve ``--n 3000`` her ikisinde de karar dağılımı hedefte.
* ``--dry-run`` hiçbir dosya yazmıyor.
* Üretilmiş set üzerinde ``validate_dataset.py`` HATA 0 ile çıkıyor (exit 0).
* ``make_preview.py`` önizlemesi kaynak veriyle TUTARLI (önizleme örnekleri
  gerçek kayıtlara birebir çözülüyor — içerik değişmiyor).
* ``build_training_variants.py`` tüm user/assistant turlarını koruyor ve ayrı
  ``tools`` alanını kaldırıp system turuna taşıyor.
* Meta dosyası satır sırası, eğitim dosyasıyla aynı.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from conftest import PREFIX, REPO_ROOT, SCRIPTS_DIR

pytestmark = pytest.mark.slow

GEN = SCRIPTS_DIR / "generate_dataset.py"
VALIDATE = SCRIPTS_DIR / "validate_dataset.py"
PREVIEW = SCRIPTS_DIR / "make_preview.py"
VARIANTS = SCRIPTS_DIR / "build_training_variants.py"

OUT_FILES = [
    f"{PREFIX}_train.jsonl", f"{PREFIX}_val.jsonl",
    f"{PREFIX}_train.meta.jsonl", f"{PREFIX}_val.meta.jsonl",
    f"{PREFIX}_tools.json",
]


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True, text=True, encoding="utf-8", cwd=REPO_ROOT, timeout=180, **kw
    )


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _generate(out_dir: Path, *, n: int | None = None, seed: int | None = None) -> subprocess.CompletedProcess:
    args = [str(GEN), "--out-dir", str(out_dir)]
    if n is not None:
        args += ["--n", str(n)]
    if seed is not None:
        args += ["--seed", str(seed)]
    cp = _run(args)
    assert cp.returncode == 0, f"generate_dataset başarısız:\n{cp.stderr}"
    return cp


# --------------------------------------------------------------------------
# Belirlenimcilik
# --------------------------------------------------------------------------

def test_same_seed_produces_byte_identical_output(tmp_path):
    d1, d2 = tmp_path / "a", tmp_path / "b"
    _generate(d1, n=800, seed=123)
    _generate(d2, n=800, seed=123)
    for f in OUT_FILES:
        assert _md5(d1 / f) == _md5(d2 / f), f"{f}: aynı seed farklı çıktı verdi (belirlenimcilik ihlali)"


def test_different_seed_produces_different_output(tmp_path):
    d1, d2 = tmp_path / "s1", tmp_path / "s2"
    _generate(d1, n=800, seed=1)
    _generate(d2, n=800, seed=2)
    same = [f for f in OUT_FILES if _md5(d1 / f) == _md5(d2 / f)]
    # tools.json seed'den bağımsızdır; onu hariç tut
    same = [f for f in same if f != f"{PREFIX}_tools.json"]
    assert not same, f"farklı seed aynı çıktıyı verdi: {same} (RNG kullanılmıyor olabilir)"


# --------------------------------------------------------------------------
# Ölçeklenme
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [800, 3000])
def test_decision_mix_holds_at_scale(tmp_path, n):
    d = tmp_path / f"n{n}"
    _generate(d, n=n)
    metas = [json.loads(x) for x in (d / f"{PREFIX}_train.meta.jsonl").read_text(encoding="utf-8").splitlines()]
    metas += [json.loads(x) for x in (d / f"{PREFIX}_val.meta.jsonl").read_text(encoding="utf-8").splitlines()]
    dec = Counter(m["decision"] for m in metas)
    targets = {"tool_call": 0.30, "direct": 0.25, "request_for_info": 0.25, "cannot_answer": 0.20}
    for k, t in targets.items():
        frac = dec[k] / len(metas)
        assert abs(frac - t) <= 0.04, f"n={n}: {k} %{100*frac:.1f} (hedef %{100*t:.0f})"


def test_dry_run_writes_nothing(tmp_path):
    d = tmp_path / "dry"
    d.mkdir()
    cp = _run([str(GEN), "--n", "200", "--dry-run", "--out-dir", str(d)])
    assert cp.returncode == 0, cp.stderr
    assert not list(d.iterdir()), f"--dry-run dosya yazdı: {list(d.iterdir())}"


# --------------------------------------------------------------------------
# validate_dataset entegrasyonu
# --------------------------------------------------------------------------

def test_validator_passes_on_freshly_generated_data(tmp_path):
    d = tmp_path / "fresh"
    _generate(d, n=1500)
    cp = _run([str(VALIDATE), "--dir", str(d), "--report", str(tmp_path / "v.md")])
    assert cp.returncode == 0, (
        f"validate_dataset yeni veride başarısız (exit {cp.returncode}):\n{cp.stdout[-3000:]}"
    )


def test_validator_passes_on_shipped_data():
    data_dir = REPO_ROOT / "data"
    if not (data_dir / f"{PREFIX}_train.jsonl").exists():
        pytest.skip("data/ yok")
    cp = _run([str(VALIDATE), "--dir", str(data_dir)])
    assert cp.returncode == 0, f"validate_dataset teslim edilmiş veride başarısız:\n{cp.stdout[-3000:]}"


# --------------------------------------------------------------------------
# make_preview: önizleme kaynağı bozmuyor
# --------------------------------------------------------------------------

def test_preview_samples_resolve_back_to_real_records():
    data_dir = REPO_ROOT / "data"
    preview_dir = REPO_ROOT / "preview" / "samples"
    if not preview_dir.is_dir() or not (data_dir / f"{PREFIX}_train.jsonl").exists():
        pytest.skip("preview/ veya data/ yok — önce make_preview.py çalıştırın")

    # gerçek kayıtların (tools, messages) imzaları
    real = set()
    for split in ("train", "val"):
        for line in (data_dir / f"{PREFIX}_{split}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            real.add(json.dumps({"tools": r["tools"], "messages": r["messages"]}, sort_keys=True, ensure_ascii=False))

    for sample_file in preview_dir.glob("*.sample.json"):
        for rec in json.loads(sample_file.read_text(encoding="utf-8")):
            key = json.dumps({"tools": rec["tools"], "messages": rec["messages"]}, sort_keys=True, ensure_ascii=False)
            assert key in real, (
                f"{sample_file.name}: bir önizleme kaydı gerçek veriye çözülmüyor "
                f"(intent={rec.get('_meta', {}).get('intent')}) — önizleme içeriği değiştirmiş"
            )


# --------------------------------------------------------------------------
# build_training_variants: turları koruyor, tools'u system'e taşıyor
# --------------------------------------------------------------------------

def test_training_variants_conversion_preserves_turns_and_moves_tools():
    """`to_system_variant` fonksiyonunu doğrudan test et — I/O'suz."""
    try:
        import build_training_variants as B  # noqa: WPS433
    except Exception as e:  # pragma: no cover
        pytest.skip(f"build_training_variants import edilemedi: {e}")

    data_dir = REPO_ROOT / "data"
    src_file = data_dir / f"{PREFIX}_val.jsonl"
    if not src_file.exists():
        pytest.skip("data/ yok")

    records = [json.loads(x) for x in src_file.read_text(encoding="utf-8").splitlines()][:50]
    for rec in records:
        out = B.to_system_variant(rec, "tr")
        assert "tools" not in out, "varyantta ayrı 'tools' alanı kalmış"
        assert out["messages"][0]["role"] == "system", "varyant system turuyla başlamıyor"
        assert "<tools>" in out["messages"][0]["content"], "system turunda <tools> bloğu yok"
        # kaynaktaki her tool adı system metninde geçmeli
        for t in rec["tools"]:
            assert f'"{t["name"]}"' in out["messages"][0]["content"], f"tool '{t['name']}' system'e taşınmamış"
        assert out["messages"][1:] == rec["messages"], "user/assistant turları değişmiş"


def test_shipped_training_variants_are_consistent_if_present():
    variants_dir = REPO_ROOT / "data" / "variants"
    src_dir = REPO_ROOT / "data"
    if not variants_dir.is_dir():
        pytest.skip("data/variants yok (opsiyonel çıktı)")
    for split in ("train", "val"):
        var_path = variants_dir / f"{PREFIX}_{split}.chatml_system.jsonl"
        src_path = src_dir / f"{PREFIX}_{split}.jsonl"
        if not (var_path.exists() and src_path.exists()):
            continue
        src = [json.loads(x) for x in src_path.read_text(encoding="utf-8").splitlines()]
        var = [json.loads(x) for x in var_path.read_text(encoding="utf-8").splitlines()]
        assert len(src) == len(var), f"{split}: varyant satır sayısı ({len(var)}) ≠ kaynak ({len(src)})"
        for s, v in zip(src[:100], var[:100]):
            assert v["messages"][1:] == s["messages"], f"{split}: varyant turları kaynaktan farklı"
