# -*- coding: utf-8 -*-
"""
test_curriculum_and_difficulty.py — ZORLUK etiketlemesinin bütünlüğü (§25, §7)
=====================================================================

`difficulty` alanı gerçek karmaşıklığı yansıtmalı: `cok_zor` örnekler somut bir
zorluk kaynağına sahip olmalı; `kolay` örnekler gerçekten basit olmalı; ortalama
karmaşıklık skoru zorlukla monoton artmalı.

Karmaşıklık sinyalleri: tur sayısı, ilk-tur metin uzunluğu, hedef tool sayısı,
WRITE olması, eksik parametre sayısı, çok-adımlı zincir olması.

Kapsam
------
* `kolay`: her zaman 2 tur, tek tool, WRITE değil, çok turlu değil, ilk tur ≤ 135 krktr.
* `cok_zor`: en az bir zorluk kaynağı var (çok turlu ∨ 6-tur ∨ uzun ∨ çoklu-tool ∨ uzun-register).
* Ortalama karmaşıklık skoru: kolay < orta < zor < cok_zor (kesin monoton).
* `register == "uzun"` ilk turları, `register == "kisa"` ilk turlarından belirgin
  daha uzun (min uzun-uzunluk > max kisa-uzunluk).
* Çok-adımlı zincir örneklerinin TAMAMI `cok_zor`.
* WRITE + onay + çok turlu (`gen_confirmed`) örnekleri en az `zor`.
* 4 zorluk seviyesinin her biri her ana domain'de (meta/kapsanmayan hariç) temsil
  edilmiş — ya da en az 3'ü.
* `bump_difficulty` mantığı: çok turlu bir örnek asla `kolay` değildir; uzun bir
  ilk tur asla `kolay` değildir.
* Zorluk dağılımı bimodal çökmemiş: `orta`+`zor` toplamı ≥ %55 (gövde var).
"""
from __future__ import annotations

import statistics
from collections import Counter

import pytest

from conftest import user_turns

DIFF_ORDER = ["kolay", "orta", "zor", "cok_zor"]
LONG_CHARS = 135


def first_len(rec) -> int:
    return len(user_turns(rec["messages"])[0])


def complexity_score(rec, meta) -> int:
    return (
        meta["turns"]
        + len(meta.get("target_tools") or [])
        + (2 if first_len(rec) > LONG_CHARS else 0)
        + (1 if meta.get("is_write") else 0)
        + len(meta.get("missing_parameters") or [])
        + (2 if meta.get("chain") else 0)
    )


@pytest.fixture(scope="session")
def by_difficulty(paired):
    out: dict[str, list] = {d: [] for d in DIFF_ORDER}
    for rec, meta in paired:
        out.setdefault(meta["difficulty"], []).append((rec, meta))
    return out


# --------------------------------------------------------------------------
# kolay gerçekten kolay
# --------------------------------------------------------------------------

def test_easy_examples_are_genuinely_simple(by_difficulty):
    bad = []
    for rec, meta in by_difficulty["kolay"]:
        reasons = []
        if meta["turns"] != 2:
            reasons.append(f"{meta['turns']} tur")
        if meta["multi_turn"]:
            reasons.append("multi_turn")
        if len(meta.get("target_tools", [])) > 1:
            reasons.append("çoklu-tool")
        if meta.get("is_write"):
            reasons.append("WRITE")
        if first_len(rec) > LONG_CHARS:
            reasons.append("uzun ilk tur")
        if reasons:
            bad.append(f"{meta['id']}: {', '.join(reasons)}")
    assert not bad, "'kolay' etiketli ama karmaşık örnekler:\n  " + "\n  ".join(bad[:20])


# --------------------------------------------------------------------------
# cok_zor gerçekten zor
# --------------------------------------------------------------------------

def test_hardest_examples_have_an_explicit_complexity_source(by_difficulty):
    unexplained = []
    for rec, meta in by_difficulty["cok_zor"]:
        if (meta["multi_turn"] or meta["turns"] == 6 or first_len(rec) > LONG_CHARS
                or len(meta.get("target_tools", [])) > 1 or meta["register"] == "uzun"):
            continue
        unexplained.append(f"{meta['id']} ({meta['intent']}, {meta['register']}, {meta['turns']} tur)")
    assert not unexplained, "Zorluk kaynağı belirsiz 'cok_zor' örnekleri:\n  " + "\n  ".join(unexplained[:20])


def test_all_chains_are_labelled_cok_zor(paired):
    bad = [meta["id"] for _, meta in paired if meta.get("chain") and meta["difficulty"] != "cok_zor"]
    assert not bad, f"çok-adımlı zincir ama cok_zor değil: {bad}"


def test_confirmed_write_flows_are_at_least_hard(paired):
    for _, meta in paired:
        if meta.get("is_write") and meta.get("confirmation_required") and meta["decision"] == "tool_call":
            assert meta["difficulty"] in ("zor", "cok_zor"), (
                f"{meta['id']}: onaylı WRITE yürütmesi '{meta['difficulty']}' (en az 'zor' olmalı)"
            )


# --------------------------------------------------------------------------
# Monotonluk
# --------------------------------------------------------------------------

def test_mean_complexity_is_strictly_monotone(by_difficulty):
    means = {
        d: statistics.mean(complexity_score(rec, meta) for rec, meta in items)
        for d, items in by_difficulty.items() if items
    }
    ordered = [means[d] for d in DIFF_ORDER if d in means]
    assert all(a < b for a, b in zip(ordered, ordered[1:])), (
        f"ortalama karmaşıklık skoru kesin monoton değil: {means}"
    )


def test_mean_turn_count_is_monotone(by_difficulty):
    means = {d: statistics.mean(m["turns"] for _, m in items) for d, items in by_difficulty.items() if items}
    ordered = [means[d] for d in DIFF_ORDER if d in means]
    assert all(a <= b for a, b in zip(ordered, ordered[1:])), f"ortalama tur sayısı monoton değil: {means}"
    assert means["cok_zor"] - means["kolay"] >= 1.0, f"cok_zor/kolay tur farkı çok küçük: {means}"


# --------------------------------------------------------------------------
# Register ↔ uzunluk
# --------------------------------------------------------------------------

def test_long_register_is_longer_than_short_register(paired):
    uzun = [first_len(rec) for rec, m in paired if m["register"] == "uzun"]
    kisa = [first_len(rec) for rec, m in paired if m["register"] == "kisa"]
    if not (uzun and kisa):
        pytest.skip("uzun/kisa register örneği yok")
    assert min(uzun) > max(kisa), (
        f"'uzun' en kısası ({min(uzun)}) ≤ 'kisa' en uzunu ({max(kisa)}) — register etiketi uzunlukla çelişiyor"
    )
    assert statistics.mean(uzun) >= 2 * statistics.mean(kisa), (
        "'uzun' ortalaması 'kisa' ortalamasının en az 2 katı olmalı"
    )


# --------------------------------------------------------------------------
# bump_difficulty tutarlılığı
# --------------------------------------------------------------------------

def test_multi_turn_examples_are_never_easy(paired):
    bad = [m["id"] for _, m in paired if m["multi_turn"] and m["difficulty"] == "kolay"]
    assert not bad, f"çok turlu ama 'kolay': {bad}"


def test_long_first_turn_examples_are_never_easy(paired):
    bad = [m["id"] for rec, m in paired if first_len(rec) > LONG_CHARS and m["difficulty"] == "kolay"]
    assert not bad, f"uzun ilk tur ama 'kolay': {bad[:15]}"


# --------------------------------------------------------------------------
# Dağılım şekli
# --------------------------------------------------------------------------

def test_difficulty_distribution_has_a_body(all_meta):
    n = len(all_meta)
    dist = Counter(m["difficulty"] for m in all_meta)
    body = (dist["orta"] + dist["zor"]) / n
    assert body >= 0.55, f"orta+zor gövdesi yalnızca %{100*body:.1f} (< %55) — dağılım bimodal çökmüş olabilir"
    for d in DIFF_ORDER:
        assert dist[d] / n >= 0.05, f"'{d}' payı %{100*dist[d]/n:.1f} (< %5)"


def test_each_difficulty_spans_multiple_domains(all_meta):
    for d in DIFF_ORDER:
        domains = {m["domain"] for m in all_meta if m["difficulty"] == d} - {"meta", "kapsanmayan"}
        assert len(domains) >= 3, f"'{d}' yalnızca {len(domains)} fonksiyonel domain'de: {domains}"
