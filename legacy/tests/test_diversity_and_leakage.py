# -*- coding: utf-8 -*-
"""
test_diversity_and_leakage.py — SIZINTI ve yüzey çeşitliliği (§7, §32, §31)
=======================================================================

Datasetin başarı kriteri "surface-form memorization" değil "intent
generalization". Bu, iki somut gereksinime indirgenir:

  1. Eğitim ve doğrulama bölmeleri ayrık olmalı (klasik + normalize imza düzeyinde).
  2. Aynı örneğin yalnızca ID/rakam değiştirilerek çoğaltılması yasak.

Kapsam
------
* Hiçbir konuşma (tüm kullanıcı turlarının birleşimi) birebir tekrar etmiyor.
* Train ↔ Val kesişimi: birebir kullanıcı metni düzeyinde 0.
* Train ↔ Val kesişimi: normalize imza (rakam + EMP-/LV- silinmiş) düzeyinde 0.
* Tüm dataset genelinde normalize-imza benzersizliği ≥ %98 (§32 "sadece EMP-ID
  değiştir" klonları üretilmez).
* Her (decision, intent) grubunda (n ≥ 12): birinci kullanıcı turunun normalize
  imza benzersizlik oranı ≥ 0.55.
* Aşırı sık ilk-4-kelime öneki yok (stil öneklerini hariç tut) — ≤ %6.
* `norm_sig` gerçekten rakam/ID'leri siliyor (regresyon koruması).
* Aynı intent, farklı register'larda temsil ediliyor.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from conftest import fold, user_turns

STYLE_PREFIXES = {
    "sayın", "sayin", "ilgili", "bilgi", "merhaba,", "ya", "abi", "şuna", "suna",
    "bir", "pardon", "önümüzdeki", "onumuzdeki", "yöneticimle", "yoneticimle",
    "muhasebeyle", "kafam", "bu", "aylık", "aylik", "konu:", "hocam", "bak",
    "şey,", "sey,", "sabahtan", "eşimle", "esimle", "i̇k", "ik", "bir toplantıya",
}


@pytest.fixture(scope="session")
def norm_sig(gen):
    return gen.norm_sig


def _convo_sig(norm_sig, messages):
    return norm_sig(" || ".join(user_turns(messages)))


# --------------------------------------------------------------------------
# Birebir tekrar
# --------------------------------------------------------------------------

def test_no_exact_duplicate_conversations(all_records):
    keys = [" || ".join(user_turns(r["messages"])) for r in all_records]
    dupes = {k for k, c in Counter(keys).items() if c > 1}
    assert not dupes, f"{len(dupes)} birebir tekrar eden konuşma (ilk: {next(iter(dupes))[:120]!r})"


# --------------------------------------------------------------------------
# Train ↔ Val sızıntısı
# --------------------------------------------------------------------------

def test_no_verbatim_train_val_overlap(train_records, val_records):
    tr = {" || ".join(user_turns(r["messages"])) for r in train_records}
    va = {" || ".join(user_turns(r["messages"])) for r in val_records}
    overlap = tr & va
    assert not overlap, f"{len(overlap)} birebir train/val kesişimi"


def test_no_normalized_signature_train_val_overlap(train_records, val_records, norm_sig):
    tr = {_convo_sig(norm_sig, r["messages"]) for r in train_records}
    va = {_convo_sig(norm_sig, r["messages"]) for r in val_records}
    overlap = tr & va
    assert not overlap, (
        f"{len(overlap)} normalize-imza train/val kesişimi — 'aynı cümle farklı ID' sızıntısı"
    )


# --------------------------------------------------------------------------
# Klon üretimi (§32)
# --------------------------------------------------------------------------

def test_global_normalized_signature_uniqueness(all_records, norm_sig):
    sigs = [_convo_sig(norm_sig, r["messages"]) for r in all_records]
    uniq = len(set(sigs))
    ratio = uniq / len(sigs)
    assert ratio >= 0.98, (
        f"normalize-imza benzersizliği %{100 * ratio:.1f} ({len(sigs) - uniq} klon) — "
        f"§32: yalnız ID değiştiren kopyalar"
    )


def test_per_intent_first_turn_diversity(paired, norm_sig):
    groups: dict[tuple, list[str]] = defaultdict(list)
    for rec, meta in paired:
        first_user = user_turns(rec["messages"])[0]
        groups[(meta["decision"], meta["intent"])].append(norm_sig(first_user))

    weak = []
    for (dec, intent), sigs in groups.items():
        if len(sigs) < 12:
            continue
        ratio = len(set(sigs)) / len(sigs)
        if ratio < 0.55:
            weak.append(f"{dec}/{intent}: %{100 * ratio:.0f} ({len(set(sigs))}/{len(sigs)})")
    assert not weak, "Düşük ilk-tur yüzey çeşitliliği:\n  " + "\n  ".join(weak)


def test_no_overconcentrated_opening_phrase(all_records):
    counts = Counter()
    for rec in all_records:
        toks = fold(user_turns(rec["messages"])[0]).split()
        if toks and toks[0] in STYLE_PREFIXES:
            continue
        counts[" ".join(toks[:4])] += 1
    n = len(all_records)
    hot = [(p, c) for p, c in counts.most_common(8) if c / n > 0.06]
    assert not hot, "Aşırı sık açılış kalıpları: " + ", ".join(f"{p!r} %{100*c/n:.1f}" for p, c in hot)


def test_same_intent_expressed_in_multiple_registers(paired):
    reg_by_intent: dict[str, set] = defaultdict(set)
    for _, meta in paired:
        reg_by_intent[meta["intent"]].add(meta["register"])
    thin = [i for i, regs in reg_by_intent.items()
            if sum(1 for _, m in paired if m["intent"] == i) >= 15 and len(regs) < 3]
    assert not thin, f"tek/iki register'a sıkışmış (n≥15) intent'ler: {thin}"


# --------------------------------------------------------------------------
# norm_sig regresyon koruması
# --------------------------------------------------------------------------

def test_norm_sig_strips_ids_and_digits(norm_sig):
    a = norm_sig("EMP-1042 için 15-20 Eylül 2026 tarihlerinde yıllık izin")
    b = norm_sig("EMP-9999 için 3-8 Ekim 2026 tarihlerinde yıllık izin")
    # ay adları farklı olduğu için tam eşit değil; ama ID ve gün sayıları düşmeli
    assert "emp" not in a and not any(ch.isdigit() for ch in a), f"norm_sig ID/rakam bırakıyor: {a!r}"
    assert "1042" not in a and "9999" not in b


def test_norm_sig_collapses_pure_id_variants(norm_sig):
    variants = [
        "EMP-1001 çalışanının yıllık izin bakiyesini göster",
        "EMP-2002 çalışanının yıllık izin bakiyesini göster",
        "EMP-3003 çalışanının yıllık izin bakiyesini göster",
    ]
    assert len({norm_sig(v) for v in variants}) == 1, (
        "norm_sig, yalnız-ID-farklı cümleleri aynı imzaya indirgemeli (klon tespiti buna dayanır)"
    )
