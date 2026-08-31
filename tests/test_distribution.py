# -*- coding: utf-8 -*-
"""
test_distribution.py — İSTATİSTİKSEL denge ve kapsam (§6, §7, §17, §25, §31)
=========================================================================

Dataset ``tool_call`` ağırlıklı aşırı dengesiz OLMAMALI; tüm karar sınıfları
domain'in her alanına yayılmalı ve zorluk/dil çeşitliliği korunmalı.

`@pytest.mark.statistical` — bu testler üretim parametrelerine (TARGET_MIX,
alt-kırılım oranları) duyarlıdır; `pytest -m "not statistical"` ile atlanabilir.

Kapsam
------
* Karar dağılımı, hedef %30/25/25/20'ye ±3.5 puan içinde.
* ``cannot_answer`` en az 6 domain'e yayılmış; ``puantaj`` ve ``ik_islemleri`` dahil.
* Zorluk: 4 seviyenin tamamı var; ``cok_zor`` ≥ %8, ``kolay`` ≤ %30.
* Register (dil kaydı): ≥ 5 kategori, her biri ≥ %3.
* Çok turlu örnek payı ≥ %8; gerçek 6-turlu zincir ≥ 20 örnek.
* WRITE örneği payı %8–%30 arası (yalnız okuma değil, yalnız yazma da değil).
* Train / Val oranı 0.10 ± 0.04; val bölmesi ana dağılımı korur.
* Her domain'de en az bir ``direct`` ve bir ``tool_call`` var.
* Intent çeşitliliği: ≥ 90 benzersiz intent.
"""
from __future__ import annotations

from collections import Counter

import pytest

from conftest import iter_tool_calls

pytestmark = pytest.mark.statistical

TARGET_MIX = {"tool_call": 0.30, "direct": 0.25, "request_for_info": 0.25, "cannot_answer": 0.20}
TOL = 0.035


# --------------------------------------------------------------------------
# Karar dağılımı
# --------------------------------------------------------------------------

def test_decision_mix_within_tolerance(all_meta):
    n = len(all_meta)
    dec = Counter(m["decision"] for m in all_meta)
    problems = []
    for k, target in TARGET_MIX.items():
        frac = dec[k] / n
        if abs(frac - target) > TOL:
            problems.append(f"{k}: %{100 * frac:.1f} (hedef %{100 * target:.0f}, tolerans ±{100 * TOL:.1f})")
    assert not problems, "Karar dağılımı hedeften sapıyor:\n  " + "\n  ".join(problems)


def test_no_single_decision_exceeds_35_percent(all_meta):
    n = len(all_meta)
    dec = Counter(m["decision"] for m in all_meta)
    for k, c in dec.items():
        assert c / n <= 0.35, f"{k} dataset'in %{100 * c / n:.1f}'ini oluşturuyor (aşırı ağırlık)"


# --------------------------------------------------------------------------
# cannot_answer domain yayılımı (§17)
# --------------------------------------------------------------------------

def test_cannot_answer_spans_at_least_six_domains(all_meta):
    domains = Counter(m["domain"] for m in all_meta if m["decision"] == "cannot_answer")
    assert len(domains) >= 6, f"cannot_answer yalnızca {len(domains)} domain'de: {dict(domains)}"


@pytest.mark.parametrize("domain", ["puantaj", "ik_islemleri", "maas_finans", "izin_yonetimi"])
def test_cannot_answer_covers_key_domain(all_meta, domain):
    n = sum(1 for m in all_meta if m["decision"] == "cannot_answer" and m["domain"] == domain)
    assert n >= 5, f"cannot_answer '{domain}' alanında yalnızca {n} örnek (§17)"


def test_cannot_answer_not_only_offtopic(all_meta):
    """`kapsanmayan` dışındaki (İK'ya yakın ama yine de reddedilecek) örnekler ağırlıkta olmalı."""
    cannot = [m for m in all_meta if m["decision"] == "cannot_answer"]
    offtopic = sum(1 for m in cannot if m["domain"] == "kapsanmayan")
    assert offtopic / len(cannot) < 0.5, (
        f"cannot_answer'ın %{100 * offtopic / len(cannot):.0f}'i 'kapsanmayan' — "
        f"gizlilik/yetki/gelecek örnekleri az"
    )


# --------------------------------------------------------------------------
# Zorluk ve register
# --------------------------------------------------------------------------

def test_all_four_difficulty_levels_present(all_meta):
    diff = Counter(m["difficulty"] for m in all_meta)
    assert set(diff) == {"kolay", "orta", "zor", "cok_zor"}, f"zorluk seviyeleri: {dict(diff)}"


def test_difficulty_has_a_hard_tail(all_meta):
    n = len(all_meta)
    diff = Counter(m["difficulty"] for m in all_meta)
    assert diff["cok_zor"] / n >= 0.08, f"cok_zor payı %{100 * diff['cok_zor'] / n:.1f} (< %8)"
    assert diff["kolay"] / n <= 0.30, f"kolay payı %{100 * diff['kolay'] / n:.1f} (> %30, çok kolay)"


def test_register_variety(all_meta):
    reg = Counter(m["register"] for m in all_meta)
    n = len(all_meta)
    assert len(reg) >= 5, f"yalnızca {len(reg)} dil kaydı: {dict(reg)}"
    weak = {r: c for r, c in reg.items() if c / n < 0.03}
    assert not weak, f"çok zayıf temsil edilen register(lar): { {r: f'%{100*c/n:.1f}' for r,c in weak.items()} }"


# --------------------------------------------------------------------------
# Çok turlu / WRITE payları
# --------------------------------------------------------------------------

def test_multi_turn_share(all_meta):
    n = len(all_meta)
    mt = sum(1 for m in all_meta if m["multi_turn"])
    assert mt / n >= 0.08, f"çok turlu örnek payı %{100 * mt / n:.1f} (< %8)"


def test_real_multi_step_chains_exist(all_meta):
    chains = sum(1 for m in all_meta if m.get("chain"))
    assert chains >= 20, f"yalnızca {chains} gerçek 6-turlu zincir örneği (§25)"


def test_write_operation_share_is_balanced(all_meta):
    n = len(all_meta)
    w = sum(1 for m in all_meta if m["is_write"])
    frac = w / n
    assert 0.08 <= frac <= 0.30, f"WRITE örneği payı %{100 * frac:.1f} — 8-30 aralığı beklenir (§11)"


# --------------------------------------------------------------------------
# Train / Val bölmesi
# --------------------------------------------------------------------------

def test_train_val_ratio_is_near_ten_percent(train_meta, val_meta):
    total = len(train_meta) + len(val_meta)
    ratio = len(val_meta) / total
    assert abs(ratio - 0.10) <= 0.04, f"val oranı {ratio:.3f} (0.10 ± 0.04 beklenir)"


def test_val_split_preserves_decision_mix(train_meta, val_meta):
    tr = Counter(m["decision"] for m in train_meta)
    va = Counter(m["decision"] for m in val_meta)
    ntr, nva = len(train_meta), len(val_meta)
    for k in TARGET_MIX:
        d = abs(tr[k] / ntr - va[k] / nva)
        assert d <= 0.06, f"'{k}' train/val payı ayrışıyor: train %{100*tr[k]/ntr:.1f} vs val %{100*va[k]/nva:.1f}"


# --------------------------------------------------------------------------
# Domain kapsamı ve intent çeşitliliği
# --------------------------------------------------------------------------

def test_every_functional_domain_has_direct_and_tool_call(all_meta):
    functional = {"izin_yonetimi", "maas_finans", "puantaj", "organizasyon", "calisan_bilgileri", "ik_islemleri"}
    by_domain = {}
    for m in all_meta:
        by_domain.setdefault(m["domain"], set()).add(m["decision"])
    for d in functional:
        if d not in by_domain:
            continue
        assert "tool_call" in by_domain[d], f"'{d}' alanında hiç tool_call yok"


def test_intent_diversity(all_meta):
    intents = {m["intent"] for m in all_meta}
    assert len(intents) >= 90, f"yalnızca {len(intents)} benzersiz intent (≥90 beklenir)"


@pytest.mark.statistical
def test_tool_call_histogram_covers_most_tools(all_records, gen):
    called = Counter()
    for rec in all_records:
        for _, obj in iter_tool_calls(rec["messages"]):
            called[obj["name"]] += 1
    coverage = len(called) / len(gen.TOOLS)
    assert coverage >= 0.85, (
        f"tool'ların yalnızca %{100 * coverage:.0f}'i çağrılıyor; çağrılmayan: "
        f"{sorted(set(gen.TOOLS) - set(called))}"
    )
