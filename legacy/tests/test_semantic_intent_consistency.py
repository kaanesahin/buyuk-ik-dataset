# -*- coding: utf-8 -*-
"""
test_semantic_intent_consistency.py — INTENT etiketinin anlamsal tutarlılığı (§7, §16)
==============================================================================

Aynı `intent` ile etiketlenmiş örnekler GERÇEKTEN aynı niyeti taşımalı; farklı
intent'ler ayrışmalı. Bu, "intent generalization" hedefinin (surface-form
memorization DEĞİL) etiket düzeyinde doğrulanmasıdır.

Yöntem: her intent için ilk-tur içerik-kelime kümeleri çıkarılır. Bir intent
içindeki ortalama ikili Jaccard benzerliği (kohezyon), intent'ler ARASI ortalama
Jaccard'dan (ayrışma) belirgin yüksek olmalı.

Kapsam
------
* Hiçbir iki FARKLI intent, birebir aynı ilk-tur metnini üretmiyor (etiket belirsizliği yok).
* Küresel: ortalama intra-intent Jaccard ≥ 3 × ortalama inter-intent Jaccard.
* Her intent (n ≥ 8) için: intra-Jaccard > inter-Jaccard + 0.02 (kendine benziyor).
* Her intent tek bir karar-kümesi kalıbına uyuyor: {direct} | {cannot_answer} |
  {tool_call} | {request_for_info} | {request_for_info, tool_call}.
* Intent adı ile hedef tool adı anlamsal olarak eşleşiyor (kürasyon tablosu).
* `{request_for_info, tool_call}` kalıbındaki intent'lerde: tool_call örneklerinde
  gerekli parametre metinde var, request_for_info örneklerinde yok (When2Call ayrımı
  intent içinde tutarlı uygulanmış).
* §16 sınırları: karıştırılabilir tool çiftlerine karşılık gelen intent çiftlerinin
  ilk-tur imza kesişimi < %20.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict

import pytest

from conftest import fold, user_turns

WORD_RE = re.compile(r"[a-z]{3,}")
STOPWORDS = set(
    "icin bir bu ile misin bana mi ne var yok kac gun lazim rica ederim iyi calismalar "
    "dilerim bilgi talebi ilgili birime iletilmek uzere merhaba asagidaki hususta "
    "bilgilendirilmek istiyorum saygilarimla konu gerekiyor sayin yetkili acele hemen "
    "pardon abi hocam numarali calisan personel kodlu sicil olarak gore".split()
)
MIN_N = 8

INTENT_TOOL = {
    "get_leave_balance": "get_izin_bakiyesi", "get_leave_history": "get_izin_gecmisi",
    "get_leave_request_status": "get_izin_talebi_durumu", "get_salary": "get_maas_bilgisi",
    "get_payslip": "get_bordro", "get_bonus": "get_prim_bilgisi", "get_benefits": "get_yan_haklar",
    "get_timesheet": "get_puantaj", "get_overtime": "get_mesai_bilgisi",
    "get_employee_info": "get_employee_info", "get_employee_status": "get_employee_status",
    "get_department_info": "get_departman_bilgisi", "list_department_employees": "get_calisan_listesi",
    "get_manager": "get_yonetici_bilgisi", "check_access": "check_employee_access",
    "create_leave_request": "create_izin_talebi", "cancel_leave_request": "cancel_izin_talebi",
    "update_leave_request": "update_izin_talebi", "update_contact": "update_employee_contact",
    "update_information": "update_employee_information",
    "create_salary_change": "create_ucret_degisiklik_talebi",
    "create_position_change": "create_pozisyon_degisiklik_talebi",
}

ALLOWED_DECISION_SETS = [
    frozenset({"direct"}), frozenset({"cannot_answer"}), frozenset({"tool_call"}),
    frozenset({"request_for_info"}), frozenset({"request_for_info", "tool_call"}),
]

CONFUSABLE_INTENT_PAIRS = [
    ("get_leave_balance", "get_leave_history"),
    ("get_leave_balance", "get_leave_request_status"),
    ("get_salary", "get_payslip"),
    ("get_payslip", "get_bonus"),
    ("get_timesheet", "get_overtime"),
    ("get_employee_info", "get_employee_status"),
    ("get_employee_info", "get_manager"),
    ("create_leave_request", "update_leave_request"),
    ("cancel_leave_request", "update_leave_request"),
]


def content_words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(fold(text)) if w not in STOPWORDS}


def _mean_jaccard(a_sets, b_sets, *, same: bool) -> float:
    total, n = 0.0, 0
    for i, A in enumerate(a_sets):
        for j, B in enumerate(b_sets):
            if same and j <= i:
                continue
            union = len(A | B)
            if union:
                total += len(A & B) / union
                n += 1
    return total / n if n else 0.0


@pytest.fixture(scope="session")
def first_turn_words(paired):
    out: dict[str, list[set]] = defaultdict(list)
    for rec, meta in paired:
        out[meta["intent"]].append(content_words(user_turns(rec["messages"])[0]))
    return out


@pytest.fixture(scope="session")
def decision_sets(all_meta):
    out: dict[str, set] = defaultdict(set)
    for m in all_meta:
        out[m["intent"]].add(m["decision"])
    return out


# --------------------------------------------------------------------------
# Etiket belirsizliği yok
# --------------------------------------------------------------------------

def test_no_two_intents_share_an_identical_first_turn(paired):
    seen: dict[str, str] = {}
    collisions = []
    for rec, meta in paired:
        ft = user_turns(rec["messages"])[0].strip()
        if ft in seen and seen[ft] != meta["intent"]:
            collisions.append(f"{ft[:70]!r} → {seen[ft]} & {meta['intent']}")
        seen[ft] = meta["intent"]
    assert not collisions, "Aynı ilk-tur, farklı intent (etiket belirsizliği):\n  " + "\n  ".join(collisions[:15])


# --------------------------------------------------------------------------
# Kohezyon > ayrışma
# --------------------------------------------------------------------------

def test_global_intra_intent_cohesion_dominates(first_turn_words):
    intents = [i for i, l in first_turn_words.items() if len(l) >= MIN_N]
    rng = random.Random(0)
    intra, inter = [], []
    for it in intents:
        lists = first_turn_words[it]
        intra.append(_mean_jaccard(lists, lists, same=True))
        others = [s for o in intents if o != it for s in first_turn_words[o]]
        rng.shuffle(others)
        inter.append(_mean_jaccard(lists, others[:60], same=False))
    mean_intra = sum(intra) / len(intra)
    mean_inter = sum(inter) / len(inter)
    assert mean_intra >= 3 * mean_inter, (
        f"intra-intent kohezyon ({mean_intra:.3f}) inter-intent ayrışmanın ({mean_inter:.3f}) 3 katı değil"
    )


def test_every_intent_is_more_similar_to_itself(first_turn_words):
    intents = [i for i, l in first_turn_words.items() if len(l) >= MIN_N]
    rng = random.Random(1)
    weak = []
    for it in intents:
        lists = first_turn_words[it]
        intra = _mean_jaccard(lists, lists, same=True)
        others = [s for o in intents if o != it for s in first_turn_words[o]]
        rng.shuffle(others)
        inter = _mean_jaccard(lists, others[:80], same=False)
        if intra <= inter + 0.02:
            weak.append(f"{it}: intra={intra:.3f} inter={inter:.3f} (n={len(lists)})")
    assert not weak, "Kendine yeterince benzemeyen intent'ler:\n  " + "\n  ".join(weak)


# --------------------------------------------------------------------------
# Etiket kalıpları
# --------------------------------------------------------------------------

def test_each_intent_follows_an_allowed_decision_pattern(decision_sets):
    bad = {
        i: sorted(s) for i, s in decision_sets.items()
        if frozenset(s) not in ALLOWED_DECISION_SETS
    }
    assert not bad, f"izin verilmeyen karar-kümesi kalıpları: {bad}"


def test_intent_name_matches_target_tool(paired):
    seen: dict[str, set] = defaultdict(set)
    for _, meta in paired:
        if meta.get("target_tool"):
            seen[meta["intent"]].add(meta["target_tool"])
    bad = []
    for intent, expected in INTENT_TOOL.items():
        if intent in seen and seen[intent] != {expected}:
            bad.append(f"{intent}: bekleniyor {expected}, bulundu {seen[intent]}")
    assert not bad, "Intent adı ↔ tool adı uyuşmazlığı:\n  " + "\n  ".join(bad)


# --------------------------------------------------------------------------
# When2Call ayrımı intent içinde tutarlı
# --------------------------------------------------------------------------

def test_dual_pattern_intents_split_by_parameter_presence(paired):
    """{request_for_info, tool_call} kalıbındaki intent'lerde: employee_id
    tool_call örneklerinde metinde var, request_for_info örneklerinde yok."""
    emp_ref = re.compile(r"emp-?\d+|sicil no \d+|\d{3,4} numar")
    by_intent: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for rec, meta in paired:
        by_intent[meta["intent"]][meta["decision"]].append(rec)

    failures = []
    for intent, buckets in by_intent.items():
        if set(buckets) != {"request_for_info", "tool_call"}:
            continue
        # yalnızca employee_id'nin zorunlu olduğu okuma intent'leri
        if intent not in {"get_leave_balance", "get_salary", "get_bonus", "get_benefits",
                          "get_leave_history", "get_leave_request_status", "get_manager"}:
            continue
        for rec in buckets["tool_call"]:
            if not emp_ref.search(fold(" ".join(user_turns(rec["messages"])))):
                failures.append(f"{intent}: tool_call örneğinde kimlik referansı yok")
                break
        for rec in buckets["request_for_info"]:
            if emp_ref.search(fold(" ".join(user_turns(rec["messages"])))):
                failures.append(f"{intent}: request_for_info örneğinde kimlik VAR (çağrılmalıydı)")
                break
    assert not failures, "\n  ".join(failures)


# --------------------------------------------------------------------------
# §16 sınırları — intent düzeyinde
# --------------------------------------------------------------------------

def test_confusable_intent_pairs_are_surface_separable(paired, gen):
    def sigs(intent):
        return {
            gen.norm_sig(user_turns(rec["messages"])[0])
            for rec, meta in paired if meta["intent"] == intent
        }
    problems = []
    for a, b in CONFUSABLE_INTENT_PAIRS:
        sa, sb = sigs(a), sigs(b)
        if len(sa) < 5 or len(sb) < 5:
            continue
        overlap = len(sa & sb) / len(sa | sb)
        if overlap >= 0.20:
            problems.append(f"{a} ↔ {b}: imza kesişimi %{100*overlap:.0f}")
    assert not problems, "Ayrışması zayıf intent çiftleri (§16):\n  " + "\n  ".join(problems)
