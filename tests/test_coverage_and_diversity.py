# -*- coding: utf-8 -*-
"""Kapsama, çeşitlilik, tool-sonucu turu, hard-negative, opsiyonel parametre."""
from collections import Counter

from conftest import fold


def test_all_train_tools_covered(train_meta, catalog):
    train_tools = {t.name for t in catalog if t.split == "train"}
    tgt = Counter()
    for m in train_meta:
        for t in m.get("target_tools", []):
            tgt[t] += 1
    missing = train_tools - set(tgt)
    assert not missing, f"hiç hedeflenmemiş train tool: {missing}"
    assert min(tgt[t] for t in train_tools) >= 40, "bazı tool'lar çok az temsil edilmiş"
    assert max(tgt.values()) / min(tgt[t] for t in train_tools) < 6, "dağılım çok eğri"


def test_all_domains_present(train_meta, catalog):
    doms = {t.domain for t in catalog}
    seen = {m["domain"] for m in train_meta if m["domain"] in doms}
    assert seen == doms, f"eksik domain: {doms - seen}"


def test_decision_mix(train_meta):
    n = len(train_meta)
    d = Counter(m["decision"] for m in train_meta)
    assert 0.45 < d["tool_call"] / n < 0.65
    assert d["request_for_info"] / n > 0.15
    assert d["cannot_answer"] / n > 0.08
    assert d["direct"] / n > 0.08


def test_tool_result_turns_present_and_varied(train_meta):
    n = len(train_meta)
    tc = sum(1 for m in train_meta if m["decision"] == "tool_call")
    tr = sum(1 for m in train_meta if m.get("has_tool_result"))
    assert tr / tc > 0.25, f"tool-sonucu turu payı düşük: {tr/tc:.2f}"
    modes = Counter(m.get("tool_result_mode") for m in train_meta if m.get("has_tool_result"))
    for want in ("ok", "empty", "error", "partial"):
        assert modes[want] > 20, f"tool-sonucu modu '{want}' az: {modes[want]}"


def test_multi_tool_sequential_and_parallel(train_meta):
    par = sum(1 for m in train_meta if m.get("scenario") == "multi_parallel")
    seq = sum(1 for m in train_meta if m.get("sequential"))
    assert par > 200 and seq > 150, f"multi-tool az: par={par} seq={seq}"


def test_hard_negatives_present(train_meta):
    hn = Counter(m.get("hard_negative") for m in train_meta if m.get("hard_negative"))
    for want in ("A_keyword_ambiguous", "E_conflict", "F_tool_absent", "D_user_names_wrong_tool"):
        assert hn[want] > 100, f"hard-negative '{want}' az: {hn[want]}"


def test_optional_param_behaviour_taught(train_meta):
    """Kullanıcı opsiyonel bilgi verince model dolduruyor (Durum A)."""
    used = sum(1 for m in train_meta if m.get("optional_params_used"))
    assert used > 400, f"opsiyonel parametre kullanımı az: {used}"


def test_register_diversity(train_meta):
    reg = Counter(m["register"] for m in train_meta)
    assert len(reg) >= 6
    assert min(reg.values()) / len(train_meta) > 0.03, "bir register çok az"
    assert reg["typo"] / len(train_meta) > 0.08


def test_first_turn_lexical_diversity(train):
    u1 = [next(x["content"] for x in r["messages"] if x["role"] == "user") for r in train]
    uniq = len(set(fold(x) for x in u1))
    assert uniq / len(u1) > 0.9, f"ilk-tur çeşitliliği düşük: {uniq/len(u1):.2f}"


def test_direct_answer_not_overly_repeated(train, train_meta):
    ans = Counter(r["messages"][1]["content"] for r, m in zip(train, train_meta)
                  if m["decision"] == "direct")
    top = ans.most_common(1)[0][1]
    total = sum(ans.values())
    assert top / total < 0.06, f"bir direct cevap çok tekrar ediyor: {top}/{total}"


def test_train_val_no_leakage(train, val):
    def sig(recs):
        return {fold(" || ".join(x["content"] for x in r["messages"] if x["role"] == "user"))
                for r in recs}
    assert len(sig(train) & sig(val)) <= 3
