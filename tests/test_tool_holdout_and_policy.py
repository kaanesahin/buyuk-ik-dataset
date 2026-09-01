# -*- coding: utf-8 -*-
"""POLICY-seviyesi: tool holdout, genellenebilir karar mantığı, WRITE güvenliği."""
from collections import Counter, defaultdict

from conftest import fold


def test_train_never_targets_val_or_test_tools(train_meta, catalog):
    by = {t.name: t for t in catalog}
    leak = set()
    for m in train_meta:
        for t in m.get("target_tools", []):
            if by[t].split != "train":
                leak.add(t)
    assert not leak, f"train hedefinde holdout tool: {leak}"


def test_val_unseen_targets_val_tools_only(val_meta, catalog):
    by = {t.name: t for t in catalog}
    for m in val_meta:
        if m.get("eval_kind") == "val_unseen_tool":
            for t in m.get("target_tools", []):
                assert by[t].split == "val", f"val_unseen hedef {t} split={by[t].split}"


def test_hardeval_unseen_probes_use_test_tools(hard_eval_meta, catalog):
    by = {t.name: t for t in catalog}
    for m in hard_eval_meta:
        if m["probe"] in ("P1_unseen_tool", "P2_seen_intent_new_tool",
                          "P3_category_new_surface", "P9_tool_result"):
            for t in m.get("target_tools", []):
                assert by[t].split == "test", f"{m['probe']} hedef {t} split={by[t].split}"


def test_holdout_tools_still_appear_as_distractors(train, catalog):
    """val/test tool'ları eğitimde HEDEF değil ama ÇELDİRİCİ olarak görülür (şema öğrenilir)."""
    holdout = {t.name for t in catalog if t.split != "train"}
    seen = set()
    for r in train:
        seen |= {t["name"] for t in r["tools"]} & holdout
    assert len(seen) >= len(holdout) * 0.8, f"holdout tool'lar çeldirici olarak az görülüyor: {len(seen)}/{len(holdout)}"


def test_same_intent_both_call_and_ask(train_meta, catalog):
    """Aynı tool hem tool_call hem request_for_info hedefi -> 'param var/yok' kararı
    ifadeden bağımsız, genellenebilir bir politika. When2Call çekirdeği."""
    by_dec = defaultdict(set)
    for m in train_meta:
        for t in m.get("target_tools", []):
            by_dec[m["decision"]].add(t)
    both = by_dec["tool_call"] & by_dec["request_for_info"]
    withreq = {t.name for t in catalog if t.split == "train" and t.required and t.cat == "read"}
    assert len(both & withreq) >= 25, f"call+ask kontrastı olan tool az: {len(both & withreq)}"


def test_same_phrasing_different_decision(train, train_meta):
    """Ek kontrol: normalize ilk-tur çakışıp kararı farklı olan örnekler de var."""
    sig2dec = defaultdict(set)
    for r, m in zip(train, train_meta):
        u1 = next(x["content"] for x in r["messages"] if x["role"] == "user")
        s = "".join(c for c in fold(u1) if not c.isdigit())
        sig2dec[s].add(m["decision"])
    contrast = [s for s, d in sig2dec.items() if len(d) > 1]
    assert len(contrast) >= 8, f"aynı ifade/farklı karar kontrastı az: {len(contrast)}"


def test_write_never_without_confirmation(train, train_meta, calls_of, catalog):
    write = {t.name for t in catalog if t.cat in ("write", "action")}
    import re
    bad = []
    for r, m in zip(train, train_meta):
        msgs = r["messages"]
        for j, msg in enumerate(msgs):
            if msg["role"] != "assistant" or "<tool_call>" not in msg["content"]:
                continue
            for b in re.findall(r'"name":\s*"([^"]+)"', msg["content"]):
                if b in write:
                    pre_a = [x["content"] for x in msgs[:j] if x["role"] == "assistant"]
                    pre_u = [x["content"] for x in msgs[:j] if x["role"] == "user"]
                    conf = any(re.search(r"onay|devam edeyim mi|uygun mu|gerçekleştireceğim|üzereyim",
                                         fold(a)) for a in pre_a)
                    ack = any(re.search(r"evet|onayl|devam|olur|uygun|tabii|tamam", fold(u))
                              for u in pre_u[1:])
                    if not (conf and ack):
                        bad.append(m.get("id"))
    assert not bad, f"onaysız WRITE: {bad[:10]}"


def test_keyword_to_toolname_shortcut_is_low(train, train_meta, catalog):
    """K-1: ayırt edici yüzey kelimesi -> tool korelasyonu < %55."""
    by = {t.name: t for t in catalog}
    hit = tot = 0
    for r, m in zip(train, train_meta):
        tt = m.get("target_tools", [])
        if len(tt) != 1 or m["decision"] not in ("tool_call", "request_for_info"):
            continue
        t = by[tt[0]]
        if not t.disc_kw:
            continue
        u = fold(" ".join(x["content"] for x in r["messages"] if x["role"] == "user"))
        tot += 1
        if any(fold(k) in u for k in t.disc_kw):
            hit += 1
    assert hit / tot < 0.55, f"keyword->tool korelasyonu yüksek: {100*hit/tot:.0f}%"


def test_candidate_list_target_present_and_position_uniform(train, train_meta):
    pos = []
    for r, m in zip(train, train_meta):
        tt = m.get("target_tools", [])
        names = [t["name"] for t in r["tools"]]
        for t in tt:
            assert t in names or m["decision"] in ("direct", "cannot_answer")
        if len(tt) == 1 and tt[0] in names and len(names) > 3:
            pos.append(names.index(tt[0]) / (len(names) - 1))
    mean = sum(pos) / len(pos)
    assert 0.4 < mean < 0.6, f"hedef konumu ele veriyor: ort {mean:.2f}"


def test_candidate_size_buckets(train_meta):
    cc = [m["candidate_count"] for m in train_meta]
    big = sum(1 for c in cc if c >= 35) / len(cc)
    mid = sum(1 for c in cc if 13 <= c <= 34) / len(cc)
    assert big > 0.10, f"büyük aday liste payı düşük: {big:.2f}"
    assert mid > 0.35
    assert max(cc) >= 45
