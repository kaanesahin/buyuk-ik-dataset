# -*- coding: utf-8 -*-
"""
test_tool_selection_discriminability.py — TOOL SEÇİMİ öğrenilebilir mi? (§16, §27)
=========================================================================

Datasetin amacı doğru tool'u ÇAĞIRMAK değil, **doğru tool'u SEÇMEYİ** öğretmektir.
Bunun için her örnekte:
  (a) doğru tool + aynı alandan çeldiriciler bulunmalı (teaching signal),
  (b) kullanıcı metni, çeldiriciler arasından doğru tool'a işaret etmeli —
      karıştırılabilir bir kardeşin AYIRT EDİCİ sinyali metinde olmamalı.

`get_izin_bakiyesi` ("kalan") ile `get_izin_gecmisi` ("kullandığım") gibi
sınırlar (§16) burada sıkı biçimde kontrol edilir: bir tool'u hedefleyen örnek,
kardeş tool'un dışlayıcı terimlerini İÇERMEZ.

Kapsam
------
* Her `tool_call` örneğinde çeldirici sayısı 3–8 (§16).
* `tool_call` örneklerinin ≥ %85'inde hedefin `CONFUSABLE` komşusu çeldirici olarak var.
* Çeldiriciler hedeften farklı (hedef, çeldirici kümesinde değil).
* **Ayırt edicilik**: tek-tool okuma çağrılarında, kardeş tool'un dışlayıcı sinyal
  terimleri kullanıcı metninde (kelime sınırıyla) HİÇ geçmiyor.
* Çoklu-tool örneklerinde her hedef `tools` listesinde tanımlı.
* Aynı `intent`, konuşmalar arası HEP aynı `target_tool`'a gidiyor (etiket kararlılığı).
* Benzer kelime / farklı niyet: `get_izin_bakiyesi` ve `get_izin_gecmisi`
  örnekleri, ilk-tur imzalarına göre ayrıştırılabilir (kesişim < %15).
"""
from __future__ import annotations

import re
from collections import defaultdict

import pytest

from conftest import fold, iter_tool_calls, user_turns

# EXCLUSIVE_SIGNALS[X] = "metinde geçerse hedefin X DEĞİL, X'in bir KARDEŞİ olduğunu
# gösteren terimler". Bir örnek X'i hedefliyorsa bu terimlerden HİÇBİRİ kullanıcı
# metninde bulunmamalı (§16: 'kalan izin' → get_izin_bakiyesi, 'kullandığım izinler'
# → get_izin_gecmisi gibi sınırlar). Not: göreli/stil ifadeleri değil, ayırt edici
# içerik terimleri.
EXCLUSIVE_SIGNALS: dict[str, list[str]] = {
    "get_izin_bakiyesi": ["gecmis", "kullandigi", "kullandigim", "daha once",
                          "dokumunu", "hareketleri", "en son hangi tarihlerde"],
    "get_izin_gecmisi": ["bakiye", "kalan yillik", "kalan izni", "ne kadar kaldi",
                         "birikmis", "kullanabilecegi"],
    "get_maas_bilgisi": ["bordro", "pusula"],
    "get_bordro": [],
    "get_prim_bilgisi": ["bordro", "yan hak"],
    "get_yan_haklar": ["prim", "bordro", "bonus"],
    "get_puantaj": ["fazla mesai", "ek calisma"],
    "get_mesai_bilgisi": ["puantaj", "devamsizlik", "giris cikis"],
    "get_izin_talebi_durumu": ["bakiye", "gecmis izin", "kalan izni"],
    "get_employee_info": ["yonetici kim", "amiri kim", "aktif mi"],
    "get_employee_status": ["pozisyonda", "ise giris tarihi", "yonetici kim"],
    "get_yonetici_bilgisi": ["pozisyonda calisiyor", "ise giris", "temel bilgi"],
}


def _wb(text: str, kw: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(kw)}(?![a-z])", text) is not None


@pytest.fixture(scope="session")
def tool_call_meta(all_meta):
    return [m for m in all_meta if m["decision"] == "tool_call"]


@pytest.fixture(scope="session")
def single_read_calls(paired):
    """Tek turlu, tek-tool, okuma (WRITE değil) tool_call örnekleri."""
    return [
        (rec, meta) for rec, meta in paired
        if meta["decision"] == "tool_call" and not meta["multi_turn"]
        and len(meta.get("target_tools", [])) == 1 and not meta.get("is_write")
    ]


# --------------------------------------------------------------------------
# Yapısal teaching signal
# --------------------------------------------------------------------------

def test_distractor_count_in_range(tool_call_meta):
    for m in tool_call_meta:
        targets = set(m["target_tools"] or [m["target_tool"]])
        names = {t["name"] for t in m["tools"]}
        distractors = names - targets
        assert 3 <= len(distractors) <= 8, (
            f"{m['id']}: {len(distractors)} çeldirici (3–8 beklenir, §16)"
        )


def test_target_is_present_and_distinct(tool_call_meta):
    for m in tool_call_meta:
        targets = set(m["target_tools"] or [m["target_tool"]])
        names = {t["name"] for t in m["tools"]}
        assert targets <= names, f"{m['id']}: hedef tool(lar) {targets - names} listede yok"


def test_most_examples_include_a_confusable_neighbour(tool_call_meta, gen):
    without = 0
    for m in tool_call_meta:
        targets = m["target_tools"] or [m["target_tool"]]
        names = {t["name"] for t in m["tools"]}
        neigh = set()
        for t in targets:
            neigh |= set(gen.CONFUSABLE.get(t, []))
        if not (neigh & names):
            without += 1
    frac = 1 - without / len(tool_call_meta)
    assert frac >= 0.85, (
        f"tool_call örneklerinin yalnızca %{100*frac:.1f}'inde karıştırılabilir komşu çeldirici var "
        f"(< %85) — tool seçimi öğretisi zayıf"
    )


# --------------------------------------------------------------------------
# Ayırt edicilik — kardeşin dışlayıcı sinyali metinde yok
# --------------------------------------------------------------------------

def test_no_sibling_exclusive_signal_in_user_text(single_read_calls):
    """Hedef X ise, X'in KENDİ dışlayıcı-sinyal listesindeki hiçbir terim (yani bir
    kardeşe işaret eden terimler) kullanıcı metninde bulunmamalı."""
    failures = []
    for rec, meta in single_read_calls:
        tool = meta["target_tool"]
        sibling_signals = EXCLUSIVE_SIGNALS.get(tool)
        if not sibling_signals:
            continue
        blob = fold(" ".join(user_turns(rec["messages"])))
        hit = [s for s in sibling_signals if _wb(blob, s)]
        if hit:
            failures.append(f"{meta['id']} → {tool}: kardeşe işaret eden {hit} kullanıcı metninde")
    assert not failures, (
        "Yanlış tool sinyali içeren örnekler (§16 sınır ihlali):\n  " + "\n  ".join(failures[:25])
    )


def test_balance_vs_history_are_surface_separable(paired, gen):
    """§16 açık örneği: 'kalan izin' ile 'kullandığım izinler' ayrı tool'lardır ve
    örnekleri ilk-tur imzalarına göre büyük ölçüde ayrışır."""
    def sigs(tool):
        return {
            gen.norm_sig(user_turns(rec["messages"])[0])
            for rec, meta in paired
            if meta["decision"] == "tool_call" and meta.get("target_tool") == tool
        }
    bal, hist = sigs("get_izin_bakiyesi"), sigs("get_izin_gecmisi")
    if len(bal) < 10 or len(hist) < 10:
        pytest.skip("yeterli örnek yok")
    overlap = len(bal & hist) / len(bal | hist)
    assert overlap < 0.15, (
        f"get_izin_bakiyesi ve get_izin_gecmisi ilk-tur imzaları %{100*overlap:.0f} kesişiyor (> %15)"
    )


# --------------------------------------------------------------------------
# Etiket kararlılığı
# --------------------------------------------------------------------------

def test_intent_maps_to_a_single_target_tool(paired):
    tools_per_intent: dict[str, set] = defaultdict(set)
    for _, meta in paired:
        if meta["decision"] == "tool_call" and meta.get("target_tool"):
            tools_per_intent[meta["intent"]].add(meta["target_tool"])
    unstable = {i: t for i, t in tools_per_intent.items() if len(t) > 1}
    assert not unstable, f"aynı intent farklı target_tool'lara gidiyor: {unstable}"


def test_intent_maps_to_a_single_domain(all_meta):
    dom_per_intent: dict[str, set] = defaultdict(set)
    for m in all_meta:
        dom_per_intent[m["intent"]].add(m["domain"])
    unstable = {i: d for i, d in dom_per_intent.items() if len(d) > 1}
    assert not unstable, f"aynı intent farklı domain'lerde: {unstable}"


def test_multi_intent_targets_are_declared(paired):
    for rec, meta in paired:
        if len(meta.get("target_tools", [])) < 2:
            continue
        names = {t["name"] for t in rec["tools"]}
        assert set(meta["target_tools"]) <= names, (
            f"{meta['id']}: multi-intent hedefleri {set(meta['target_tools']) - names} listede yok"
        )
        called = {obj["name"] for _, obj in iter_tool_calls(rec["messages"])}
        assert called == set(meta["target_tools"]), (
            f"{meta['id']}: çağrılan {called} ≠ hedef {set(meta['target_tools'])}"
        )
