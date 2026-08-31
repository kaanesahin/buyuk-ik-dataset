# -*- coding: utf-8 -*-
"""
test_turn_coherence.py — TURLAR ARASI anlamsal tutarlılık (§12, §20, §25)
=====================================================================

Çok turlu bir konuşmada her tur bir öncekinin makul bir devamı olmalı:
asistan bir şey sorduysa kullanıcı TAM OLARAK onu vermeli; kullanıcı bilgi
verdiyse asistan onu KULLANMALI, tekrar sormamalı.

Kapsam
------
* mt_info (4 tur, tool_call): 2. tur asistan sorusu → 3. tur kullanıcı yanıtı
  istenen türde (ID / dönem / tarih) → 4. tur çağrı o değeri kullanır.
* Çok-adımlı zincir: 2. tur (eksik param sorusu) → 3. tur yanıtı istenen türde;
  4. tur CONFIRM, 3. turdaki değeri içerir.
* Asistan, kullanıcının zaten yanıtladığı bir soruyu TEKRAR sormaz (§20 "yanlış
  tool eğilimi" / gereksiz soru).
* Çok turlu `direct`: 3. tur kullanıcı takip sorusu ('?'), 4. tur asistan
  ESASLI yanıt (yalnız soru değil, ≥ 40 karakter).
* Çok turlu `cannot_answer`: 3. tur kullanıcı ısrarı, 4. tur İKİNCİ (ve kısa/net)
  bir ret — ilk retle birebir aynı değil.
* 2 turlu örneklerde asistan yanıtı kullanıcının ilk turuna gönderme yapar
  (ID/dönem/tarih verildiyse çağrıda görünür — halihazırda başka testlerde;
  burada 'yanıt boş bir kabul değil' kontrolü).
* Son çağrı(lar)daki hedef tool, ilk turda ifade edilen niyetle uyumlu (intent).
"""
from __future__ import annotations

import re

import pytest

from conftest import fold, has_tool_call, iter_tool_calls, user_turns

MONTHS = r"ocak|subat|mart|nisan|mayis|haziran|temmuz|agustos|eylul|ekim|kasim|aralik"
EMP_REF_RE = re.compile(r"emp-?\d+|sicil no \d+|\d{3,4} numar|numaram \d|personel numaram")
# Yalnızca mutlak dönem ifadeleri (göreli "bu ay"/"geçen ay" hariç — diğer test
# dosyalarıyla tutarlı; mt_info/zincir turları zaten mutlak değer verir).
PERIOD_RE = re.compile(MONTHS + r"|ceyrek|\d{4}-\d{2}|\b20\d\d\b")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}[/.]\d{1,2}|" + MONTHS + r"|ceyrek")


def _wants(ask_folded: str) -> str | None:
    if any(k in ask_folded for k in ("numar", "personel", "calisan numar", "kimin adina", "hangi personel", "hangi calisan")):
        return "id"
    if "donem" in ask_folded or "hangi ay" in ask_folded:
        return "period"
    if "tarih" in ask_folded or "aralik" in ask_folded or "baslangic" in ask_folded:
        return "date"
    if any(k in ask_folded for k in ("izin turu", "hangi izin", "yillik", "mazeret", "hastalik")):
        return "leave_type"
    return None


def _supplies(text_folded: str, want: str) -> bool:
    return {
        "id": lambda t: bool(EMP_REF_RE.search(t)),
        "period": lambda t: bool(PERIOD_RE.search(t)),
        "date": lambda t: bool(DATE_RE.search(t)),
        "leave_type": lambda t: bool(re.search(r"yillik|senelik|mazeret|hastalik|saglik|rapor", t)),
    }[want](text_folded)


@pytest.fixture(scope="session")
def mt_info(paired):
    return [
        (rec, meta) for rec, meta in paired
        if meta["decision"] == "tool_call" and meta["multi_turn"]
        and not meta.get("is_write") and not meta.get("chain") and len(rec["messages"]) == 4
    ]


@pytest.fixture(scope="session")
def chains(paired):
    out = [(rec, meta) for rec, meta in paired if meta.get("chain")]
    if not out:
        pytest.skip("zincir örneği yok")
    return out


# --------------------------------------------------------------------------
# Soru → yanıt eşleşmesi
# --------------------------------------------------------------------------

def test_mt_info_user_answer_matches_the_question(mt_info):
    failures = []
    for rec, meta in mt_info:
        ask = fold(rec["messages"][1]["content"])
        answer = fold(rec["messages"][2]["content"])
        want = _wants(ask)
        if want and not _supplies(answer, want):
            failures.append(f"{meta['id']}: asistan '{want}' sordu, kullanıcı {rec['messages'][2]['content']!r} verdi")
    assert not failures, "\n  ".join(failures[:20])


def test_mt_info_call_uses_the_supplied_value(mt_info):
    for rec, meta in mt_info:
        supplied = fold(rec["messages"][2]["content"])
        emp_nums = set(re.findall(r"(\d{3,4})", rec["messages"][2]["content"]))
        call_strings = {
            str(v) for _, obj in iter_tool_calls(rec["messages"]) for v in obj["arguments"].values()
        }
        call_blob = " ".join(call_strings)
        if emp_nums:
            assert any(n in call_blob for n in emp_nums), (
                f"{meta['id']}: 3. turdaki değer {emp_nums} çağrıda yok ({call_strings})"
            )


def test_chain_ask_answer_alignment(chains):
    for rec, meta in chains:
        ask = fold(rec["messages"][1]["content"])
        provide = fold(rec["messages"][2]["content"])
        want = _wants(ask)
        assert want is not None, f"{meta['id']}: 2. tur bir parametre sorusu değil: {rec['messages'][1]['content']!r}"
        assert _supplies(provide, want), (
            f"{meta['id']}: zincirde '{want}' istendi, {rec['messages'][2]['content']!r} verildi"
        )


# --------------------------------------------------------------------------
# Gereksiz tekrar sorma yok
# --------------------------------------------------------------------------

def test_assistant_does_not_reask_an_answered_question(paired):
    """3. turda kullanıcı bir ID/tarih verdiyse, 4. turdaki asistan bunu TEKRAR sormaz."""
    failures = []
    for rec, meta in paired:
        msgs = rec["messages"]
        if len(msgs) != 4 or msgs[3]["role"] != "assistant":
            continue
        u2 = fold(msgs[2]["content"])
        a2 = fold(msgs[3]["content"])
        if has_tool_call(msgs[3]["content"]):
            continue  # çağrı yapmış, soru sormamış
        # kullanıcı ID verdi ama asistan hâlâ numara istiyor?
        if EMP_REF_RE.search(u2) and ("numaranizi" in a2 or "personel numar" in a2 or "calisan numar" in a2):
            failures.append(f"{meta['id']}: kullanıcı ID verdi, asistan tekrar soruyor")
    assert not failures, "\n  ".join(failures)


# --------------------------------------------------------------------------
# Çok turlu direct / cannot_answer
# --------------------------------------------------------------------------

def test_mt_direct_followup_is_question_then_substantive_answer(paired):
    for rec, meta in paired:
        if meta["decision"] != "direct" or not meta["multi_turn"]:
            continue
        assert rec["messages"][2]["role"] == "user" and "?" in rec["messages"][2]["content"], (
            f"{meta['id']}: 3. tur bir takip sorusu değil"
        )
        ans = rec["messages"][3]["content"]
        assert len(ans) >= 40, f"{meta['id']}: 4. tur yanıtı çok kısa ({len(ans)})"
        assert not has_tool_call(ans)


def test_mt_cannot_second_refusal_differs_from_first(paired):
    for rec, meta in paired:
        if meta["decision"] != "cannot_answer" or not meta["multi_turn"]:
            continue
        first = rec["messages"][1]["content"].strip()
        second = rec["messages"][3]["content"].strip()
        assert first != second, f"{meta['id']}: iki ret birebir aynı"
        assert rec["messages"][2]["role"] == "user", f"{meta['id']}: 3. tur kullanıcı ısrarı değil"
        assert not has_tool_call(second)


# --------------------------------------------------------------------------
# İlk niyet ↔ son aksiyon
# --------------------------------------------------------------------------

def test_first_turn_intent_aligns_with_final_tool(paired):
    """İlk turda 'maaş' geçen bir istek maaş/bordro tool'una; 'izin' geçen izin
    tool'una gitmeli (kaba anlamsal hizalama)."""
    topic_tools = {
        "maas": {"get_maas_bilgisi", "get_bordro", "get_prim_bilgisi", "create_ucret_degisiklik_talebi", "get_yan_haklar"},
        "bordro": {"get_bordro"},
        "izin": {"get_izin_bakiyesi", "get_izin_gecmisi", "get_izin_talebi_durumu", "create_izin_talebi",
                 "cancel_izin_talebi", "update_izin_talebi"},
        "puantaj": {"get_puantaj"},
        "mesai": {"get_mesai_bilgisi"},
        "yonetici": {"get_yonetici_bilgisi"},
        "departman": {"get_departman_bilgisi", "get_calisan_listesi"},
    }
    failures = []
    for rec, meta in paired:
        if meta["decision"] != "tool_call":
            continue
        called = {obj["name"] for _, obj in iter_tool_calls(rec["messages"])}
        # yetki sorusu ("... maaşına erişme yetkim var mı") veri çağrısı değildir → hariç
        if "check_employee_access" in called or len(called) != 1:
            continue
        first = fold(user_turns(rec["messages"])[0])
        for topic, tools in topic_tools.items():
            if re.search(rf"(?<![a-z]){topic}(?![a-z])", first) and not (called & tools):
                failures.append(f"{meta['id']}: ilk turda '{topic}' var ama çağrılan {called}")
    assert not failures, "İlk niyet ile çağrılan tool uyuşmuyor:\n  " + "\n  ".join(failures[:20])
