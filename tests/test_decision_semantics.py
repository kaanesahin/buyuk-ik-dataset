# -*- coding: utf-8 -*-
"""
test_decision_semantics.py — DÖRT KARAR davranışının doğruluğu (§4, §35)
=====================================================================

Datasetin asıl amacı: ``direct`` / ``tool_call`` / ``request_for_info`` /
``cannot_answer`` ayrımını öğretmek. Her örneğin assistant çıktısı, meta'daki
``decision`` etiketiyle DAVRANIŞSAL olarak tutarlı olmalı.

Kapsam
------
* ``direct``           → hiç tool_call yok; cevap somut bilgi içerir (yalnız soru değil).
* ``tool_call``        → son assistant mesajı ≥1 tool_call ile biter.
* ``request_for_info`` → hiç tool_call yok; son mesaj bilgi/onay İSTEYEN bir soru.
* ``cannot_answer``    → hiç tool_call yok; son mesaj kibar bir RET + (çoğunlukla) gerekçe.
* WRITE + onay gerekli (``confirmation_required``): tek turlu istekte önce onay
  sorulur (``request_for_info``), asla doğrudan yazılmaz.
* WRITE + onay verilmiş (çok turlu ``tool_call``): son kullanıcı turu bir ONAY,
  ardından tool_call gelir.
* Meta ``is_write`` / ``confirmation_required``, çağrılan tool'un politikasıyla uyumlu.
* Aynı intent, yeterli bilgi varsa ``tool_call``, eksikse ``request_for_info`` —
  ikisinin de örneği var (When2Call ayrımı).
"""
from __future__ import annotations

import re

import pytest

from conftest import (
    TOOLCALL_RE, assistant_turns, has_tool_call, iter_tool_calls, user_turns, fold,
)

REQUEST_LEMMAS = (
    "?", "misiniz", "musunuz", "misin", "musun", "paylasir", "paylasabilir",
    "iletir", "iletebilir", "belirt", "verir mi", "verebilir mi", "ihtiyacim var",
    "gerekiyor", "hangi ", "onayliyor", "onayiniz", "ister misiniz", "yazarsaniz",
    "soyler misiniz",
)

# Türkçe olumsuz yeterlik/olamama kalıpları + reddetme/gizlilik anahtarları
# (fold'lanmış metinde aranır).
REFUSAL_RE = re.compile(
    r"m[iu]yor"            # -mıyor / -miyor / -muyor  (üretemiyorum, sağlamıyor, atamıyorum)
    r"|[ae]m[ae]m\b"       # -amam / -emem  (yapamam, veremem, söyleyemem, kestiremem)
    r"|[ae]meyiz\b"        # -emeyiz
    r"|[ae]mez\b"          # -emez / -amaz
    r"|olmaz\b"            # 'doğru olmaz'
    r"|degil\b"            # 'uygun değil', 'arasında değil'
    r"|\byok\b|yoktur"     # 'aracım yok', 'erişimi yoktur'
    r"|disinda"            # 'kapsamının dışında'
    r"|spekulasyon"
    r"|kapsam"
    r"|reddet"
    r"|geri cevir"         # 'geri çevirmem gerekiyor'
    r"|tahminde bulunmuyorum"
    # gizlilik/yetki temelli retler (olumsuz fiil içermeyebilir):
    r"|sahibinin gorebilec"   # 'yalnızca sahibinin görebileceği bir kayıt'
    r"|gizli bir bilgi"
    r"|ozel yetki"
    r"|kisisel ver"           # 'kişisel verilerin korunması'
    r"|yetkili ik"            # 'yetkili İK ekibine açıktır'
)


@pytest.fixture(scope="session")
def by_decision(paired):
    buckets = {"direct": [], "tool_call": [], "request_for_info": [], "cannot_answer": []}
    for rec, meta in paired:
        buckets.setdefault(meta["decision"], []).append((rec, meta))
    return buckets


# --------------------------------------------------------------------------
# decision ↔ tool_call varlığı
# --------------------------------------------------------------------------

def test_meta_decision_is_one_of_four(all_meta):
    allowed = {"direct", "tool_call", "request_for_info", "cannot_answer"}
    bad = {m["decision"] for m in all_meta} - allowed
    assert not bad, f"beklenmeyen decision değer(ler)i: {bad}"


@pytest.mark.parametrize("decision", ["direct", "request_for_info", "cannot_answer"])
def test_non_tool_decisions_never_contain_tool_calls(by_decision, decision):
    offenders = [
        m["id"] for rec, m in by_decision[decision]
        if any(True for _ in iter_tool_calls(rec["messages"]))
    ]
    assert not offenders, f"{decision} ama tool_call içeren örnekler: {offenders[:20]}"


def test_tool_call_decision_ends_with_a_call(by_decision):
    offenders = [
        m["id"] for rec, m in by_decision["tool_call"]
        if not has_tool_call(rec["messages"][-1]["content"])
    ]
    assert not offenders, f"tool_call ama son mesajda çağrı olmayan örnekler: {offenders[:20]}"


# --------------------------------------------------------------------------
# request_for_info — son mesaj gerçekten bilgi/onay isteyen bir soru
# --------------------------------------------------------------------------

def test_request_for_info_last_turn_is_a_question(by_decision):
    weak = []
    for rec, meta in by_decision["request_for_info"]:
        last = fold(rec["messages"][-1]["content"])
        if not any(lem in last for lem in REQUEST_LEMMAS):
            weak.append((meta["id"], rec["messages"][-1]["content"][:90]))
    assert not weak, f"bilgi/onay isteği gibi görünmeyen request_for_info sonları: {weak[:20]}"


def test_request_for_info_does_not_answer_the_question(by_decision):
    """Model eksik bilgi isterken, cevabı da vermemeli (yarım tool sonucu sızdırmamalı)."""
    for rec, meta in by_decision["request_for_info"]:
        last = rec["messages"][-1]["content"]
        assert len(last) < 400, (
            f"{meta['id']}: request_for_info son mesajı beklenmedik kadar uzun ({len(last)} krktr) "
            f"— muhtemelen soru yerine açıklama"
        )


# --------------------------------------------------------------------------
# cannot_answer — kibar ret
# --------------------------------------------------------------------------

def test_cannot_answer_last_turn_reads_as_a_refusal(by_decision):
    weak = []
    for rec, meta in by_decision["cannot_answer"]:
        last = fold(rec["messages"][-1]["content"])
        if not REFUSAL_RE.search(last):
            weak.append((meta["id"], rec["messages"][-1]["content"][:90]))
    assert not weak, f"ret gibi görünmeyen cannot_answer sonları: {weak[:15]}"


def test_cannot_answer_does_not_leak_tool_names_as_if_used(by_decision):
    """Ret ederken 'get_maas_bilgisi çağırdım' gibi ifade olmamalı."""
    for rec, meta in by_decision["cannot_answer"]:
        for a in assistant_turns(rec["messages"]):
            assert "<tool_call>" not in a, f"{meta['id']}: cannot_answer içinde <tool_call>"


# --------------------------------------------------------------------------
# direct — somut bilgi içerir
# --------------------------------------------------------------------------

def test_direct_answers_are_substantive(by_decision, all_meta):
    meta_by_id = {m["id"]: m for m in all_meta}
    short = []
    for rec, meta in by_decision["direct"]:
        # selamlaşma/teşekkür/veda kısa olabilir; onları hariç tut
        if meta["intent"] in {"greeting", "thanks", "farewell"}:
            continue
        first_answer = next(m["content"] for m in rec["messages"] if m["role"] == "assistant")
        if len(first_answer) < 60:
            short.append((meta["id"], first_answer))
    assert not short, f"içerik bakımından zayıf direct cevaplar: {short[:15]}"


def test_direct_answers_are_not_just_questions(by_decision):
    for rec, meta in by_decision["direct"]:
        if meta["intent"] in {"greeting", "thanks", "farewell"}:
            continue
        first_answer = next(m["content"] for m in rec["messages"] if m["role"] == "assistant")
        # bir cümle bilgi vermeli — sadece '?' ile biten tek cümle değil
        assert not (first_answer.count("?") >= 1 and len(first_answer) < 50), (
            f"{meta['id']}: direct cevap yalnızca soru gibi görünüyor: {first_answer!r}"
        )


# --------------------------------------------------------------------------
# WRITE onay politikası
# --------------------------------------------------------------------------

def test_write_confirmation_asked_before_execution_in_single_turn_requests(paired, gen):
    """Tek turlu WRITE isteği (2 mesaj) → tool_call DEĞİL, önce onay sorusu."""
    for rec, meta in paired:
        if not meta.get("confirmation_required"):
            continue
        if meta["turns"] != 2:
            continue
        assert meta["decision"] == "request_for_info", (
            f"{meta['id']}: tek turlu onay-gerektiren WRITE ama decision={meta['decision']}"
        )
        assert not has_tool_call(rec["messages"][-1]["content"]), (
            f"{meta['id']}: onay alınmadan tool_call yapılmış"
        )


def test_confirmed_write_flow_has_ack_before_call(paired, gen):
    """Çok turlu onaylı WRITE: son kullanıcı turu ONAY, ardından tool_call."""
    ack_folded = {fold(w) for w in gen.ACK_WORDS}
    for rec, meta in paired:
        if not (meta.get("is_write") and meta.get("confirmation_required") and meta["decision"] == "tool_call"):
            continue
        msgs = rec["messages"]
        assert msgs[-1]["role"] == "assistant" and has_tool_call(msgs[-1]["content"])
        last_user = fold(msgs[-2]["content"])
        looks_like_ack = (
            msgs[-2]["role"] == "user"
            and (any(a in last_user for a in ack_folded) or _affirmative(last_user))
        )
        assert looks_like_ack, f"{meta['id']}: tool_call öncesi kullanıcı turu onay gibi değil: {msgs[-2]['content']!r}"


def _affirmative(folded_text: str) -> bool:
    return any(w in folded_text for w in ("evet", "onayl", "tamam", "olur", "uygun", "devam", "yap"))


def test_meta_is_write_matches_called_tool_policy(paired, gen):
    for rec, meta in paired:
        called = {obj["name"] for _, obj in iter_tool_calls(rec["messages"])}
        if meta["decision"] == "tool_call" and called:
            expect_write = bool(called & gen.WRITE_TOOLS)
            assert bool(meta.get("is_write")) == expect_write, (
                f"{meta['id']}: meta.is_write={meta.get('is_write')} ama çağrılan {called}"
            )


# --------------------------------------------------------------------------
# When2Call ayrımı: aynı intent hem tool_call hem request_for_info
# --------------------------------------------------------------------------

def test_key_intents_have_both_call_and_ask_examples(paired):
    """`get_leave_balance` gibi intent'ler hem yeterli-bilgi hem eksik-bilgi örneği içermeli."""
    seen = {}
    for _, meta in paired:
        seen.setdefault(meta["intent"], set()).add(meta["decision"])
    for intent in ("get_leave_balance", "get_salary", "create_leave_request"):
        if intent not in seen:
            continue
        assert {"tool_call", "request_for_info"} <= seen[intent], (
            f"'{intent}' yalnızca {seen[intent]} kararlarında var — When2Call ayrımı eksik"
        )
