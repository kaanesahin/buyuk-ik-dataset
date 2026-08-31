# -*- coding: utf-8 -*-
"""
test_write_safety_state_machine.py — WRITE akışı biçimsel bir OTOMAT (§11, §12, §13)
=========================================================================

Değişiklik yaratan her işlem, aşağıdaki durum makinesinin GEÇERLİ bir yolu
olmalı — kısa devre yok:

    START ──(eksik param?)──▶ GATHER ──▶ CONFIRM_PENDING ──(onay)──▶ EXECUTED
      │                                    ▲
      └────(param tam)─────────────────────┘

Sabit değişmezler
-----------------
* EXECUTE (WRITE tool_call) yalnızca ``confirmation_required`` bir tool içindir.
* EXECUTE'ten önce konuşmada bir CONFIRM turu (soru işaretli, işlemi özetleyen)
  ve hemen ardından bir ONAY turu (olumlu) gelir.
* 2 turlu bir konuşma ASLA WRITE tool_call ile bitmez (onaysız yürütme yok).
* CONFIRM mesajı yürütülecek işlemi SOMUT anlatır: employee_id'nin numarası,
  talep_id, tarih(ler), yeni ücret/pozisyon/alan değeri CONFIRM metninde geçer.
* EXECUTE argümanları CONFIRM'in anlattığıyla birebir örtüşür.
* ONAY turu olumludur — "hayır / vazgeçtim / iptal / istemiyorum" içermez.
* GATHER turu (varsa) CONFIRM'den önce gelir ve tool_call içermez.
* Çok-adımlı zincirde (6 tur): 3. turda verilen değer 4. turdaki CONFIRM'de yankılanır.
"""
from __future__ import annotations

import re
from collections import defaultdict

import pytest

from conftest import fold, has_tool_call, iter_tool_calls

NEGATIVE_ACK_RE = re.compile(r"\bhayir\b|vazgec|iptal et|istemiyorum|\byapma\b|olmaz\b|dur ")


@pytest.fixture(scope="session")
def date_surfaces(gen):
    d: dict[str, list[str]] = defaultdict(list)
    for surf, b, e in (*gen.DATE_RANGES, *gen.MONTH_RANGES):
        d[b].append(surf)
        d[e].append(surf)
    return d


@pytest.fixture(scope="session")
def write_executions(paired, gen):
    """(record, meta) — is_write + decision=tool_call olan tüm örnekler."""
    out = [
        (rec, meta) for rec, meta in paired
        if meta.get("is_write") and meta["decision"] == "tool_call"
    ]
    if not out:
        pytest.skip("WRITE yürütme örneği yok")
    return out


def _assistant_indices(messages):
    return [i for i, m in enumerate(messages) if m["role"] == "assistant"]


def _confirm_turn(messages):
    """EXECUTE'ten önceki assistant turu = CONFIRM."""
    ai = _assistant_indices(messages)
    return messages[ai[-2]]["content"] if len(ai) >= 2 else None


# --------------------------------------------------------------------------
# Otomat yapısı
# --------------------------------------------------------------------------

def test_executed_tool_is_confirmation_required(write_executions, gen):
    for rec, meta in write_executions:
        for _, obj in iter_tool_calls(rec["messages"]):
            assert obj["name"] in gen.CONFIRMATION_REQUIRED, (
                f"{meta['id']}: '{obj['name']}' WRITE yürütüldü ama onay-gerektiren tool değil"
            )


def test_no_two_turn_conversation_ends_in_a_write(paired, gen):
    for rec, meta in paired:
        if len(rec["messages"]) != 2:
            continue
        for _, obj in iter_tool_calls(rec["messages"]):
            assert obj["name"] not in gen.CONFIRMATION_REQUIRED, (
                f"{meta['id']}: 2 turlu konuşma onaysız WRITE ile bitiyor ({obj['name']})"
            )


def test_execute_is_preceded_by_confirm_then_ack(write_executions):
    for rec, meta in write_executions:
        msgs = rec["messages"]
        assert msgs[-1]["role"] == "assistant" and has_tool_call(msgs[-1]["content"])
        # -2: ONAY (user), -3: CONFIRM (assistant)
        assert msgs[-2]["role"] == "user", f"{meta['id']}: EXECUTE'ten önce user (onay) turu yok"
        assert msgs[-3]["role"] == "assistant", f"{meta['id']}: onaydan önce CONFIRM (assistant) turu yok"
        assert "?" in msgs[-3]["content"], f"{meta['id']}: CONFIRM turu soru içermiyor"
        assert not has_tool_call(msgs[-3]["content"]), f"{meta['id']}: CONFIRM turunda tool_call var"


def test_ack_turn_is_affirmative(write_executions):
    bad = [
        meta["id"] for rec, meta in write_executions
        if NEGATIVE_ACK_RE.search(fold(rec["messages"][-2]["content"]))
    ]
    assert not bad, f"olumsuz onay turu içeren WRITE yürütmeleri: {bad}"


def test_gather_turns_precede_confirm_and_have_no_calls(write_executions):
    for rec, meta in write_executions:
        msgs = rec["messages"]
        # CONFIRM turu = -3; ondan önceki tüm assistant turları GATHER'dır
        for m in msgs[:-3]:
            if m["role"] == "assistant":
                assert not has_tool_call(m["content"]), f"{meta['id']}: GATHER turunda tool_call"
                assert "?" in m["content"], f"{meta['id']}: GATHER turu soru değil"


# --------------------------------------------------------------------------
# CONFIRM ↔ EXECUTE tutarlılığı
# --------------------------------------------------------------------------

def test_confirm_restates_the_concrete_operation(write_executions, date_surfaces):
    failures = []
    for rec, meta in write_executions:
        confirm = _confirm_turn(rec["messages"])
        confirm_f = fold(confirm)
        for _, obj in iter_tool_calls(rec["messages"]):
            a = obj["arguments"]
            if "employee_id" in a:
                num = a["employee_id"].split("-")[1]
                if not re.search(rf"(?<!\d){num}(?!\d)", confirm):
                    failures.append(f"{meta['id']}: employee_id {a['employee_id']} CONFIRM'de yok")
            if "talep_id" in a and a["talep_id"].lower() not in confirm.lower():
                failures.append(f"{meta['id']}: talep_id {a['talep_id']} CONFIRM'de yok")
            if "yeni_brut_ucret" in a and str(a["yeni_brut_ucret"]) not in confirm:
                failures.append(f"{meta['id']}: yeni ücret {a['yeni_brut_ucret']} CONFIRM'de yok")
            if "yeni_pozisyon" in a and a["yeni_pozisyon"] not in confirm:
                failures.append(f"{meta['id']}: yeni pozisyon CONFIRM'de yok")
            for k, v in a.items():
                if isinstance(v, str) and re.match(r"\d{4}-\d{2}-\d{2}$", v):
                    if fold(v) in confirm_f:
                        continue
                    if any(fold(s) in confirm_f for s in date_surfaces.get(v, [])):
                        continue
                    failures.append(f"{meta['id']}: tarih {v} CONFIRM metninde (yüzey olarak) yok")
    assert not failures, "CONFIRM işlemi somut anlatmıyor:\n  " + "\n  ".join(failures[:25])


def test_chain_provided_value_is_echoed_in_confirmation(paired):
    for rec, meta in paired:
        if not meta.get("chain"):
            continue
        provide = fold(rec["messages"][2]["content"])      # 3. tur: kullanıcı değeri verir
        confirm = fold(rec["messages"][3]["content"])       # 4. tur: assistant onay ister
        nums = set(re.findall(r"\d{3,}", provide))
        if nums:
            assert any(n in confirm for n in nums), (
                f"{meta['id']}: zincirde 3. turdaki değer {nums} 4. turdaki CONFIRM'de yankılanmıyor"
            )


def test_execute_args_do_not_exceed_what_confirm_described(write_executions):
    """EXECUTE, CONFIRM'de anılmayan yeni bir hedef/tutar EKLEMEZ (kapsam sızıntısı yok)."""
    for rec, meta in write_executions:
        confirm = fold(_confirm_turn(rec["messages"]))
        for _, obj in iter_tool_calls(rec["messages"]):
            for k, v in obj["arguments"].items():
                if k == "aciklama":
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    assert str(v) in confirm, f"{meta['id']}: sayısal arg {k}={v} CONFIRM'de anılmamış"


# --------------------------------------------------------------------------
# İki yönlü sayım
# --------------------------------------------------------------------------

def test_every_confirmation_ask_has_a_matching_execution_intent(paired, gen):
    """`request_for_info` + confirmation_required örnekleri ile onaylı WRITE
    yürütmeleri AYNI intent kümesinden gelir (akış çiftleri tutarlı)."""
    ask_intents = {m["intent"] for _, m in paired
                   if m["decision"] == "request_for_info" and m.get("confirmation_required")}
    exec_intents = {m["intent"] for _, m in paired
                    if m["decision"] == "tool_call" and m.get("confirmation_required")}
    only_ask = ask_intents - exec_intents
    only_exec = exec_intents - ask_intents
    assert not only_ask, f"onay isteyen ama hiç yürütülmeyen intent'ler: {only_ask}"
    assert not only_exec, f"yürütülen ama hiç onay istenmeyen intent'ler: {only_exec}"
