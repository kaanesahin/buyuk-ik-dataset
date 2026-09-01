# -*- coding: utf-8 -*-
"""
test_conversation_flow.py — TUR MEKANİĞİ ve çok-adımlı akışlar (§12, §20, §25)
=========================================================================

Sohbet yapısı, chat template'inin bozulmadan tokenize edilebilmesi için katı
kurallara uymalı; çok turlu akışlar da anlamsal olarak doğru sıralanmalı.

Kapsam
------
* Katı rol alternasyonu: user, assistant, user, assistant, …
* İlk mesaj user, son mesaj assistant.
* Tur sayısı çift; tek turlu 2, çok turlu 4 veya 6.
* ``tool_call`` yalnızca en son assistant turunda.
* Çok-adımlı ZİNCİR (meta.chain): tam 6 tur; 6. tur tool_call; öncekilerde çağrı yok;
  4. tur (assistant) bir ONAY sorusu; 5. tur (user) bir ONAY.
* Onaylı WRITE akışı (is_write + multi_turn, chain değil): 4 tur; 2. tur onay sorusu.
* Çok turlu bilgi-toplama (mt_info): 4 tur; 2. tur eksik bilgi sorusu; 4. tur tool_call;
  3. turdaki kullanıcı bilgisi, çağrının argümanında görünür.
* Çok turlu ``direct`` / ``cannot_answer``: 4 tur, hiç tool_call yok.
* Meta ``multi_turn`` bayrağı gerçek tur sayısıyla uyumlu.
"""
from __future__ import annotations

import pytest

from conftest import fold, has_tool_call, iter_tool_calls


# --------------------------------------------------------------------------
# Genel tur mekaniği
# --------------------------------------------------------------------------

def test_strict_role_alternation(all_records):
    for i, rec in enumerate(all_records):
        roles = [m["role"] for m in rec["messages"]]
        for j in range(1, len(roles)):
            assert roles[j] != roles[j - 1], f"kayıt {i}: rol alternasyonu bozuk @ {j} ({roles})"


def test_starts_with_user_ends_with_assistant(all_records):
    for i, rec in enumerate(all_records):
        roles = [m["role"] for m in rec["messages"]]
        assert roles[0] == "user", f"kayıt {i}: ilk mesaj user değil"
        assert roles[-1] == "assistant", f"kayıt {i}: son mesaj assistant değil"


def test_turn_count_is_even_and_expected(all_records):
    for i, rec in enumerate(all_records):
        n = len(rec["messages"])
        assert n % 2 == 0, f"kayıt {i}: tek sayıda tur ({n})"
        assert n in (2, 4, 6), f"kayıt {i}: beklenmeyen tur sayısı ({n})"


def test_tool_call_only_in_last_assistant_turn(all_records):
    for i, rec in enumerate(all_records):
        msgs = rec["messages"]
        for j, m in enumerate(msgs):
            if m["role"] == "assistant" and has_tool_call(m["content"]):
                assert j == len(msgs) - 1, f"kayıt {i}: erken tool_call (mesaj {j}/{len(msgs)})"


def test_meta_multi_turn_flag_matches_turn_count(paired):
    for rec, meta in paired:
        n = len(rec["messages"])
        assert bool(meta["multi_turn"]) == (n > 2), (
            f"{meta['id']}: multi_turn={meta['multi_turn']} ama {n} tur var"
        )


def test_no_duplicate_adjacent_content(all_records):
    for i, rec in enumerate(all_records):
        contents = [m["content"].strip() for m in rec["messages"]]
        for j in range(1, len(contents)):
            assert contents[j] != contents[j - 1], f"kayıt {i}: {j}. tur bir öncekiyle aynı"


# --------------------------------------------------------------------------
# Çok-adımlı ZİNCİR (parametre topla → onay iste → uygula) — §25
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def chain_examples(paired):
    out = [(r, m) for r, m in paired if m.get("chain")]
    if not out:
        pytest.skip("çok-adımlı zincir örneği yok (meta.chain)")
    return out


def test_chain_examples_have_exactly_six_turns(chain_examples):
    for rec, meta in chain_examples:
        assert len(rec["messages"]) == 6, f"{meta['id']}: zincir {len(rec['messages'])} tur (6 bekleniyor)"


def test_chain_only_calls_a_tool_on_the_final_turn(chain_examples):
    for rec, meta in chain_examples:
        msgs = rec["messages"]
        assert has_tool_call(msgs[-1]["content"]), f"{meta['id']}: zincir tool_call ile bitmiyor"
        for m in msgs[:-1]:
            assert not has_tool_call(m["content"]), f"{meta['id']}: zincirde erken tool_call"


def test_chain_second_assistant_turn_is_a_confirmation_request(chain_examples):
    """4. mesaj (assistant): parametre alındıktan sonra YAZMA için onay sorusu."""
    for rec, meta in chain_examples:
        turn4 = fold(rec["messages"][3]["content"])
        assert "?" in rec["messages"][3]["content"], f"{meta['id']}: 4. tur soru değil"
        assert any(k in turn4 for k in ("onay", "devam", "ister misiniz", "uygun mu", "onayliyor")), (
            f"{meta['id']}: 4. tur bir onay sorusu gibi görünmüyor: {rec['messages'][3]['content']!r}"
        )


def test_chain_third_user_turn_is_an_acknowledgement(chain_examples, gen):
    ack_folded = {fold(w) for w in gen.ACK_WORDS}
    for rec, meta in chain_examples:
        u3 = fold(rec["messages"][4]["content"])
        assert any(a in u3 for a in ack_folded) or any(
            w in u3 for w in ("evet", "onayl", "tamam", "olur", "uygun", "devam", "yap")
        ), f"{meta['id']}: 5. tur (onay) gibi görünmüyor: {rec['messages'][4]['content']!r}"


def test_chain_first_assistant_turn_asks_for_the_missing_parameter(chain_examples):
    """2. mesaj (assistant): eksik parametreyi ister — henüz onay/çağrı yok."""
    for rec, meta in chain_examples:
        turn2 = rec["messages"][1]["content"]
        assert "?" in turn2, f"{meta['id']}: 2. tur eksik-bilgi sorusu değil"
        assert not has_tool_call(turn2)


def test_chain_final_call_arguments_are_complete(chain_examples):
    for rec, meta in chain_examples:
        by_name = {t["name"]: t for t in rec["tools"]}
        calls = list(iter_tool_calls(rec["messages"]))
        assert calls, f"{meta['id']}: zincirde çağrı yok"
        for _, obj in calls:
            req = set(by_name[obj["name"]]["parameters"].get("required", []))
            assert req <= set(obj["arguments"]), (
                f"{meta['id']}: zincir sonu çağrıda eksik zorunlu argüman {req - set(obj['arguments'])}"
            )


# --------------------------------------------------------------------------
# 4 turlu akışlar
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def four_turn(paired):
    return [(r, m) for r, m in paired if len(r["messages"]) == 4]


def test_confirmed_write_flow_is_four_turns(paired):
    for rec, meta in paired:
        if meta.get("is_write") and meta.get("confirmation_required") and meta["decision"] == "tool_call" and not meta.get("chain"):
            assert len(rec["messages"]) == 4, f"{meta['id']}: onaylı WRITE {len(rec['messages'])} tur (4 bekleniyor)"


def test_confirmed_write_flow_second_turn_is_confirmation(paired):
    for rec, meta in paired:
        if not (meta.get("confirmation_required") and meta["decision"] == "tool_call"
                and len(rec["messages"]) == 4 and not meta.get("chain")):
            continue
        turn2 = rec["messages"][1]["content"]
        assert "?" in turn2 and not has_tool_call(turn2), (
            f"{meta['id']}: onaylı WRITE 2. turu onay sorusu değil"
        )


def test_multi_turn_info_flow_carries_user_provided_value_into_call(paired):
    """mt_info: 3. turda kullanıcının verdiği EMP-ID, 4. turdaki çağrının argümanında görünür."""
    from conftest import EMP_RE

    checked = 0
    for rec, meta in paired:
        if not (meta["decision"] == "tool_call" and meta["multi_turn"]
                and not meta.get("is_write") and not meta.get("chain") and len(rec["messages"]) == 4):
            continue
        u2_ids = {m.upper() for m in EMP_RE.findall(rec["messages"][2]["content"])}
        if not u2_ids:
            continue
        call_ids = {
            str(obj["arguments"].get("employee_id", "")).upper()
            for _, obj in iter_tool_calls(rec["messages"])
        }
        assert u2_ids & call_ids, (
            f"{meta['id']}: 3. turdaki EMP-ID {u2_ids} çağrıda {call_ids} kullanılmadı"
        )
        checked += 1
    if checked == 0:
        pytest.skip("EMP-ID sağlayan mt_info örneği bulunamadı")


def test_four_turn_direct_and_cannot_have_no_tool_calls(paired):
    for rec, meta in paired:
        if len(rec["messages"]) == 4 and meta["decision"] in ("direct", "cannot_answer"):
            assert not any(True for _ in iter_tool_calls(rec["messages"])), (
                f"{meta['id']}: çok turlu {meta['decision']} içinde tool_call"
            )
            # 3. tur bir kullanıcı takibi / ısrarı olmalı
            assert rec["messages"][2]["role"] == "user"
