# -*- coding: utf-8 -*-
"""
test_tool_calls.py — assistant TOOL ÇAĞRILARININ şemaya uygunluğu (§21, §31)
=========================================================================

`decision == tool_call` olan her örnekte assistant, kullanılabilir bir tool'u
DOĞRU parametrelerle çağırmalı.

Kapsam
------
* Tool çağrı biçimi TAM: ``<tool_call>\\n{json}\\n</tool_call>`` (Qwen şablonu).
* Blok içi JSON ayrıştırılabilir; ``name`` (str) + ``arguments`` (dict).
* Çağrılan tool, konuşmanın ``tools`` listesinde tanımlı.
* Argüman anahtarları ⊆ şema ``properties``.
* Tüm ``required`` argümanlar mevcut.
* ``enum`` argümanları yalnızca izin verilen değerleri alıyor.
* Tip uyumu: ``string`` → str, ``number`` → int/float (bool değil).
* Çoklu çağrı = art arda birden fazla blok; hepsi son assistant turunda.
* Meta ``target_tool`` / ``target_tools`` gerçekte çağrılanlarla tutarlı.
* Her WRITE tool'u en az bir kez çağrılıyor (kapsam bütünlüğü).
"""
from __future__ import annotations

import re

import pytest

from conftest import TOOLCALL_RE, has_tool_call, iter_tool_calls

STRICT_BLOCK_RE = re.compile(r"<tool_call>\n\{.*\}\n</tool_call>")


@pytest.fixture(scope="session")
def tool_call_records(paired):
    """(record, meta) — yalnızca tool_call kararlı örnekler."""
    out = [(r, m) for r, m in paired if m["decision"] == "tool_call"]
    if not out:
        pytest.skip("tool_call örneği yok")
    return out


# --------------------------------------------------------------------------
# Biçim
# --------------------------------------------------------------------------

def test_toolcall_block_format_is_canonical(tool_call_records):
    """Blok tam olarak <tool_call>\\n{...}\\n</tool_call> biçiminde olmalı."""
    for rec, meta in tool_call_records:
        final = rec["messages"][-1]["content"]
        blocks = re.findall(r"<tool_call>.*?</tool_call>", final, re.DOTALL)
        assert blocks, f"{meta['id']}: son mesajda tool_call bloğu yok"
        for b in blocks:
            assert STRICT_BLOCK_RE.fullmatch(b), (
                f"{meta['id']}: tool_call bloğu kanonik biçimde değil:\n{b!r}"
            )


def test_toolcall_json_parses_and_has_name_and_arguments(all_records):
    for i, rec in enumerate(all_records):
        for _, obj in iter_tool_calls(rec["messages"]):
            assert isinstance(obj.get("name"), str) and obj["name"], f"kayıt {i}: tool_call 'name' yok/str değil"
            assert isinstance(obj.get("arguments"), dict), f"kayıt {i}: tool_call 'arguments' dict değil"
            assert set(obj) <= {"name", "arguments"}, f"kayıt {i}: tool_call fazladan anahtar {set(obj)}"


def test_toolcall_appears_only_in_final_assistant_message(all_records):
    for i, rec in enumerate(all_records):
        msgs = rec["messages"]
        for j, m in enumerate(msgs):
            if m["role"] == "assistant" and has_tool_call(m["content"]) and j != len(msgs) - 1:
                pytest.fail(f"kayıt {i}: tool_call son olmayan assistant mesajında (mesaj {j})")


def test_toolcall_prose_is_not_mixed_with_block(tool_call_records):
    """§39: tool çağrısından önce gereksiz açıklama metni olmamalı."""
    for rec, meta in tool_call_records:
        final = rec["messages"][-1]["content"].strip()
        without_blocks = TOOLCALL_RE.sub("", final).strip()
        assert without_blocks == "", (
            f"{meta['id']}: tool_call turunda serbest metin var: {without_blocks!r}"
        )


# --------------------------------------------------------------------------
# Şemaya uygunluk
# --------------------------------------------------------------------------

def test_called_tool_is_declared_in_conversation(all_records):
    for i, rec in enumerate(all_records):
        declared = {t["name"] for t in rec["tools"]}
        for _, obj in iter_tool_calls(rec["messages"]):
            assert obj["name"] in declared, (
                f"kayıt {i}: '{obj['name']}' çağrıldı ama tools listesinde yok ({sorted(declared)})"
            )


def test_argument_keys_are_known(all_records):
    for i, rec in enumerate(all_records):
        by_name = {t["name"]: t for t in rec["tools"]}
        for _, obj in iter_tool_calls(rec["messages"]):
            props = set(by_name[obj["name"]]["parameters"]["properties"])
            unknown = set(obj["arguments"]) - props
            assert not unknown, f"kayıt {i}: '{obj['name']}' bilinmeyen argüman(lar) {unknown}"


def test_required_arguments_present(all_records):
    for i, rec in enumerate(all_records):
        by_name = {t["name"]: t for t in rec["tools"]}
        for _, obj in iter_tool_calls(rec["messages"]):
            req = set(by_name[obj["name"]]["parameters"].get("required", []))
            missing = req - set(obj["arguments"])
            assert not missing, f"kayıt {i}: '{obj['name']}' zorunlu argüman eksik {missing}"


def test_enum_arguments_use_allowed_values(all_records):
    for i, rec in enumerate(all_records):
        by_name = {t["name"]: t for t in rec["tools"]}
        for _, obj in iter_tool_calls(rec["messages"]):
            props = by_name[obj["name"]]["parameters"]["properties"]
            for k, v in obj["arguments"].items():
                enum = props.get(k, {}).get("enum")
                if enum is not None:
                    assert v in enum, f"kayıt {i}: '{obj['name']}.{k}' = {v!r} ∉ {enum}"


def test_argument_types_match_schema(all_records):
    for i, rec in enumerate(all_records):
        by_name = {t["name"]: t for t in rec["tools"]}
        for _, obj in iter_tool_calls(rec["messages"]):
            props = by_name[obj["name"]]["parameters"]["properties"]
            for k, v in obj["arguments"].items():
                declared = props.get(k, {}).get("type")
                if declared == "string":
                    assert isinstance(v, str), f"kayıt {i}: '{obj['name']}.{k}' string olmalı, {type(v).__name__} geldi"
                elif declared == "number":
                    assert isinstance(v, (int, float)) and not isinstance(v, bool), (
                        f"kayıt {i}: '{obj['name']}.{k}' sayı olmalı, {v!r} geldi"
                    )


def test_no_empty_string_arguments(all_records):
    for i, rec in enumerate(all_records):
        for _, obj in iter_tool_calls(rec["messages"]):
            for k, v in obj["arguments"].items():
                if isinstance(v, str):
                    assert v.strip(), f"kayıt {i}: '{obj['name']}.{k}' boş string"


# --------------------------------------------------------------------------
# Meta tutarlılığı ve kapsam
# --------------------------------------------------------------------------

def test_meta_target_tools_match_actual_calls(tool_call_records):
    for rec, meta in tool_call_records:
        called = {obj["name"] for _, obj in iter_tool_calls(rec["messages"])}
        targets = set(meta.get("target_tools") or ([meta["target_tool"]] if meta.get("target_tool") else []))
        assert targets == called, (
            f"{meta['id']}: meta hedef {sorted(targets)} ama çağrılan {sorted(called)}"
        )


def test_tool_call_examples_actually_end_with_a_call(tool_call_records):
    for rec, meta in tool_call_records:
        assert has_tool_call(rec["messages"][-1]["content"]), (
            f"{meta['id']}: decision=tool_call ama son mesajda çağrı yok"
        )


def test_multi_tool_examples_have_multiple_blocks(paired):
    multi = [(r, m) for r, m in paired if len(m.get("target_tools", [])) > 1]
    if not multi:
        pytest.skip("çoklu-tool örneği yok")
    for rec, meta in multi:
        n_blocks = len(TOOLCALL_RE.findall(rec["messages"][-1]["content"]))
        assert n_blocks >= 2, f"{meta['id']}: çoklu-tool ama {n_blocks} blok var"


def test_every_write_tool_is_exercised(all_records, gen):
    called = set()
    for rec in all_records:
        for _, obj in iter_tool_calls(rec["messages"]):
            called.add(obj["name"])
    missing = gen.WRITE_TOOLS - called
    assert not missing, f"hiç çağrılmayan WRITE tool(ları): {missing}"


def test_tool_call_histogram_is_not_dominated_by_one_tool(all_records):
    from collections import Counter

    hist = Counter()
    for rec in all_records:
        for _, obj in iter_tool_calls(rec["messages"]):
            hist[obj["name"]] += 1
    total = sum(hist.values())
    if total < 50:
        pytest.skip("çağrı sayısı istatistik için az")
    top_name, top_n = hist.most_common(1)[0]
    assert top_n / total <= 0.12, (
        f"'{top_name}' tüm çağrıların %{100 * top_n / total:.1f}'ini oluşturuyor (tavan %12)"
    )
