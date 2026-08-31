# -*- coding: utf-8 -*-
"""
test_chat_template_rendering.py — SOHBET ŞABLONU / TOKENİZASYON güvenliği (§21, §22, §33)
================================================================================

Kayıtlar bir sohbet şablonuyla (Qwen ChatML) tokenize edilecek. Bu dosya,
şablonun BOZULMAYACAĞINI ve tokenizer'ın yanlış sınır çıkarmayacağını garanti eder.

Kapsam
------
* Hiçbir mesaj içeriği ChatML/özel token içermiyor: ``<|im_start|>``, ``<|im_end|>``,
  ``<|endoftext|>``, ``<s>``, ``</s>``, ``[INST]`` vb.
* ``<tool_call>`` / ``</tool_call>`` işaretleri her assistant mesajında DENGELİ
  (eşit açılış/kapanış) ve yalnızca assistant turlarında.
* Hiçbir yerde ``<tool_response>`` yok (§33: bu sette tool sonucu taklidi yok).
* tool_call bloğu dışındaki metinde şablon-kıran dizi yok (``{{``, ``{%``, ``%}``).
* Assistant içeriği baş/son boşluk taşımıyor (tur sınırı belirsizliği yok).
* Minimal bir ChatML renderer ile örnek iyi biçimli render ediliyor:
  system (araçlar) + sıralı user/assistant blokları, sonda bir assistant turu,
  ``<|im_start|>``/``<|im_end|>`` çiftleri dengeli.
* ``tools`` sistem istemine gömüldüğünde geçerli JSON ve round-trip kararlı.
* Render edilen örnek makul token bütçesinde (kabaca ≤ 3000 token).
* tool_call JSON'u satır-bazlı ayrıştırılabilir: ``<tool_call>\\n{...}\\n</tool_call>``.
"""
from __future__ import annotations

import json
import re

import pytest

from conftest import strip_tool_calls

SPECIAL_TOKENS = [
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|object_ref_start|>",
    "<|object_ref_end|>", "<|box_start|>", "<|vision_start|>", "<|fim_prefix|>",
    "<s>", "</s>", "<pad>", "[INST]", "[/INST]", "<<SYS>>",
]
JINJA_BREAKERS = re.compile(r"\{\{|\{%|%\}")
STRICT_BLOCK_RE = re.compile(r"<tool_call>\n\{.*?\}\n</tool_call>", re.DOTALL)
CHAR_PER_TOKEN = 3.0
MAX_TOKENS = 3000


# --------------------------------------------------------------------------
# Minimal ChatML renderer (test amaçlı — gerçek tokenizer değil)
# --------------------------------------------------------------------------

def render_chatml(rec: dict) -> str:
    parts = []
    tools_json = json.dumps(rec["tools"], ensure_ascii=False)
    system = (
        "You are Büyük İK, a tool-using assistant.\n"
        f"# Tools\n<tools>\n{tools_json}\n</tools>"
    )
    parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    for m in rec["messages"]:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")  # üretim istemi
    return "".join(parts)


# --------------------------------------------------------------------------
# İçerik temizliği
# --------------------------------------------------------------------------

def test_no_message_contains_special_chat_tokens(all_records):
    hits = []
    for i, rec in enumerate(all_records):
        for m in rec["messages"]:
            for tok in SPECIAL_TOKENS:
                if tok in m["content"]:
                    hits.append(f"kayıt {i} ({m['role']}): {tok}")
    assert not hits, "Özel token sızıntısı:\n  " + "\n  ".join(hits[:20])


def test_tool_call_markers_are_balanced_and_assistant_only(all_records):
    for i, rec in enumerate(all_records):
        for m in rec["messages"]:
            opens = m["content"].count("<tool_call>")
            closes = m["content"].count("</tool_call>")
            if m["role"] == "user":
                assert opens == closes == 0, f"kayıt {i}: user turunda <tool_call> işareti"
            else:
                assert opens == closes, f"kayıt {i}: dengesiz <tool_call> işaretleri ({opens}/{closes})"


def test_no_tool_response_markers_anywhere(all_records):
    for i, rec in enumerate(all_records):
        for m in rec["messages"]:
            assert "<tool_response>" not in m["content"] and "</tool_response>" not in m["content"], (
                f"kayıt {i}: <tool_response> var — bu sette tool sonucu taklidi yok (§33)"
            )


def test_prose_has_no_template_breaking_sequences(all_records):
    hits = []
    for i, rec in enumerate(all_records):
        for m in rec["messages"]:
            prose = strip_tool_calls(m["content"]) if m["role"] == "assistant" else m["content"]
            if JINJA_BREAKERS.search(prose):
                hits.append(f"kayıt {i} ({m['role']}): {prose[:60]!r}")
    assert not hits, "Şablon-kıran dizi ({{ / {% / %}):\n  " + "\n  ".join(hits[:15])


def test_assistant_content_has_no_boundary_whitespace(all_records):
    for i, rec in enumerate(all_records):
        for j, m in enumerate(rec["messages"]):
            if m["role"] != "assistant":
                continue
            assert m["content"] == m["content"].strip(), (
                f"kayıt {i} mesaj {j}: assistant içeriği baş/son boşluk taşıyor (tur sınırı belirsizliği)"
            )


# --------------------------------------------------------------------------
# tool_call bloğu satır-bazlı ayrıştırılabilir
# --------------------------------------------------------------------------

def test_tool_call_blocks_are_line_parseable(all_records):
    for i, rec in enumerate(all_records):
        for m in rec["messages"]:
            if m["role"] != "assistant" or "<tool_call>" not in m["content"]:
                continue
            blocks = re.findall(r"<tool_call>.*?</tool_call>", m["content"], re.DOTALL)
            for b in blocks:
                assert STRICT_BLOCK_RE.fullmatch(b), (
                    f"kayıt {i}: tool_call bloğu <tool_call>\\n{{...}}\\n</tool_call> deseninde değil"
                )
                inner = b[len("<tool_call>\n"):-len("\n</tool_call>")]
                assert "\n" not in inner, f"kayıt {i}: tool_call JSON'u tek satır değil (streaming parser kırılır)"
                json.loads(inner)  # geçerli JSON


# --------------------------------------------------------------------------
# Renderer çıktısı iyi biçimli
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rendered(all_records):
    return [(rec, render_chatml(rec)) for rec in all_records]


def test_rendered_output_has_balanced_im_markers(rendered):
    for rec, text in rendered:
        opens = text.count("<|im_start|>")
        closes = text.count("<|im_end|>")
        # son üretim istemi kapatılmadığı için +1 açık
        assert opens == closes + 1, f"render: <|im_start|> ({opens}) / <|im_end|> ({closes}) dengesiz"


def test_rendered_output_ends_with_assistant_prompt(rendered):
    for _, text in rendered:
        assert text.endswith("<|im_start|>assistant\n"), "render son üretim istemiyle bitmiyor"


def test_rendered_output_role_sequence_is_system_then_alternating(rendered):
    role_re = re.compile(r"<\|im_start\|>(\w+)\n")
    for rec, text in rendered:
        roles = role_re.findall(text)
        assert roles[0] == "system", f"ilk blok system değil: {roles[:3]}"
        convo = roles[1:-1]  # sondaki üretim istemi hariç
        assert convo == [m["role"] for m in rec["messages"]], "render rol sırası mesajlarla uyuşmuyor"
        for a, b in zip(convo, convo[1:]):
            assert a != b, f"render'da ardışık aynı rol: {convo}"


def test_injected_tools_json_round_trips(rendered):
    tools_re = re.compile(r"<tools>\n(.*?)\n</tools>", re.DOTALL)
    for rec, text in rendered:
        m = tools_re.search(text)
        assert m, "render'da <tools> bloğu yok"
        parsed = json.loads(m.group(1))
        assert parsed == rec["tools"], "sistem istemine gömülen tools round-trip'te değişti"


def test_rendered_example_fits_token_budget(rendered):
    over = []
    for rec, text in rendered:
        est = len(text) / CHAR_PER_TOKEN
        if est > MAX_TOKENS:
            over.append(f"~{est:.0f} token")
    assert not over, f"{len(over)} örnek token bütçesini (~{MAX_TOKENS}) aşıyor: {over[:5]}"


def test_rendered_example_is_not_pathologically_short(rendered):
    for rec, text in rendered:
        assert len(text) > 200, "render edilen örnek şüpheli kadar kısa"
