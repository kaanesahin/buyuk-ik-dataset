# -*- coding: utf-8 -*-
"""
test_encoding_and_serialization.py — BAYT/SERİLEŞTİRME sağlamlığı (§21, §22, §37)
========================================================================

Eğitim boru hattı dosyayı yükleyip yeniden serileştirdiğinde HİÇBİR ŞEY
değişmemeli; tool_call blokları TEK bir kanonik biçimde olmalı; Türkçe
karakterler literal (kaçışsız) ve NFC-normal olmalı.

Kapsam
------
* Her kayıt round-trip kararlı: ``json.loads(json.dumps(x, ensure_ascii=False)) == x``.
* Türkçe karakterler dosyada LİTERAL — hiç ``\\uXXXX`` kaçış dizisi yok
  (``ensure_ascii=False`` sözleşmesi).
* Tüm metin NFC-normal (birleşik ç/ş/ğ değil, ayrı birleştirici işaret yok).
* Yalın vekil (lone surrogate) yok; her kayıt UTF-8'e sorunsuz kodlanıyor.
* Satır sonu boşluğu yok; sekme karakteri yok (tool_call içi hariç zaten yok).
* JSON satırlarında tekrar eden anahtar yok (sessiz veri kaybı riski).
* **tool_call bloğu kanonik**: her blok, ``generate_dataset.tool_call_block``
  çıktısıyla BİREBİR yeniden üretilebiliyor — biçim asla elle yazılmış değil.
* ``arguments`` alan sırası ``json.dumps`` çıktısıyla aynı (deterministik anahtar sırası).
* ``tools.json`` tam olarak ``indent=2`` kanonik biçiminde (+ son newline).
* meta.jsonl'deki ``tools``/``messages``, data.jsonl ile BİREBİR aynı nesne.
"""
from __future__ import annotations

import json
import re
import unicodedata

import pytest

from conftest import DATA_DIR, PREFIX

TOOLCALL_BLOCK_RE = re.compile(r"<tool_call>\n(\{.*?\})\n</tool_call>", re.DOTALL)
UNICODE_ESCAPE_RE = re.compile(r"\\u[0-9a-fA-F]{4}")


# --------------------------------------------------------------------------
# Round-trip
# --------------------------------------------------------------------------

def test_records_are_round_trip_stable(all_records):
    for i, rec in enumerate(all_records):
        again = json.loads(json.dumps(rec, ensure_ascii=False))
        assert again == rec, f"kayıt {i}: json round-trip nesneyi değiştirdi"


def test_meta_records_are_round_trip_stable(all_meta):
    for i, m in enumerate(all_meta):
        assert json.loads(json.dumps(m, ensure_ascii=False)) == m, f"meta {i}: round-trip kararsız"


# --------------------------------------------------------------------------
# Kodlama
# --------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["train", "val"])
def test_turkish_characters_are_literal_not_escaped(split):
    raw = (DATA_DIR / f"{PREFIX}_{split}.jsonl").read_text(encoding="utf-8")
    escapes = UNICODE_ESCAPE_RE.findall(raw)
    assert not escapes, f"{split}: {len(escapes)} adet \\uXXXX kaçış — ensure_ascii=False olmalı"
    assert all(ch in raw for ch in "çşğüöıİ"), f"{split}: beklenen Türkçe karakterler dosyada yok"


def test_all_text_is_nfc_normalized(all_records):
    for i, rec in enumerate(all_records):
        blob = json.dumps(rec, ensure_ascii=False)
        assert unicodedata.normalize("NFC", blob) == blob, (
            f"kayıt {i}: NFC-normal değil (birleştirici işaret / ayrık diyakritik)"
        )


def test_no_lone_surrogates(all_records):
    for i, rec in enumerate(all_records):
        try:
            json.dumps(rec, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError as e:
            pytest.fail(f"kayıt {i}: kodlanamıyor (yalın vekil?) — {e}")


@pytest.mark.parametrize("fname", [f"{PREFIX}_train.jsonl", f"{PREFIX}_val.jsonl",
                                   f"{PREFIX}_train.meta.jsonl", f"{PREFIX}_val.meta.jsonl"])
def test_no_trailing_whitespace_and_no_tabs(fname):
    path = DATA_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} yok")
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        assert line == line.rstrip(), f"{fname}:{n} satır sonu boşluğu"
        assert "\t" not in line, f"{fname}:{n} sekme karakteri"


# --------------------------------------------------------------------------
# JSON bütünlüğü
# --------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["train", "val"])
def test_no_duplicate_json_keys(split):
    class _Dup(Exception):
        pass

    def _check(pairs):
        seen = set()
        for k, _ in pairs:
            if k in seen:
                raise _Dup(k)
            seen.add(k)
        return dict(pairs)

    path = DATA_DIR / f"{PREFIX}_{split}.jsonl"
    if not path.exists():
        pytest.skip(f"{split} yok")
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            json.loads(line, object_pairs_hook=_check)
        except _Dup as e:  # noqa: PERF203
            pytest.fail(f"{split}:{n} tekrar eden JSON anahtarı: {e}")


# --------------------------------------------------------------------------
# tool_call bloğu kanonik
# --------------------------------------------------------------------------

def test_every_tool_call_block_is_canonically_reconstructable(all_records, gen):
    failures = []
    for i, rec in enumerate(all_records):
        for m in rec["messages"]:
            if m["role"] != "assistant" or "<tool_call>" not in m["content"]:
                continue
            payloads = TOOLCALL_BLOCK_RE.findall(m["content"])
            assert payloads, f"kayıt {i}: <tool_call> var ama kanonik blok deseni yok"
            rebuilt = "\n".join(
                gen.tool_call_block(json.loads(p)["name"], json.loads(p)["arguments"])
                for p in payloads
            )
            if rebuilt != m["content"].strip():
                failures.append(f"kayıt {i}: blok yeniden üretilemedi")
    assert not failures, "Kanonik olmayan tool_call blokları:\n  " + "\n  ".join(failures[:15])


def test_tool_call_argument_key_order_is_deterministic(all_records):
    """Blok içindeki JSON, json.dumps'ın anahtar sırasını korur (elle sıralama yok)."""
    for i, rec in enumerate(all_records):
        for m in rec["messages"]:
            if m["role"] != "assistant":
                continue
            for p in TOOLCALL_BLOCK_RE.findall(m["content"]):
                obj = json.loads(p)
                assert p == json.dumps(obj, ensure_ascii=False), (
                    f"kayıt {i}: tool_call JSON'u json.dumps kanonik çıktısıyla aynı değil"
                )


# --------------------------------------------------------------------------
# Dosya kanoniklikleri
# --------------------------------------------------------------------------

def test_tools_json_is_canonical_indent_2(tools_inventory, gen):
    path = DATA_DIR / f"{PREFIX}_tools.json"
    actual = path.read_text(encoding="utf-8")
    expected = json.dumps(list(gen.TOOLS.values()), ensure_ascii=False, indent=2) + "\n"
    assert actual == expected, "tools.json kanonik indent=2 (+ son newline) biçiminde değil"


def test_meta_tools_and_messages_are_identical_objects(paired):
    for rec, meta in paired:
        assert meta.get("tools") == rec["tools"], f"{meta['id']}: meta.tools ≠ record.tools"
        assert meta.get("messages") == rec["messages"], f"{meta['id']}: meta.messages ≠ record.messages"


def test_jsonl_files_end_with_single_newline():
    for fname in (f"{PREFIX}_train.jsonl", f"{PREFIX}_val.jsonl",
                  f"{PREFIX}_train.meta.jsonl", f"{PREFIX}_val.meta.jsonl"):
        path = DATA_DIR / fname
        if not path.exists():
            continue
        raw = path.read_bytes()
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n"), f"{fname}: tam olarak tek son newline değil"
