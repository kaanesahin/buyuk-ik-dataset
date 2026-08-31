# -*- coding: utf-8 -*-
"""
test_jsonl_structure.py — dosya ve kayıt düzeyinde YAPISAL bütünlük
==================================================================

En temel garanti: eğitim boru hattı (HuggingFace `datasets`, streaming
yükleyiciler) dosyayı hatasız okuyabilmeli.

Kapsam
------
* Her `.jsonl` satırı bağımsız, geçerli JSON.
* Kodlama UTF-8, BOM yok, satır sonu `\n`, dosya `\n` ile biter, boş satır yok.
* Eğitim kaydı TAM OLARAK ``{"tools": [...], "messages": [...]}`` — fazladan alan yok
  (When2Call §22/§37: eğitim girdisi metadata ile kirletilmez).
* `messages` boş değil; her turda `role` ∈ {user, assistant} ve `content` boş olmayan str.
* `tools` boş olmayan liste; her tool bir dict.
* Meta dosyası satır sayısı, eğitim dosyasıyla birebir aynı (§23).
"""
from __future__ import annotations

import json

import pytest

from conftest import DATA_DIR, PREFIX

DATA_FILES = [f"{PREFIX}_train.jsonl", f"{PREFIX}_val.jsonl"]
META_FILES = [f"{PREFIX}_train.meta.jsonl", f"{PREFIX}_val.meta.jsonl"]
ALL_JSONL = DATA_FILES + META_FILES

TRAIN_KEYS = {"tools", "messages"}
META_REQUIRED_KEYS = {
    "decision", "intent", "target_tool", "target_tools", "required_parameters",
    "missing_parameters", "is_write", "confirmation_required", "domain",
    "difficulty", "register", "multi_turn", "turns", "id",
}


# --------------------------------------------------------------------------
# Dosya düzeyi
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fname", ALL_JSONL)
def test_file_exists_and_nonempty(fname):
    path = DATA_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} yok — önce üreticiyi çalıştırın")
    assert path.stat().st_size > 0, f"{fname} boş"


@pytest.mark.parametrize("fname", ALL_JSONL)
def test_encoding_is_utf8_without_bom(fname):
    path = DATA_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} yok")
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), f"{fname} BOM ile başlıyor"
    raw.decode("utf-8")  # UnicodeDecodeError -> test hatası


@pytest.mark.parametrize("fname", ALL_JSONL)
def test_newline_hygiene(fname):
    path = DATA_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} yok")
    raw = path.read_bytes()
    assert b"\r\n" not in raw, f"{fname} Windows satır sonu (CRLF) içeriyor"
    assert raw.endswith(b"\n"), f"{fname} sonunda newline yok"
    text = raw.decode("utf-8")
    assert "\n\n" not in text, f"{fname} boş satır içeriyor"
    assert not text.startswith("\n"), f"{fname} boş satırla başlıyor"


@pytest.mark.parametrize("fname", ALL_JSONL)
def test_every_line_is_valid_json_object(fname):
    path = DATA_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} yok")
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        obj = json.loads(line)  # JSONDecodeError -> anlaşılır test hatası
        assert isinstance(obj, dict), f"{fname}:{i} JSON objesi değil ({type(obj).__name__})"


def test_data_and_meta_line_counts_match():
    for data_f, meta_f in zip(DATA_FILES, META_FILES):
        dp, mp = DATA_DIR / data_f, DATA_DIR / meta_f
        if not (dp.exists() and mp.exists()):
            pytest.skip("veri/meta dosyaları yok")
        dn = len(dp.read_text(encoding="utf-8").splitlines())
        mn = len(mp.read_text(encoding="utf-8").splitlines())
        assert dn == mn, f"{data_f} ({dn}) ile {meta_f} ({mn}) satır sayısı farklı"


def test_dataset_is_not_trivially_small(all_records):
    assert len(all_records) >= 500, (
        f"toplam {len(all_records)} örnek — LoRA için anlamlı bir set beklenir"
    )


# --------------------------------------------------------------------------
# Eğitim kaydı şeması
# --------------------------------------------------------------------------

def test_training_record_has_exactly_tools_and_messages(all_records):
    offenders = [i for i, r in enumerate(all_records) if set(r) != TRAIN_KEYS]
    assert not offenders, (
        f"{len(offenders)} eğitim kaydında fazladan/eksik anahtar var "
        f"(ilk: index {offenders[0]} -> {sorted(set(all_records[offenders[0]]))})"
    )


def test_tools_field_is_nonempty_list_of_dicts(all_records):
    for i, r in enumerate(all_records):
        assert isinstance(r["tools"], list) and r["tools"], f"kayıt {i}: tools boş/liste değil"
        for t in r["tools"]:
            assert isinstance(t, dict), f"kayıt {i}: tools içinde dict olmayan öğe"


def test_messages_field_is_nonempty_list(all_records):
    for i, r in enumerate(all_records):
        assert isinstance(r["messages"], list) and r["messages"], f"kayıt {i}: messages boş"


def test_every_message_has_valid_role_and_nonempty_string_content(all_records):
    for i, r in enumerate(all_records):
        for j, m in enumerate(r["messages"]):
            assert set(m) == {"role", "content"}, f"kayıt {i} mesaj {j}: beklenmeyen anahtarlar {set(m)}"
            assert m["role"] in ("user", "assistant"), f"kayıt {i} mesaj {j}: geçersiz rol {m['role']!r}"
            assert isinstance(m["content"], str), f"kayıt {i} mesaj {j}: content str değil"
            assert m["content"].strip(), f"kayıt {i} mesaj {j}: content boş/whitespace"


def test_no_control_characters_in_content(all_records):
    bad = {chr(c) for c in range(0x20)} - {"\n", "\t"}
    for i, r in enumerate(all_records):
        for j, m in enumerate(r["messages"]):
            hit = bad.intersection(m["content"])
            assert not hit, f"kayıt {i} mesaj {j}: kontrol karakteri {hit!r}"


# --------------------------------------------------------------------------
# Meta kaydı şeması
# --------------------------------------------------------------------------

def test_meta_records_have_required_keys(all_meta):
    for i, m in enumerate(all_meta):
        missing = META_REQUIRED_KEYS - set(m)
        assert not missing, f"meta {i}: eksik anahtarlar {missing}"


def test_meta_ids_are_unique_and_well_formed(train_meta, val_meta):
    for split, metas in (("train", train_meta), ("val", val_meta)):
        ids = [m["id"] for m in metas]
        assert len(ids) == len(set(ids)), f"{split}: tekrar eden id"
        for _id in ids:
            assert _id.startswith("hr_"), f"{split}: beklenmeyen id biçimi {_id!r}"


def test_meta_turns_matches_actual_message_count(paired):
    for rec, meta in paired:
        assert meta["turns"] == len(rec["messages"]), (
            f"{meta['id']}: meta.turns={meta['turns']} ama {len(rec['messages'])} mesaj var"
        )
