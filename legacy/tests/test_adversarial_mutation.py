# -*- coding: utf-8 -*-
"""
test_adversarial_mutation.py — GÜVENLİK AĞININ boş olmadığının kanıtı (§31)
=======================================================================

Bir doğrulayıcı yalnızca gerçek kusurları YAKALADIĞI ölçüde değerlidir. Bu dosya
**mutasyon testi** yapar: gerçek kayıtlara bilinçli kusurlar enjekte eder ve
`validate_dataset.check_record`'ın HER BİRİNİ yakaladığını doğrular. Ayrıca
mutasyonsuz kayıtlarda YANLIŞ POZİTİF üretmediğini kontrol eder.

Bu, meta-testtir: test altyapısının kendisini test eder.

Kapsam (her mutasyon türü yakalanmalı)
--------------------------------------
* zorunlu argümanı sil
* bilinmeyen argüman ekle
* enum değerini geçersiz yap
* string argümanı sayıya çevir (tip uyumsuzluğu)
* employee_id'yi kullanıcıda geçmeyen bir değerle değiştir (halüsinasyon)
* uydurma ISO tarih enjekte et (halüsinasyon)
* uydurma tutar enjekte et (halüsinasyon)
* çağrılan tool'u `tools` listesinden çıkar
* `direct` örneğine tool_call ekle (karar tutarsızlığı)
* `request_for_info` yanıtına "N gün" uydur (halüsinasyon)
* tool_call'ı son olmayan assistant turuna koy
* rol alternasyonunu boz / mesaj içeriğini boşalt

Ek:
* Tüm gerçek kayıtlar `check_record` ile 0 hata (yanlış pozitif yok).
* `check_record`, geçerli bir kayıt kopyasında deterministik (aynı sonuç).
"""
from __future__ import annotations

import copy
import json
import re
import sys

import pytest

from conftest import SCRIPTS_DIR

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@pytest.fixture(scope="session")
def validator():
    try:
        import validate_dataset as V  # noqa: WPS433
    except Exception as e:  # pragma: no cover
        pytest.skip(f"validate_dataset import edilemedi: {e}")
    return V


def _errors(validator, rec, meta) -> list[str]:
    rep = validator.Report()
    validator.check_record(1, rec, meta, rep)
    return rep.errors


def _final_call_obj(rec):
    m = TOOLCALL_RE.search(rec["messages"][-1]["content"])
    return json.loads(m.group(1)) if m else None


def _set_final_call(rec, obj):
    content = rec["messages"][-1]["content"]
    rec["messages"][-1]["content"] = TOOLCALL_RE.sub(
        lambda _: "<tool_call>\n" + json.dumps(obj, ensure_ascii=False) + "\n</tool_call>",
        content, count=1,
    )


@pytest.fixture(scope="session")
def pick(paired):
    def _pick(pred):
        for rec, meta in paired:
            if pred(rec, meta):
                return copy.deepcopy(rec), copy.deepcopy(meta)
        pytest.skip("mutasyon için uygun temel kayıt bulunamadı")
    return _pick


@pytest.fixture(scope="session")
def simple_read(pick):
    return pick(lambda r, m: (
        m["decision"] == "tool_call" and not m["multi_turn"]
        and len(m.get("target_tools", [])) == 1 and not m.get("is_write")
        and (_final_call_obj(r) or {}).get("arguments")
    ))


# --------------------------------------------------------------------------
# Yanlış pozitif yok
# --------------------------------------------------------------------------

def test_all_real_records_pass_check_record(validator, paired):
    dirty = []
    for rec, meta in paired:
        errs = _errors(validator, rec, meta)
        if errs:
            dirty.append(f"{meta['id']}: {errs[0]}")
    assert not dirty, "check_record gerçek kayıtlarda hata üretiyor (yanlış pozitif):\n  " + "\n  ".join(dirty[:20])


def test_check_record_is_deterministic(validator, simple_read):
    rec, meta = simple_read
    a = _errors(validator, copy.deepcopy(rec), copy.deepcopy(meta))
    b = _errors(validator, copy.deepcopy(rec), copy.deepcopy(meta))
    assert a == b == [], "check_record deterministik değil ya da temel kayıt geçersiz"


# --------------------------------------------------------------------------
# Argüman düzeyi mutasyonlar
# --------------------------------------------------------------------------

def test_dropping_a_required_argument_is_caught(validator, simple_read):
    rec, meta = simple_read
    obj = _final_call_obj(rec)
    key = next(iter(obj["arguments"]))
    del obj["arguments"][key]
    _set_final_call(rec, obj)
    assert _errors(validator, rec, meta), f"'{key}' silindi ama yakalanmadı"


def test_adding_an_unknown_argument_is_caught(validator, simple_read):
    rec, meta = simple_read
    obj = _final_call_obj(rec)
    obj["arguments"]["__sahte_alan__"] = "x"
    _set_final_call(rec, obj)
    assert _errors(validator, rec, meta), "bilinmeyen argüman yakalanmadı"


def _has_arg(rec, key):
    obj = _final_call_obj(rec)
    return bool(obj) and key in obj.get("arguments", {})


def test_invalid_enum_value_is_caught(validator, pick):
    rec, meta = pick(lambda r, m: m["decision"] == "tool_call" and _has_arg(r, "izin_tipi"))
    obj = _final_call_obj(rec)
    obj["arguments"]["izin_tipi"] = "GECERSIZ"
    _set_final_call(rec, obj)
    errs = _errors(validator, rec, meta)
    assert any("enum" in e for e in errs), f"geçersiz enum değeri yakalanmadı: {errs}"


def test_type_mismatch_is_caught(validator, pick):
    rec, meta = pick(lambda r, m: (
        m["decision"] == "tool_call"
        and _has_arg(r, "employee_id")
    ))
    obj = _final_call_obj(rec)
    obj["arguments"]["employee_id"] = 12345  # string olmalı
    _set_final_call(rec, obj)
    assert _errors(validator, rec, meta), "tip uyumsuzluğu (string yerine sayı) yakalanmadı"


# --------------------------------------------------------------------------
# Halüsinasyon mutasyonları
# --------------------------------------------------------------------------

def test_hallucinated_employee_id_is_caught(validator, pick):
    rec, meta = pick(lambda r, m: (
        m["decision"] == "tool_call"
        and _has_arg(r, "employee_id")
    ))
    obj = _final_call_obj(rec)
    obj["arguments"]["employee_id"] = "EMP-9998"  # kullanıcı metninde yok
    _set_final_call(rec, obj)
    errs = _errors(validator, rec, meta)
    assert any("HALÜSİNASYON" in e for e in errs), f"uydurma employee_id yakalanmadı: {errs}"


def test_fabricated_iso_date_is_caught(validator, pick):
    rec, meta = pick(lambda r, m: (
        m["decision"] == "tool_call"
        and _has_arg(r, "baslangic_tarihi")
    ))
    obj = _final_call_obj(rec)
    obj["arguments"]["baslangic_tarihi"] = "2099-01-01"
    _set_final_call(rec, obj)
    errs = _errors(validator, rec, meta)
    assert any("HALÜSİNASYON" in e or "türetilemiyor" in e for e in errs), f"uydurma tarih yakalanmadı: {errs}"


def test_fabricated_amount_is_caught(validator, pick):
    rec, meta = pick(lambda r, m: (
        m["decision"] == "tool_call"
        and _has_arg(r, "yeni_brut_ucret")
    ))
    obj = _final_call_obj(rec)
    obj["arguments"]["yeni_brut_ucret"] = 987654
    _set_final_call(rec, obj)
    errs = _errors(validator, rec, meta)
    assert any("HALÜSİNASYON" in e for e in errs), f"uydurma tutar yakalanmadı: {errs}"


def test_fabricated_number_in_refusal_response_is_caught(validator, pick):
    rec, meta = pick(lambda r, m: m["decision"] == "request_for_info")
    rec["messages"][-1]["content"] = "İzin bakiyeniz 12 gün."
    errs = _errors(validator, rec, meta)
    assert any("HALÜSİNASYON" in e for e in errs), f"request_for_info yanıtındaki uydurma sayı yakalanmadı: {errs}"


# --------------------------------------------------------------------------
# Yapısal mutasyonlar
# --------------------------------------------------------------------------

def test_removing_the_called_tool_from_inventory_is_caught(validator, simple_read):
    rec, meta = simple_read
    obj = _final_call_obj(rec)
    rec["tools"] = [t for t in rec["tools"] if t["name"] != obj["name"]]
    errs = _errors(validator, rec, meta)
    assert any("tools listesinde yok" in e for e in errs), f"tanımsız tool çağrısı yakalanmadı: {errs}"


def test_adding_a_tool_call_to_a_direct_example_is_caught(validator, pick):
    rec, meta = pick(lambda r, m: m["decision"] == "direct")
    tool = rec["tools"][0]["name"]
    rec["messages"][-1]["content"] += f'\n<tool_call>\n{{"name": "{tool}", "arguments": {{}}}}\n</tool_call>'
    errs = _errors(validator, rec, meta)
    assert any("decision=direct" in e for e in errs), f"direct + tool_call yakalanmadı: {errs}"


def test_tool_call_in_non_final_turn_is_caught(validator, pick):
    rec, meta = pick(lambda r, m: m["decision"] == "tool_call" and m["turns"] >= 4)
    rec["messages"][1]["content"] += '\n<tool_call>\n{"name": "x", "arguments": {}}\n</tool_call>'
    errs = _errors(validator, rec, meta)
    assert any("son olmayan" in e for e in errs), f"erken tool_call yakalanmadı: {errs}"


def test_broken_role_alternation_is_caught(validator, simple_read):
    rec, meta = simple_read
    rec["messages"].insert(1, {"role": "user", "content": "araya giren mesaj"})
    errs = _errors(validator, rec, meta)
    assert any("alternasyon" in e for e in errs), f"bozuk rol alternasyonu yakalanmadı: {errs}"


def test_empty_message_content_is_caught(validator, simple_read):
    rec, meta = simple_read
    rec["messages"][0]["content"] = "   "
    errs = _errors(validator, rec, meta)
    assert any("içeriği boş" in e for e in errs), f"boş mesaj yakalanmadı: {errs}"


# --------------------------------------------------------------------------
# Toplu mutasyon taraması
# --------------------------------------------------------------------------

def test_a_broad_sweep_of_argument_mutations_is_always_caught(validator, paired):
    """20 farklı `tool_call` örneğine 'zorunlu argümanı sil' uygula — hepsi yakalansın."""
    calls = [(r, m) for r, m in paired
             if m["decision"] == "tool_call" and (_final_call_obj(r) or {}).get("arguments")]
    missed = 0
    for rec, meta in calls[:20]:
        rc, mc = copy.deepcopy(rec), copy.deepcopy(meta)
        obj = _final_call_obj(rc)
        required = validator.G.TOOLS[obj["name"]]["parameters"].get("required", [])
        droppable = [k for k in obj["arguments"] if k in required]
        if not droppable:
            continue
        del obj["arguments"][droppable[0]]
        _set_final_call(rc, obj)
        if not _errors(validator, rc, mc):
            missed += 1
    assert missed == 0, f"{missed} örnekte 'zorunlu argüman silme' mutasyonu yakalanmadı"
