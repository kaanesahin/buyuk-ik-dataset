# -*- coding: utf-8 -*-
"""
test_parameter_grounding_precision.py — ARGÜMANLARIN slot havuzuna kesinliği (§9, §30)
==============================================================================

`test_no_hallucination.py` "değer kullanıcı metninden türetilebiliyor mu" sorusunu
sorar. Bu dosya daha ileri gider: üretilen argüman değeri, üreticinin KAPALI
slot havuzlarından (DATE_RANGES, MONTH_RANGES, DONEMLER, AMOUNT_POOL,
GEREKCE_POOL, POZISYONLAR) BİRİNE ait mi? Tarih çiftleri iç tutarlı mı? Kullanıcı
açık bir ISO tarih verdiyse çağrı onu aynen kullanıyor mu?

Bu, "model serbest metin uydurmuyor, tanımlı değer uzayında kalıyor" güvencesidir.

Kapsam
------
* Her ISO tarih argümanı (`baslangic_tarihi` / `bitis_tarihi` / `yeni_*`) bir
  DATE_RANGES veya MONTH_RANGES kaydının uç değeri.
* Her `donem` argümanı DONEMLER/DONEM_YIL kanoniği VEYA göreli-çözümlenmiş değer.
* `yeni_brut_ucret` ∈ AMOUNT_POOL; `gerekce` ∈ GEREKCE_POOL; `yeni_pozisyon` ∈ POZISYONLAR.
* Tarih çifti: `baslangic ≤ bitis`, süre ≤ 92 gün, ve (b, e) AYNI kaydın uçları
  (iki farklı aralıktan karıştırılmamış).
* Ters yön: kullanıcı turunda birebir ISO tarih varsa, çağrı onu içeriyor.
* `check_employee_access`: `requester_id` ile `hedef_employee_id` FARKLI kişiler.
* Puantaj/geçmiş çağrılarında tarih aralığı bir AY sınırına veya çeyreğe oturuyor
  (rastgele gün aralığı değil).
* Opsiyonel parametreler yalnızca kullanıcı bilgi verdiğinde dolduruluyor
  (`izin_tipi`, `tur`, `durum`) — yoksa çağrı zorunlularla sınırlı.
"""
from __future__ import annotations

import re
from datetime import date

import pytest

from conftest import iter_tool_calls, user_blob

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_DATE_IN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
DATE_ARG_KEYS = {"baslangic_tarihi", "bitis_tarihi", "yeni_baslangic_tarihi",
                 "yeni_bitis_tarihi", "gecerlilik_tarihi"}
MAX_SPAN_DAYS = 92


@pytest.fixture(scope="session")
def pools(gen):
    valid_dates: set[str] = set()
    pair_surface: dict[tuple, str] = {}
    for surf, b, e in (*gen.DATE_RANGES, *gen.MONTH_RANGES):
        valid_dates.add(b)
        valid_dates.add(e)
        pair_surface[(b, e)] = surf
    valid_periods = {c for _, c in gen.DONEMLER} | {c for _, c in gen.DONEM_YIL}
    for term in ("bu ay", "geçen ay", "geçen ayki", "önceki ay", "bu yıl", "geçen yıl", "geçen sene"):
        v = gen.resolve_relative_period(term, date.fromisoformat(gen.DEFAULT_TODAY))
        if v:
            valid_periods.add(v)
    month_range_ends = {(b, e) for _, b, e in gen.MONTH_RANGES}
    return {
        "dates": valid_dates, "pairs": pair_surface, "periods": valid_periods,
        "amounts": set(gen.AMOUNT_POOL), "gerekce": set(gen.GEREKCE_POOL),
        "pozisyon": set(gen.POZISYONLAR), "month_ranges": month_range_ends,
    }


def _all_calls(all_records):
    for i, rec in enumerate(all_records):
        for _, obj in iter_tool_calls(rec["messages"]):
            yield i, rec, obj


# --------------------------------------------------------------------------
# Kapalı havuz üyeliği
# --------------------------------------------------------------------------

def test_iso_date_args_come_from_range_pools(all_records, pools):
    bad = []
    for i, _, obj in _all_calls(all_records):
        for k, v in obj["arguments"].items():
            if isinstance(v, str) and ISO_DATE.match(v) and v not in pools["dates"]:
                bad.append(f"kayıt {i} {obj['name']}.{k} = {v}")
    assert not bad, "Tanımlı aralık havuzunda olmayan ISO tarih argümanları:\n  " + "\n  ".join(bad[:20])


def test_period_args_are_canonical_or_relative(all_records, pools):
    bad = []
    for i, _, obj in _all_calls(all_records):
        v = obj["arguments"].get("donem")
        if v is not None and v not in pools["periods"]:
            bad.append(f"kayıt {i} {obj['name']}.donem = {v}")
    assert not bad, "Tanımsız dönem argümanları:\n  " + "\n  ".join(bad[:20])


def test_numeric_and_freetext_args_come_from_pools(all_records, pools):
    bad = []
    for i, _, obj in _all_calls(all_records):
        a = obj["arguments"]
        if "yeni_brut_ucret" in a and a["yeni_brut_ucret"] not in pools["amounts"]:
            bad.append(f"kayıt {i}: yeni_brut_ucret {a['yeni_brut_ucret']} ∉ AMOUNT_POOL")
        if "gerekce" in a and a["gerekce"] not in pools["gerekce"]:
            bad.append(f"kayıt {i}: gerekce {a['gerekce']!r} ∉ GEREKCE_POOL")
        if "yeni_pozisyon" in a and a["yeni_pozisyon"] not in pools["pozisyon"]:
            bad.append(f"kayıt {i}: yeni_pozisyon {a['yeni_pozisyon']!r} ∉ POZISYONLAR")
    assert not bad, "\n  ".join(bad[:20])


# --------------------------------------------------------------------------
# Tarih çifti tutarlılığı
# --------------------------------------------------------------------------

def test_date_pairs_are_ordered_and_bounded(all_records):
    for i, _, obj in _all_calls(all_records):
        a = obj["arguments"]
        b = a.get("baslangic_tarihi") or a.get("yeni_baslangic_tarihi")
        e = a.get("bitis_tarihi") or a.get("yeni_bitis_tarihi")
        if not (b and e):
            continue
        assert b <= e, f"kayıt {i} {obj['name']}: başlangıç {b} > bitiş {e}"
        span = (date.fromisoformat(e) - date.fromisoformat(b)).days
        assert 0 <= span <= MAX_SPAN_DAYS, f"kayıt {i} {obj['name']}: {span} günlük mantıksız aralık"


def test_date_pairs_are_not_mixed_from_two_ranges(all_records, pools):
    bad = []
    for i, _, obj in _all_calls(all_records):
        a = obj["arguments"]
        b = a.get("baslangic_tarihi") or a.get("yeni_baslangic_tarihi")
        e = a.get("bitis_tarihi") or a.get("yeni_bitis_tarihi")
        if b and e and (b, e) not in pools["pairs"]:
            bad.append(f"kayıt {i} {obj['name']}: ({b}, {e}) tek bir tanımlı aralığa ait değil")
    assert not bad, "Karıştırılmış tarih çiftleri:\n  " + "\n  ".join(bad[:20])


def test_timesheet_and_history_ranges_align_to_month_or_quarter(all_records, pools):
    """get_puantaj / get_izin_gecmisi tarih aralıkları ay/çeyrek sınırına oturur."""
    for i, _, obj in _all_calls(all_records):
        if obj["name"] not in ("get_puantaj", "get_izin_gecmisi"):
            continue
        a = obj["arguments"]
        b, e = a.get("baslangic_tarihi"), a.get("bitis_tarihi")
        if not (b and e):
            continue
        assert (b, e) in pools["month_ranges"], (
            f"kayıt {i} {obj['name']}: ({b}, {e}) ay/çeyrek aralığı değil"
        )


# --------------------------------------------------------------------------
# Ters yön grounding
# --------------------------------------------------------------------------

def test_explicit_user_iso_dates_are_used_verbatim(all_records):
    for i, rec in enumerate(all_records):
        user_isos = set(ISO_DATE_IN.findall(user_blob(rec["messages"])))
        if not user_isos:
            continue
        call_vals = {
            str(v) for _, obj in iter_tool_calls(rec["messages"]) for v in obj["arguments"].values()
        }
        if not call_vals:
            continue
        assert user_isos & call_vals, (
            f"kayıt {i}: kullanıcı ISO tarihi {user_isos} çağrıda {call_vals} kullanılmadı"
        )


# --------------------------------------------------------------------------
# İki-kişi / yetki argümanları
# --------------------------------------------------------------------------

def test_access_check_uses_two_distinct_people(all_records):
    for i, _, obj in _all_calls(all_records):
        if obj["name"] != "check_employee_access":
            continue
        a = obj["arguments"]
        assert a.get("requester_id") and a.get("hedef_employee_id"), f"kayıt {i}: eksik taraf"
        assert a["requester_id"] != a["hedef_employee_id"], (
            f"kayıt {i}: yetki sorgusunda talep eden ve hedef aynı kişi ({a['requester_id']})"
        )


# --------------------------------------------------------------------------
# Opsiyonel parametreler yalnız bilgi verildiğinde
# --------------------------------------------------------------------------

def test_optional_enum_args_only_when_user_specified(all_records, gen):
    """izin_tipi / tur / durum opsiyoneldir; çağrıda varsa kullanıcı metninde
    bilinen bir yüzeyi geçmeli (§10). Yüzey haritaları üreticiden alınır."""
    from conftest import fold

    izin_surf: dict[str, list[str]] = {}
    for s, canon in gen.IZIN_TIPI_YUZEY.items():
        izin_surf.setdefault(canon, []).append(fold(s))
    surf = {
        "izin_tipi": izin_surf,
        "tur": {"net": ["net", "elime", "eline"], "brut": ["brut"]},
        "durum": {"aktif": ["aktif"], "izinli": ["izin"], "ayrildi": ["ayril", "isten cik"]},
    }
    for i, rec, obj in _all_calls(all_records):
        blob = fold(user_blob(rec["messages"]))
        for key, mapping in surf.items():
            if key not in obj["arguments"]:
                continue
            val = obj["arguments"][key]
            variants = mapping.get(val, [val])
            assert any(s in blob for s in variants), (
                f"kayıt {i} {obj['name']}: opsiyonel '{key}={val}' kullanıcı metninde geçmiyor"
            )
