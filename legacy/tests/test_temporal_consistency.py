# -*- coding: utf-8 -*-
"""
test_temporal_consistency.py — TARİH/ZAMAN akıl yürütmesinin doğruluğu (§9, §25, §30)
============================================================================

Tarih argümanları yalnızca "kullanıcı metninden geliyor" değil, aynı zamanda
TAKVİM AÇISINDAN doğru ve İŞ MANTIĞI açısından tutarlı olmalı:

* aralık uçları geçerli takvim günü (2026-02-30 yok, Şubat 2026 = 28 gün);
* "5 Kasım başlangıçlı 3 günlük" → bitiş = başlangıç + 2 gün (ay/yıl taşması dahil);
* ay/çeyrek aralıkları gerçek takvim sınırına oturur;
* göreli ifade çözümü (`bu ay`, `geçen ay`, `geçen yıl`) `--today`'e göre doğru;
* **izin talebi tarihleri geleceğe dönük** (geçmişe izin oluşturulmaz);
* **bordro / mesai dönemi makul bir pencerede** (bugünden en fazla 1 ay ileri,
  ~15 ay geri) — gelecekteki bir maaş pusulası istenmez;
* geçmiş sorguları (`get_puantaj`, `get_izin_gecmisi`) geleceğe uzanmaz;
* hiçbir dönem argümanı, yüzeyinin ima ettiği yıldan farklı bir yıla ait değil.
"""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

import pytest

from conftest import iter_tool_calls, user_blob

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATE_ARG_KEYS = {"baslangic_tarihi", "bitis_tarihi", "yeni_baslangic_tarihi",
                 "yeni_bitis_tarihi", "gecerlilik_tarihi"}
N_DAY_SURFACE = re.compile(r"başlangıçlı (\d+) günlük")
YEAR_IN_SURFACE = re.compile(r"\b(20\d{2})\b")


@pytest.fixture(scope="session")
def today(gen) -> date:
    return date.fromisoformat(gen.DEFAULT_TODAY)


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


# --------------------------------------------------------------------------
# Slot havuzlarının takvim doğruluğu (üreticiye karşı)
# --------------------------------------------------------------------------

def test_all_range_pool_dates_are_valid_calendar_days(gen):
    for surf, b, e in (*gen.DATE_RANGES, *gen.MONTH_RANGES):
        db = date.fromisoformat(b)   # ValueError → test hatası
        de = date.fromisoformat(e)
        assert db <= de, f"{surf!r}: başlangıç {b} > bitiş {e}"


def test_n_day_surfaces_compute_the_correct_end_date(gen):
    for surf, b, e in gen.DATE_RANGES:
        m = N_DAY_SURFACE.search(surf)
        if not m:
            continue
        n = int(m.group(1))
        expected = (date.fromisoformat(b) + timedelta(days=n - 1)).isoformat()
        assert e == expected, f"{surf!r}: bitiş {e}, {n} gün için beklenen {expected}"


def test_month_and_quarter_ranges_align_to_calendar_boundaries(gen):
    for surf, b, e in gen.MONTH_RANGES:
        db, de = date.fromisoformat(b), date.fromisoformat(e)
        assert db.day == 1, f"{surf!r}: başlangıç ayın 1'i değil ({b})"
        last = calendar.monthrange(de.year, de.month)[1]
        assert de.day == last, f"{surf!r}: bitiş ayın son günü değil ({e}, olması gereken gün {last})"
        # çeyrek ise 3 ay
        if "çeyrek" in surf:
            assert _months_between(db, de) == 2, f"{surf!r}: çeyrek 3 ay sürmüyor"


def test_period_pool_months_are_in_range(gen):
    for surf, canon in gen.DONEMLER:
        y, mo = canon.split("-")
        assert 1 <= int(mo) <= 12, f"{surf!r} → {canon}: geçersiz ay"
        assert 2024 <= int(y) <= 2027, f"{surf!r} → {canon}: mantıksız yıl"


@pytest.mark.parametrize("term,expected", [
    ("bu ay", "%Y-%m"), ("geçen ay", "prev-month"), ("geçen ayki", "prev-month"),
    ("önceki ay", "prev-month"), ("bu yıl", "%Y"), ("geçen yıl", "prev-year"),
])
def test_relative_period_resolution_is_correct(gen, today, term, expected):
    got = gen.resolve_relative_period(term, today)
    if expected == "%Y-%m":
        want = f"{today:%Y-%m}"
    elif expected == "%Y":
        want = f"{today:%Y}"
    elif expected == "prev-month":
        want = f"{(today.replace(day=1) - timedelta(days=1)):%Y-%m}"
    else:
        want = str(today.year - 1)
    assert got == want, f"resolve_relative_period({term!r}) = {got}, beklenen {want}"


# --------------------------------------------------------------------------
# Üretilmiş çağrılarda takvim geçerliliği
# --------------------------------------------------------------------------

def test_every_iso_date_argument_is_a_valid_calendar_day(all_records):
    for i, rec in enumerate(all_records):
        for _, obj in iter_tool_calls(rec["messages"]):
            for k, v in obj["arguments"].items():
                if isinstance(v, str) and ISO_DATE.match(v):
                    try:
                        date.fromisoformat(v)
                    except ValueError:
                        pytest.fail(f"kayıt {i} {obj['name']}.{k}: geçersiz takvim günü {v}")


def test_date_argument_year_matches_the_user_surface(all_records):
    """Çağrıdaki tarih yılı, kullanıcının yazdığı yılla aynı olmalı."""
    failures = []
    for i, rec in enumerate(all_records):
        years_in_text = set(YEAR_IN_SURFACE.findall(user_blob(rec["messages"])))
        if not years_in_text:
            continue
        for _, obj in iter_tool_calls(rec["messages"]):
            for k, v in obj["arguments"].items():
                if isinstance(v, str) and (ISO_DATE.match(v) or re.match(r"^\d{4}-\d{2}$", v)):
                    yr = v[:4]
                    if yr not in years_in_text:
                        failures.append(f"kayıt {i} {obj['name']}.{k}={v} — metindeki yıllar {years_in_text}")
    assert not failures, "Tarih yılı kullanıcı metniyle uyuşmuyor:\n  " + "\n  ".join(failures[:20])


# --------------------------------------------------------------------------
# İş mantığı: gelecek vs geçmiş
# --------------------------------------------------------------------------

def test_leave_requests_are_for_future_dates(all_records, today):
    """`create_izin_talebi` / `update_izin_talebi` tarihleri bugünden ileride."""
    failures = []
    for i, rec in enumerate(all_records):
        for _, obj in iter_tool_calls(rec["messages"]):
            if obj["name"] not in ("create_izin_talebi", "update_izin_talebi"):
                continue
            for k in ("baslangic_tarihi", "yeni_baslangic_tarihi"):
                v = obj["arguments"].get(k)
                if v and date.fromisoformat(v) < today:
                    failures.append(f"kayıt {i} {obj['name']}.{k}={v} bugünden ({today}) önce")
    assert not failures, "Geçmişe izin talebi:\n  " + "\n  ".join(failures[:20])


def test_payslip_and_overtime_periods_are_within_a_sane_window(all_records, today):
    """`get_bordro` / `get_mesai_bilgisi` dönemi: en fazla 1 ay ileri, ~15 ay geri."""
    ref = today.replace(day=1)
    failures = []
    for i, rec in enumerate(all_records):
        for _, obj in iter_tool_calls(rec["messages"]):
            if obj["name"] not in ("get_bordro", "get_mesai_bilgisi"):
                continue
            donem = obj["arguments"].get("donem")
            if not (donem and re.match(r"^\d{4}-\d{2}$", donem)):
                continue
            y, mo = map(int, donem.split("-"))
            delta = _months_between(date(y, mo, 1), ref)  # + = geçmiş, - = gelecek
            if not (-1 <= delta <= 15):
                failures.append(f"kayıt {i} {obj['name']}.donem={donem} ({delta} ay uzaklıkta)")
    assert not failures, "Mantıksız bordro/mesai dönemi:\n  " + "\n  ".join(failures[:20])


def test_historical_queries_do_not_extend_into_the_future(all_records, today):
    """`get_puantaj` / `get_izin_gecmisi` — geçmiş kayıt sorguları geleceğe uzanmaz."""
    failures = []
    for i, rec in enumerate(all_records):
        for _, obj in iter_tool_calls(rec["messages"]):
            if obj["name"] not in ("get_puantaj", "get_izin_gecmisi"):
                continue
            b = obj["arguments"].get("baslangic_tarihi")
            if b and date.fromisoformat(b) > today:
                failures.append(f"kayıt {i} {obj['name']}.baslangic_tarihi={b} gelecekte")
    assert not failures, "Geleceğe uzanan geçmiş sorgusu:\n  " + "\n  ".join(failures[:20])


def test_date_range_arguments_do_not_span_more_than_one_quarter(all_records):
    for i, rec in enumerate(all_records):
        for _, obj in iter_tool_calls(rec["messages"]):
            a = obj["arguments"]
            b = a.get("baslangic_tarihi") or a.get("yeni_baslangic_tarihi")
            e = a.get("bitis_tarihi") or a.get("yeni_bitis_tarihi")
            if b and e:
                span = (date.fromisoformat(e) - date.fromisoformat(b)).days
                assert span <= 92, f"kayıt {i} {obj['name']}: {span} günlük aralık (bir çeyrekten uzun)"
