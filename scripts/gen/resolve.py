# -*- coding: utf-8 -*-
"""Takvim ve göreli-tarih çözümleme. Üretici ve validator AYNI mantığı kullanır.

Amaç: kullanıcı metnindeki bir tarih yüzeyini ("14 Mart 2027", "önümüzdeki salı",
"geçen ay") kanonik ISO değerine deterministik olarak çözmek — böylece validator,
tool_call argümanını kullanıcı metninden bağımsız olarak yeniden türetip
doğrulayabilir (halüsinasyon kontrolü).
"""
from __future__ import annotations

import re
from datetime import date, timedelta

TR_MONTHS = {
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
    7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
}
TR_MONTHS_LOWER = {v.lower(): k for k, v in TR_MONTHS.items()}
# yaygın ekli/eksik yazımlar
TR_MONTHS_LOWER.update({
    "subat": 2, "agustos": 8, "eylul": 9, "aralik": 12, "mart": 3, "mayis": 5,
    "haziran": 6, "temmuz": 7, "ekim": 10, "kasim": 11, "ocak": 1, "nisan": 4,
})

TR_WEEKDAYS = {
    0: "pazartesi", 1: "salı", 2: "çarşamba", 3: "perşembe", 4: "cuma",
    5: "cumartesi", 6: "pazar",
}
TR_WEEKDAYS_IDX = {}
for _i, _n in TR_WEEKDAYS.items():
    TR_WEEKDAYS_IDX[_n] = _i
TR_WEEKDAYS_IDX.update({"sali": 1, "carsamba": 2, "persembe": 3})


def iso(d: date) -> str:
    return d.isoformat()


def add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    # ay sonu taşması
    from calendar import monthrange
    day = min(d.day, monthrange(y, m)[1])
    return date(y, m, day)


def month_bounds(y: int, m: int):
    from calendar import monthrange
    return date(y, m, 1), date(y, m, monthrange(y, m)[1])


def quarter_bounds(y: int, q: int):
    start_m = (q - 1) * 3 + 1
    a, _ = month_bounds(y, start_m)
    _, b = month_bounds(y, start_m + 2)
    return a, b


def _fold(s: str) -> str:
    s = s.replace("İ", "i").replace("I", "ı").lower()
    for a, b in (("ı", "i"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("ç", "c")):
        s = s.replace(a, b)
    return s


def resolve_date(surface: str, today: date) -> str | None:
    """Bir tarih yüzeyini ISO 'YYYY-MM-DD'ye çöz. Çözülemezse None."""
    s = surface.strip()
    f = _fold(s)

    # ISO
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # DD/MM/YYYY veya DD.MM.YYYY
    m = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return iso(date(y, mo, d))
        except ValueError:
            return None
    # "14 Mart 2027" / "14 mart" (yıl yoksa today.year veya sonraki oluşum)
    m = re.search(r"\b(\d{1,2})\s+([a-zçğıöşü]+)\s*(\d{4})?\b", f)
    if m and m.group(2) in TR_MONTHS_LOWER:
        d = int(m.group(1))
        mo = TR_MONTHS_LOWER[m.group(2)]
        y = int(m.group(3)) if m.group(3) else today.year
        try:
            cand = date(y, mo, d)
        except ValueError:
            return None
        if not m.group(3) and cand < today:
            try:
                cand = date(y + 1, mo, d)
            except ValueError:
                return None
        return iso(cand)
    # göreli
    if f in ("bugun",):
        return iso(today)
    if f in ("yarin",):
        return iso(today + timedelta(days=1))
    if f in ("obur gun", "obürgün", "oburgun"):
        return iso(today + timedelta(days=2))
    if f in ("dun",):
        return iso(today - timedelta(days=1))
    m = re.search(r"(\d+)\s+gun\s+sonra", f)
    if m:
        return iso(today + timedelta(days=int(m.group(1))))
    m = re.search(r"(\d+)\s+gun\s+once", f)
    if m:
        return iso(today - timedelta(days=int(m.group(1))))
    m = re.search(r"(\d+)\s+hafta\s+sonra", f)
    if m:
        return iso(today + timedelta(weeks=int(m.group(1))))
    # "önümüzdeki salı" / "gelecek cuma" / "bu perşembe"
    m = re.search(r"(onumuzdeki|gelecek|bu|haftaya)\s+([a-z]+)", f)
    if m and m.group(2) in TR_WEEKDAYS_IDX:
        target = TR_WEEKDAYS_IDX[m.group(2)]
        delta = (target - today.weekday()) % 7
        if m.group(1) in ("onumuzdeki", "gelecek", "haftaya"):
            delta = delta or 7
            if delta < 7 and m.group(1) in ("onumuzdeki", "gelecek"):
                delta += 7 if delta == 0 else 0
        if delta == 0 and m.group(1) == "bu":
            delta = 0
        return iso(today + timedelta(days=delta or (7 if m.group(1) != "bu" else 0)))
    if "haftaya" in f and "basi" in f:
        # haftanın başı = önümüzdeki pazartesi
        delta = (0 - today.weekday()) % 7 or 7
        return iso(today + timedelta(days=delta))
    if f in ("ay sonu", "ayin sonu", "bu ay sonu"):
        _, b = month_bounds(today.year, today.month)
        return iso(b)
    if f in ("ay basi", "ayin basi", "onumuzdeki ay basi", "gelecek ay basi"):
        return iso(add_months(today.replace(day=1), 1))
    return None


def resolve_range(surface: str, today: date):
    """Bir tarih ARALIĞI yüzeyini (start_iso, end_iso)'ya çöz. Çözülemezse (None, None).
    gen_date_range'in ürettiği biçimleri kapsar (üretici + validator paylaşır)."""
    s = surface.strip()
    f = _fold(s)
    # "... başlangıçlı N günlük"
    m = re.search(r"(.+?)\s+baslangicli\s+(\d+)\s+gunluk", f)
    if m:
        st = resolve_date(m.group(1), today)
        if st:
            d = date.fromisoformat(st) + timedelta(days=int(m.group(2)) - 1)
            return st, iso(d)
    # ISO / ISO
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s*/\s*(\d{4}-\d{2}-\d{2})", s)
    if m:
        return m.group(1), m.group(2)
    # DD/MM/YYYY ile DD/MM/YYYY
    m = re.search(r"(\d{1,2}[./]\d{1,2}[./]\d{4})\s*(?:ile|-|–)?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})", s)
    if m:
        return resolve_date(m.group(1), today), resolve_date(m.group(2), today)
    # "D Ay YYYY - D Ay YYYY"  /  "D - D Ay YYYY"  /  "D-D Ay YYYY"
    m = re.search(r"(\d{1,2})(?:\s+([a-zçğıöşü]+))?\s+(\d{4})?\s*[-–]\s*(\d{1,2})\s+([a-zçğıöşü]+)\s+(\d{4})", f)
    if m:
        d1, mo1, y1, d2, mo2, y2 = m.groups()
        mo2n = TR_MONTHS_LOWER.get(mo2)
        y2n = int(y2)
        mo1n = TR_MONTHS_LOWER.get(mo1) if mo1 else mo2n
        y1n = int(y1) if y1 else y2n
        if mo1n and mo2n:
            try:
                return iso(date(y1n, mo1n, int(d1))), iso(date(y2n, mo2n, int(d2)))
            except ValueError:
                return None, None
    # "D-D Ay YYYY"  (aynı ay)
    m = re.search(r"(\d{1,2})\s*[-–]\s*(\d{1,2})\s+([a-zçğıöşü]+)\s+(\d{4})", f)
    if m and m.group(3) in TR_MONTHS_LOWER:
        mo = TR_MONTHS_LOWER[m.group(3)]
        y = int(m.group(4))
        try:
            return iso(date(y, mo, int(m.group(1)))), iso(date(y, mo, int(m.group(2))))
        except ValueError:
            return None, None
    return None, None


def resolve_period(surface: str, today: date) -> str | None:
    """Ay dönemi yüzeyini 'YYYY-MM'ye çöz."""
    s = surface.strip()
    f = _fold(s)
    m = re.search(r"\b(\d{4})-(\d{2})\b", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"\b(\d{1,2})/(\d{4})\b", s)
    if m:
        return f"{int(m.group(2)):04d}-{int(m.group(1)):02d}"
    m = re.search(r"\b([a-zçğıöşü]+)\s+(\d{4})\b", f)
    if m and m.group(1) in TR_MONTHS_LOWER:
        return f"{int(m.group(2)):04d}-{TR_MONTHS_LOWER[m.group(1)]:02d}"
    m = re.search(r"\b(\d{4})\s+([a-zçğıöşü]+)\b", f)
    if m and m.group(2) in TR_MONTHS_LOWER:
        return f"{int(m.group(1)):04d}-{TR_MONTHS_LOWER[m.group(2)]:02d}"
    if f in ("bu ay", "icinde bulundugumuz ay", "su anki ay"):
        return f"{today:%Y-%m}"
    if f in ("gecen ay", "onceki ay", "gecen ayki"):
        p = today.replace(day=1) - timedelta(days=1)
        return f"{p:%Y-%m}"
    if f in ("evvelki ay", "iki ay once"):
        p = add_months(today.replace(day=1), -2)
        return f"{p:%Y-%m}"
    return None


def resolve_year(surface: str, today: date) -> str | None:
    f = _fold(surface.strip())
    m = re.search(r"\b(20\d{2})\b", surface)
    if m:
        return m.group(1)
    if f in ("bu yil", "bu sene", "icinde bulundugumuz yil"):
        return f"{today.year}"
    if f in ("gecen yil", "gecen sene", "onceki yil"):
        return f"{today.year - 1}"
    if f in ("evvelki yil", "iki yil once"):
        return f"{today.year - 2}"
    if f in ("gelecek yil", "onumuzdeki yil", "seneye"):
        return f"{today.year + 1}"
    return None
