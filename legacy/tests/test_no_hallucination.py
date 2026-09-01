# -*- coding: utf-8 -*-
"""
test_no_hallucination.py — assistant HİÇBİR bilgiyi uydurmuyor (§18, §30, §31)
==========================================================================

When2Call yaklaşımının kalbi: model, tool'dan alınması gereken kişisel/kurumsal
veriyi TAHMİN ETMEMELİ. Bu test seti, her ``tool_call`` argümanının kullanıcı
turlarından türetilebildiğini bağımsız olarak kanıtlar ve assistant düz metninde
uydurulmuş kimlik/sayı arar.

Yaklaşım
--------
`validate_dataset.py` gibi, `generate_dataset` modülünün YÜZEY HARİTALARINI
(izin türü, dönem, tarih aralığı, departman, medeni/öğrenim durumu) yeniden
kullanır — çünkü "15-20 Eylül 2026" gibi bir yüzeyin "2026-09-15" kanoniğine
çözülmesi domain bilgisi gerektirir. Haritalar dışında bağımsızdır.

Kapsam
------
* ``employee_id``  → kullanıcı metninde birebir VEYA rakam alt dizisi olarak var.
* ``talep_id``     → kullanıcı metninde birebir var.
* tarih / dönem    → ISO değer metinde var VEYA bilinen bir yüzeyi metinde var
                     VEYA "bu ay / geçen ay" gibi göreli ifade var.
* ``yeni_brut_ucret`` (sayı) → rakam kullanıcı metninde var.
* enum'lar (izin_tipi, kaynak_tipi, durum, tur, medeni_durum, ogrenim_durumu)
  → değerin bilinen bir yüzeyi kullanıcı metninde var.
* serbest metin (gerekce, adres, pozisyon, acil_durum_*) → gevşek altdizi olarak var.
* Assistant DÜZ METNİNDE (tool_call bloğu hariç) kullanıcıda geçmeyen EMP-/LV- yok.
* ``request_for_info`` / ``cannot_answer`` yanıtlarında kullanıcının vermediği
  "N gün / N TL / N saat" tipi sayı yok.
"""
from __future__ import annotations

import re
from collections import defaultdict

import pytest

from conftest import (
    EMP_RE, ISO_DATE_RE, ISO_PERIOD_RE, LV_RE, TOOLCALL_RE,
    assistant_turns, fold, iter_tool_calls, loose, strip_tool_calls, user_blob,
)


# --- Yüzey → kanonik haritalar (generate_dataset'ten) --------------------

@pytest.fixture(scope="session")
def surfaces(gen):
    date_surf: dict[str, list[str]] = defaultdict(list)
    for surf, b, e in (*gen.DATE_RANGES, *gen.MONTH_RANGES):
        date_surf[b].append(surf)
        date_surf[e].append(surf)
    period_surf: dict[str, list[str]] = defaultdict(list)
    for surf, canon in (*gen.DONEMLER, *gen.DONEM_YIL):
        period_surf[canon].append(surf)
    izin_surf: dict[str, list[str]] = defaultdict(list)
    for surf, canon in gen.IZIN_TIPI_YUZEY.items():
        izin_surf[canon].append(surf)
    medeni_surf: dict[str, list[str]] = defaultdict(list)
    for surf, canon in gen.MEDENI_YUZEY.items():
        medeni_surf[canon].append(surf)
    ogrenim_surf: dict[str, list[str]] = defaultdict(list)
    for surf, canon in gen.OGRENIM_YUZEY.items():
        ogrenim_surf[canon].append(surf)
    dept_surf: dict[str, list[str]] = defaultdict(list)
    for surf, canon in gen.DEPARTMAN_YUZEY.items():
        dept_surf[canon].append(surf)
    for d in gen.DEPARTMANLAR:
        dept_surf[d].append(d)
    return {
        "date": date_surf, "period": period_surf, "izin": izin_surf,
        "medeni": medeni_surf, "ogrenim": ogrenim_surf, "dept": dept_surf,
        "kaynak": gen.KAYNAK_SURF,
    }


DURUM_SURF = {
    "aktif": ["aktif", "calisan"],
    "izinli": ["izin", "izinde", "izinli"],
    "ayrildi": ["ayril", "isten cik", "ayrilmis"],
}
TUR_SURF = {"net": ["net", "elime", "eline"], "brut": ["brut", "brüt"]}
RELATIVE_PERIOD_RE = re.compile(r"bu (yil|ay)|gecen (yil|ay|sene)|onceki (ay|yil)")


# --- Tek argüman izleyici ------------------------------------------------

def _trace(key: str, value, prop_schema: dict, blob: str, blob_folded: str, surfaces: dict) -> str | None:
    """Argüman değeri kullanıcı metninden türetilebiliyor mu? Sorun varsa açıklama döndürür."""
    sval = str(value)

    if EMP_RE.fullmatch(sval):
        num = sval.split("-")[1]
        if sval.lower() in blob.lower():
            return None
        if re.search(rf"(?<!\d){re.escape(num)}(?!\d)", blob):
            return None
        return f"employee_id '{sval}' kullanıcı metninde yok (HALÜSİNASYON)"

    if LV_RE.fullmatch(sval):
        return None if sval.lower() in blob.lower() else f"talep_id '{sval}' kullanıcı metninde yok"

    if key == "donem" or ISO_DATE_RE.fullmatch(sval) or ISO_PERIOD_RE.fullmatch(sval):
        if fold(sval) in blob_folded:
            return None
        cands = surfaces["date"].get(sval, []) + surfaces["period"].get(sval, [])
        if any(fold(s) in blob_folded for s in cands):
            return None
        if key == "donem" and RELATIVE_PERIOD_RE.search(blob_folded):
            return None
        return f"tarih/dönem '{sval}' kullanıcı metninden türetilemiyor"

    enum = prop_schema.get("enum")
    if enum:
        if key == "izin_tipi":
            ok = any(fold(s) in blob_folded for s in surfaces["izin"].get(sval, [sval]))
        elif key == "kaynak_tipi":
            ok = fold(surfaces["kaynak"].get(sval, sval)) in blob_folded or fold(sval) in blob_folded
        elif key == "medeni_durum":
            ok = any(fold(s) in blob_folded for s in surfaces["medeni"].get(sval, [sval]))
        elif key == "ogrenim_durumu":
            ok = any(fold(s) in blob_folded for s in surfaces["ogrenim"].get(sval, [sval]))
        elif key == "durum":
            ok = any(s in blob_folded for s in DURUM_SURF.get(sval, [sval]))
        elif key == "tur":
            ok = any(s in blob_folded for s in TUR_SURF.get(sval, [sval]))
        else:
            ok = fold(sval) in blob_folded
        return None if ok else f"enum {key}='{sval}' kullanıcı metninde görünmüyor"

    if prop_schema.get("type") == "number":
        if re.search(rf"(?<!\d){re.escape(sval)}(?!\d)", blob.replace(".", "")):
            return None
        return f"sayısal arg {key}={sval} kullanıcı metninde yok (HALÜSİNASYON)"

    if key == "departman_adi":
        ok = any(fold(s) in blob_folded for s in surfaces["dept"].get(sval, [sval]))
        return None if ok else f"departman '{sval}' kullanıcı metninde görünmüyor"

    # serbest metin: gerekce / adres / pozisyon / acil_durum_* / aciklama
    return None if loose(sval) in loose(blob) else f"serbest arg {key}='{sval}' kullanıcı metninde yok"


# --------------------------------------------------------------------------
# Ana test: her argüman izlenebilir
# --------------------------------------------------------------------------

def test_every_tool_call_argument_traces_to_user_text(all_records, surfaces):
    failures: list[str] = []
    for i, rec in enumerate(all_records):
        by_name = {t["name"]: t for t in rec["tools"]}
        blob = user_blob(rec["messages"])
        blob_folded = fold(blob)
        for _, obj in iter_tool_calls(rec["messages"]):
            props = by_name[obj["name"]]["parameters"]["properties"]
            for k, v in obj["arguments"].items():
                problem = _trace(k, v, props.get(k, {}), blob, blob_folded, surfaces)
                if problem:
                    failures.append(f"kayıt {i} ({obj['name']}): {problem}")
    assert not failures, "Uydurulmuş argüman(lar):\n  " + "\n  ".join(failures[:40])


def test_employee_ids_in_calls_are_never_from_a_fixed_default(all_records):
    """§18: eksik kimlik 'EMP-1001' gibi sabit bir değere düşülmemeli."""
    from collections import Counter

    ids = Counter()
    for rec in all_records:
        for _, obj in iter_tool_calls(rec["messages"]):
            v = obj["arguments"].get("employee_id")
            if v:
                ids[v] += 1
    if not ids:
        pytest.skip("employee_id çağrısı yok")
    total = sum(ids.values())
    top_id, top_n = ids.most_common(1)[0]
    assert top_n / total < 0.05, (
        f"'{top_id}' employee_id çağrılarının %{100 * top_n / total:.1f}'inde — sabit varsayılan şüphesi"
    )


# --------------------------------------------------------------------------
# Assistant düz metninde uydurma kimlik / sayı
# --------------------------------------------------------------------------

def test_assistant_prose_has_no_invented_employee_or_request_ids(all_records):
    failures = []
    for i, rec in enumerate(all_records):
        blob = user_blob(rec["messages"])
        for a in assistant_turns(rec["messages"]):
            prose = strip_tool_calls(a)
            for emp in EMP_RE.findall(prose):
                num = emp.split("-")[1]
                if emp.lower() in blob.lower() or re.search(rf"(?<!\d){re.escape(num)}(?!\d)", blob):
                    continue
                failures.append(f"kayıt {i}: assistant metninde uydurulmuş '{emp}'")
            for lv in LV_RE.findall(prose):
                if lv.lower() not in blob.lower():
                    failures.append(f"kayıt {i}: assistant metninde uydurulmuş '{lv}'")
    assert not failures, "\n  ".join(failures[:40])


def test_refusal_and_clarification_responses_have_no_fabricated_numbers(paired):
    """§18: 'İzin bakiyeniz 12 gün' gibi uydurma rakam, cevap vermeyen kararlarda olmamalı."""
    num_re = re.compile(r"\b(\d[\d.]*)\s*(?:gün|gun|TL|₺|saat)\b")
    failures = []
    for rec, meta in paired:
        if meta["decision"] not in ("request_for_info", "cannot_answer"):
            continue
        blob_folded = fold(user_blob(rec["messages"]))
        for a in (m["content"] for m in rec["messages"] if m["role"] == "assistant"):
            for num in num_re.findall(a):
                if fold(num) not in blob_folded:
                    failures.append(f"{meta['id']} ({meta['decision']}): uydurma sayı '{num}'")
    assert not failures, "\n  ".join(failures[:40])


def test_direct_answers_do_not_reference_specific_employee_records(paired):
    """`direct` cevaplar GENEL bilgi olmalı; somut bir çalışanın verisini içermemeli."""
    failures = []
    for rec, meta in paired:
        if meta["decision"] != "direct":
            continue
        blob = user_blob(rec["messages"])
        for a in (m["content"] for m in rec["messages"] if m["role"] == "assistant"):
            for emp in EMP_RE.findall(a):
                if emp.lower() not in blob.lower():
                    failures.append(f"{meta['id']}: direct cevapta '{emp}'")
    assert not failures, "\n  ".join(failures)
