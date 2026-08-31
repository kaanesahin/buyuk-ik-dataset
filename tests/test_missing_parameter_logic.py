# -*- coding: utf-8 -*-
"""
test_missing_parameter_logic.py — REQUEST_FOR_INFO'nun mantıksal doğruluğu (§3, §10, §30)
=================================================================================

`request_for_info` üç koşulu birden sağlamalı:
  1. ``missing_parameters`` gerçekten kullanıcı metninde YOK,
  2. eksik OLMAYAN zorunlu parametreler metinde VAR (kısmi-bilgi senaryosu),
  3. asistanın sorusu TAM OLARAK eksik olan parametre(ler)i istiyor.

Ayrıca sınıflandırma temiz olmalı: her `request_for_info` ya eksik-bilgi
(``missing_parameters != []``) ya da onay-bekleme (``confirmation_required``)
olmalı — ikisi de olmayan "sahte" bir request_for_info bulunmamalı.

Kapsam
------
* `request_for_info` partisyonu: {eksik-bilgi} ∪ {onay-bekleme}, kesişim ve
  boşluk yok.
* Her eksik parametre kullanıcı metninde yok (employee_id / talep_id / tarih /
  dönem / izin_tipi / ücret / iletişim).
* Eksik olmayan her zorunlu parametre kullanıcı metninde var.
* Asistan sorusu eksik parametreye özgü anahtar kelime içeriyor.
* `missing_parameters` ⊆ hedef tool'un `required` listesi (opsiyonel alan "eksik"
  diye sorulmaz — §10).
* Onay-bekleme mesajı, yapılacak işlemi SOMUT olarak özetliyor (emp/talep/tarih).
* `resolve_employee_identity`: isim verilmiş ama employee_id istenmiş.
"""
from __future__ import annotations

import re

import pytest

from conftest import fold, has_tool_call, iter_tool_calls, user_blob

EMP_REF_RE = re.compile(
    r"emp-?\d+|sicil no \d+|\d{3,4} numarali (calisan|personel)|\d{3,4} numarali calisan"
)
LV_REF_RE = re.compile(r"lv-?\d")
# Yalnızca MUTLAK dönem/tarih ifadeleri. "bu ay" / "geçen ay" gibi göreli ifadeler
# BİLEREK dışarıda: (a) UZUN stil önekinde ("Bu ay birkaç işi toparlıyorum…")
# gürültü olarak geçer, (b) üretici bu göreli ifadeleri `donem` için yetersiz sayıp
# yine de sorar. `test_decision_oracle.py` ile aynı tanım.
MONTH_RE = (r"nisan|mayis|haziran|temmuz|agustos|eylul|ekim|kasim|aralik|ocak|subat|mart"
            r"|ceyrek|\b20\d\d\b|\d{4}-\d{2}")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}[/.]\d{1,2}|" + MONTH_RE)
IZIN_TIPI_RE = re.compile(r"yillik|senelik|mazeret|hastalik|saglik|rapor")

# eksik parametre → asistanın sorusunda beklenen anahtar kelimeler
QUESTION_KEYWORDS: dict[str, list[str]] = {
    "employee_id": ["calisan numar", "personel numar", "personel kimli", "emp-", "sicil",
                    "kimin adina", "hangi personel", "hangi calisan", "numaraniz"],
    "baslangic_tarihi": ["tarih", "baslangic", "bitis", "hangi gun", "ne zaman"],
    "bitis_tarihi": ["tarih", "baslangic", "bitis", "hangi gun", "ne zaman"],
    "donem": ["donem", "hangi ay", "hangi doneme", "orn", "ornek", "2026-"],
    "izin_tipi": ["tur", "yillik", "mazeret", "hastalik", "hangi izin", "ne tur"],
    "talep_id": ["talep numar", "lv-", "talep kimli", "hangi talep", "talebin kimli"],
    "yeni_brut_ucret": ["brut ucret", "yeni ucret", "yeni brut", "tutar"],
    "gerekce": ["gerekce", "neden", "sebep"],
    "telefon": ["telefon", "e-posta", "eposta", "adres", "iletisim", "hangi bilgi", "ne olarak"],
    "departman_adi": ["departman", "birim", "ekip", "hangi departman"],
    "medeni_durum": ["medeni", "ogrenim", "acil durum", "hangi bilgi", "hangi alan"],
}

PRESENCE_CHECK = {
    "employee_id": lambda f: bool(EMP_REF_RE.search(f)),
    "talep_id": lambda f: bool(LV_REF_RE.search(f)),
    "donem": lambda f: bool(re.search(MONTH_RE, f)),
    "baslangic_tarihi": lambda f: bool(DATE_RE.search(f)),
    "bitis_tarihi": lambda f: bool(DATE_RE.search(f)),
    "izin_tipi": lambda f: bool(IZIN_TIPI_RE.search(f)),
}


@pytest.fixture(scope="session")
def rfi(paired):
    return [(rec, meta) for rec, meta in paired if meta["decision"] == "request_for_info"]


@pytest.fixture(scope="session")
def rfi_missing(rfi):
    return [(rec, meta) for rec, meta in rfi if meta["missing_parameters"]]


# --------------------------------------------------------------------------
# Partisyon
# --------------------------------------------------------------------------

def test_request_for_info_partitions_cleanly(rfi):
    orphans = [
        meta["id"] for rec, meta in rfi
        if not meta["missing_parameters"] and not meta["confirmation_required"]
    ]
    assert not orphans, (
        f"ne eksik-bilgi ne onay-bekleme olan 'sahte' request_for_info: {orphans[:20]}"
    )


def test_request_for_info_never_contains_a_tool_call(rfi):
    for rec, meta in rfi:
        assert not any(True for _ in iter_tool_calls(rec["messages"])), (
            f"{meta['id']}: request_for_info içinde tool_call"
        )
        assert not has_tool_call(rec["messages"][-1]["content"])


# --------------------------------------------------------------------------
# Eksik parametre gerçekten yok
# --------------------------------------------------------------------------

def test_missing_parameters_are_genuinely_absent(rfi_missing):
    failures = []
    for rec, meta in rfi_missing:
        f = fold(user_blob(rec["messages"]))
        for p in meta["missing_parameters"]:
            check = PRESENCE_CHECK.get(p)
            if check and check(f):
                failures.append(f"{meta['id']}: '{p}' eksik deniyor ama metinde var")
    assert not failures, "\n  ".join(failures[:25])


def test_present_required_parameters_are_actually_present(rfi_missing):
    failures = []
    for rec, meta in rfi_missing:
        f = fold(user_blob(rec["messages"]))
        present_required = set(meta["required_parameters"]) - set(meta["missing_parameters"])
        for p in present_required:
            check = PRESENCE_CHECK.get(p)
            if check and not check(f):
                failures.append(f"{meta['id']} ({meta['target_tool']}): '{p}' eksik değil ama metinde yok")
    assert not failures, "\n  ".join(failures[:25])


# Bu intent'lerde kullanıcı "bilgimi güncelle" der; HANGİ alanın değişeceği
# şemada opsiyoneldir ama işlem için gereklidir → "eksik" olarak sorulur.
_WHICH_FIELD_INTENTS = {"update_contact", "update_information", "resolve_employee_identity"}


def test_missing_parameters_are_valid_and_mostly_required(rfi_missing, gen):
    for rec, meta in rfi_missing:
        tool = meta["target_tool"]
        props = set(gen.TOOLS[tool]["parameters"]["properties"])
        required = set(gen.TOOLS[tool]["parameters"].get("required", []))
        missing = set(meta["missing_parameters"])

        # (a) hiçbir eksik parametre şemada tanımsız olamaz
        assert missing <= props, (
            f"{meta['id']}: '{tool}' şemasında olmayan alan(lar) eksik deniyor: {missing - props}"
        )
        # (b) "hangi alan" intent'leri dışında: eksik ⊆ zorunlu (§10 — opsiyonel sorulmaz)
        if meta["intent"] not in _WHICH_FIELD_INTENTS:
            assert missing <= required, (
                f"{meta['id']}: opsiyonel alan(lar) eksik diye soruluyor: {missing - required} (§10)"
            )


# --------------------------------------------------------------------------
# Asistan sorusu doğru parametreyi hedefliyor
# --------------------------------------------------------------------------

def test_question_targets_the_missing_parameter(rfi_missing):
    failures = []
    for rec, meta in rfi_missing:
        q = fold(rec["messages"][-1]["content"])
        ok = any(
            any(kw in q for kw in QUESTION_KEYWORDS.get(p, [p]))
            for p in meta["missing_parameters"]
        )
        if not ok:
            failures.append(f"{meta['id']} missing={meta['missing_parameters']}: {rec['messages'][-1]['content'][:90]!r}")
    assert not failures, "Eksik parametreyi hedeflemeyen sorular:\n  " + "\n  ".join(failures[:25])


def test_identity_resolution_asks_for_id_when_name_given(paired):
    seen = 0
    for rec, meta in paired:
        if meta["intent"] != "resolve_employee_identity":
            continue
        seen += 1
        assert meta["decision"] == "request_for_info"
        q = fold(rec["messages"][-1]["content"])
        assert "emp" in q or "personel numar" in q or "numar" in q, (
            f"{meta['id']}: isimle gelen istekte personel numarası istenmemiş"
        )
    if seen == 0:
        pytest.skip("resolve_employee_identity örneği yok")


# --------------------------------------------------------------------------
# Onay-bekleme mesajı somut
# --------------------------------------------------------------------------

def test_confirmation_prompt_restates_the_concrete_operation(rfi):
    from conftest import user_blob as _ub

    weak = []
    for rec, meta in rfi:
        if not (meta["confirmation_required"] and not meta["missing_parameters"]):
            continue
        confirm = rec["messages"][-1]["content"]
        u = _ub(rec["messages"])
        # onay mesajı kullanıcının verdiği somut değerlerden en az birini tekrar etmeli
        emp = re.findall(r"EMP-\d+", u, re.I)
        lv = re.findall(r"LV-\d[\d-]*", u, re.I)
        anchors = emp + lv
        if anchors and not any(a.lower() in confirm.lower() for a in anchors):
            weak.append(f"{meta['id']}: onay mesajı {anchors} değerini tekrar etmiyor")
    assert not weak, "Somut olmayan onay mesajları:\n  " + "\n  ".join(weak[:20])
