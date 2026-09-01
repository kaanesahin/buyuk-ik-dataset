# -*- coding: utf-8 -*-
"""
test_privacy_and_safety.py — SENTETİK veri ve gizlilik/yetki davranışı (§2, §19, §27, §30)
====================================================================================

Büyük İK çalışanların finansal ve kişisel verilerini içerdiği için dataset:
  (a) gerçek PII içermemeli — her şey sentetik,
  (b) başkasının özel bilgisini isteyen talepleri tool'a zorlamamalı.

Kapsam
------
* Gerçek TC kimlik no kalıbı yok (11 hane, geçerli algoritma).
* IBAN kalıbı yok (`TR` + 24 hane).
* Tüm ``employee_id``'ler ``EMP-\\d{4}`` ve makul aralıkta (1000–6999).
* Tüm ``talep_id``'ler ``LV-\\d{4}-\\d{3,4}``.
* Telefon numaraları yalnızca sentetik havuzdan; "555 55 55" gibi bariz sahte.
* E-posta adresleri yalnızca ``ornek.com`` / ``example.com`` alan adlarında.
* Kişi adları yalnızca üreticinin ``FIRST_NAMES`` havuzundan.
* Başka çalışanın maaş/izin/iletişim bilgisi isteyen intent'ler → ``cannot_answer``
  (veya yetki kontrolü tool'una yönlenir), asla doğrudan veri döndürmez.
* Geleceğe yönelik / doğrulanamaz tahmin isteyen intent'ler → ``cannot_answer``.
* ``check_employee_access`` yalnızca yetki sorgusu için kullanılıyor; assistant
  kendi kafasından "yetkiniz var/yok" demiyor.
* Yıkıcı işlemler (kalıcı silme, toplu onay) → ``cannot_answer``.
"""
from __future__ import annotations

import re

import pytest

from conftest import EMP_RE, LV_RE, assistant_turns, fold, iter_tool_calls

TCKN_RE = re.compile(r"(?<!\d)\d{11}(?!\d)")
IBAN_RE = re.compile(r"\bTR\d{24}\b", re.IGNORECASE)
# Kullanıcı yazım-hatalı turlarda 'emp-1042' gibi küçük harf yazabilir — biçim
# kontrolü büyük/küçük harften bağımsız.
EMP_STRICT_RE = re.compile(r"^EMP-\d{4}$", re.IGNORECASE)
LV_STRICT_RE = re.compile(r"^LV-\d{4}-\d{3,4}$", re.IGNORECASE)
LONG_DIGIT_RUN_RE = re.compile(r"(?<!\d)\d{10,}(?!\d)")

PRIVACY_INTENTS = {
    "coworker_salary", "compare_others_salaries", "ceo_bank_details", "others_home_address",
    "all_employees_id_numbers", "coworker_leave_records", "highest_paid_employee",
    "why_coworker_left", "coworker_timesheet", "coworker_position_history",
    "approve_on_behalf_of_manager",
}
FUTURE_INTENTS = {
    "predict_company_inflation", "predict_economic_growth", "predict_stock_market",
    "predict_future_raises", "predict_future_promotions", "predict_layoffs",
    "predict_exact_future_leave", "guarantee_leave_approval", "future_company_headcount",
    "predict_future_overtime", "predict_own_leave_rejection",
}
DESTRUCTIVE_INTENTS = {
    "permanently_delete_record", "bulk_process_all_requests", "approve_own_leave",
    "reset_manager_credentials", "set_performance_score", "edit_own_timesheet",
}


@pytest.fixture(scope="session")
def full_text(all_records):
    """Tüm mesaj içeriklerinin birleşimi (hem user hem assistant)."""
    return "\n".join(m["content"] for r in all_records for m in r["messages"])


# --------------------------------------------------------------------------
# Gerçek PII kalıpları
# --------------------------------------------------------------------------

def test_no_tckn_like_numbers(full_text):
    hits = set(TCKN_RE.findall(full_text))
    assert not hits, f"11 haneli (TC kimlik benzeri) sayı(lar): {list(hits)[:5]}"


def test_no_iban_patterns(full_text):
    hits = set(IBAN_RE.findall(full_text))
    assert not hits, f"IBAN kalıbı: {list(hits)[:5]}"


def test_no_long_digit_runs(full_text):
    hits = set(LONG_DIGIT_RUN_RE.findall(full_text))
    assert not hits, f"10+ haneli sayı dizisi (hesap no şüphesi): {list(hits)[:5]}"


# --------------------------------------------------------------------------
# Sentetik kimlik biçimleri
# --------------------------------------------------------------------------

def test_all_employee_ids_are_synthetic_format(full_text):
    for emp in set(EMP_RE.findall(full_text)):
        assert EMP_STRICT_RE.match(emp), f"beklenmeyen employee_id biçimi: {emp!r}"
        num = int(emp.split("-")[1])
        assert 1000 <= num <= 6999, f"employee_id aralık dışı: {emp}"


def test_all_request_ids_are_synthetic_format(full_text):
    for lv in set(LV_RE.findall(full_text)):
        assert LV_STRICT_RE.match(lv), f"beklenmeyen talep_id biçimi: {lv!r}"


def test_phone_numbers_come_from_synthetic_pool(all_records, gen):
    pool = set(gen.PHONE_POOL)
    for rec in all_records:
        for _, obj in iter_tool_calls(rec["messages"]):
            for k in ("telefon", "acil_durum_telefonu"):
                v = obj["arguments"].get(k)
                if v is not None:
                    assert v in pool, f"tanımsız telefon numarası argümanı: {v!r}"


def test_emails_only_use_example_domains(full_text):
    emails = re.findall(r"[\w.+-]+@[\w.-]+\.\w+", full_text)
    bad = [e for e in emails if not e.lower().endswith(("ornek.com", "example.com"))]
    assert not bad, f"sentetik olmayan e-posta alan adları: {set(bad)}"


def test_person_names_are_from_generator_pool(all_records, gen):
    """`resolve_employee_identity` gibi intent'lerde geçen isimler havuzdan olmalı."""
    pool = set(gen.FIRST_NAMES)
    # kullanıcı turlarında geçen "{isim}'in ..." kalıbını yakala
    name_re = re.compile(r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)(?:'[a-zçğıöşü]{1,3})?\s+(?:adlı|isimli|adında)\b")
    for rec in all_records:
        for u in (m["content"] for m in rec["messages"] if m["role"] == "user"):
            for m in name_re.findall(u):
                assert m in pool, f"havuz dışı kişi adı kullanılmış: {m!r}"


# --------------------------------------------------------------------------
# Gizlilik davranışı — başkasının verisi
# --------------------------------------------------------------------------

def test_privacy_intents_never_return_data(paired):
    for rec, meta in paired:
        if meta["intent"] not in PRIVACY_INTENTS:
            continue
        assert meta["decision"] == "cannot_answer", (
            f"{meta['id']} ({meta['intent']}): başkasının verisi isteniyor ama decision={meta['decision']}"
        )
        assert not any(True for _ in iter_tool_calls(rec["messages"])), (
            f"{meta['id']}: gizli veri isteğinde tool_call yapılmış"
        )


def test_future_prediction_intents_are_refused(paired):
    for rec, meta in paired:
        if meta["intent"] not in FUTURE_INTENTS:
            continue
        assert meta["decision"] == "cannot_answer", (
            f"{meta['id']} ({meta['intent']}): geleceğe dönük tahmin ama decision={meta['decision']}"
        )


def test_destructive_operations_are_refused(paired):
    for rec, meta in paired:
        if meta["intent"] not in DESTRUCTIVE_INTENTS:
            continue
        assert meta["decision"] == "cannot_answer", (
            f"{meta['id']} ({meta['intent']}): desteklenmeyen/yıkıcı işlem ama decision={meta['decision']}"
        )


def test_authorization_is_only_asserted_via_the_access_tool(paired):
    """Assistant düz metinde 'yetkiniz var' / 'yetkiniz yok' demez; bunu check_employee_access yapar."""
    claim_re = re.compile(r"yetki\w*\s+(var|yok|bulunmuyor|mevcut)", re.IGNORECASE)
    for rec, meta in paired:
        if meta["decision"] == "tool_call" and meta.get("target_tool") == "check_employee_access":
            continue
        for a in assistant_turns(rec["messages"]):
            m = claim_re.search(fold(a))
            assert not m, f"{meta['id']}: assistant yetki durumu uyduruyor: …{a[max(0,m.start()-30):m.end()+10]}…"


def test_check_access_tool_used_for_authorization_questions(paired):
    """'... erişme yetkim var mı' tipi sorular check_employee_access'e yönlenmeli."""
    for rec, meta in paired:
        if meta["intent"] != "check_access":
            continue
        assert meta["decision"] == "tool_call"
        called = {obj["name"] for _, obj in iter_tool_calls(rec["messages"])}
        assert called == {"check_employee_access"}, f"{meta['id']}: yetki sorusu {called} ile yanıtlanmış"


# --------------------------------------------------------------------------
# Kendi verisi — izinli erişim
# --------------------------------------------------------------------------

def test_self_service_reads_are_allowed_when_id_given(paired):
    """Kullanıcı kendi EMP-ID'sini verdiğinde maaş/izin sorgusu ENGELLENMEZ."""
    ok = 0
    for rec, meta in paired:
        if meta["intent"] in ("get_salary", "get_leave_balance") and meta["decision"] == "tool_call":
            ok += 1
    assert ok >= 10, f"yalnızca {ok} 'kendi maaş/izin bilgisi' tool_call örneği — self-service az temsil edilmiş"
