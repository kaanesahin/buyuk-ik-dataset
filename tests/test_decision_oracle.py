# -*- coding: utf-8 -*-
"""
test_decision_oracle.py — HER etiket İLK İLKELERDEN yeniden türetilir (§4, §5, §10, §12, §36)
====================================================================================

`test_decision_semantics.py` "davranış etikete uyuyor mu" der. Bu dosya daha
sert bir soru sorar: **etiketin KENDİSİ doğru mu?**

When2Call karar süreci (§36) bir ORAKEL olarak yeniden uygulanır ve 3000 örneğin
TAMAMINA karşı çalıştırılır. Orakel şunları bilir:

  * hangi intent hangi üretici spec havuzundan gelir (DIRECT / CANNOT / READ /
    MISSING / WRITE / MULTI_STEP / MT_*),
  * bir tool'un zorunlu parametreleri kullanıcı metninden çıkarılabiliyor mu,
  * işlem WRITE mı ve onay bekliyor mu.

Bundan bağımsız olarak "doğru karar ne olmalı" hesaplanır; sonra meta'daki
etiketle karşılaştırılır. TEK bir örnekte bile uyuşmazlık = etiketleme hatası.

Kapsam
------
* intent → izin verilen karar kümesi (üretici spec havuzlarından türetilir);
  her örneğin ``decision`` bu kümede.
* `tool_call` (WRITE değil): TÜM zorunlu parametreler kullanıcı metninden
  çıkarılabiliyor (aksi halde `request_for_info` olmalıydı).
* `request_for_info`: YA en az bir zorunlu parametre metinde yok YA da WRITE +
  onay bekliyor — "sahte" (her şey var ama yine de sormuş) örnek yok.
* `tool_call` + WRITE: mutlaka bir ONAY turu (son kullanıcı turu) var.
* `direct`: asistan bilgi verir — ret işareti yok, tool_call yok.
* `cannot_answer`: asistan reddeder — bilgi/çağrı yok.
* Sınır: parametreler yeterliyken `request_for_info` YALNIZCA WRITE+onay için geçerli.
* Sınır: parametre eksikken `tool_call` HER ZAMAN hatadır.
"""
from __future__ import annotations

import re

import pytest

from conftest import fold, iter_tool_calls, user_blob

MONTHS = r"ocak|subat|mart|nisan|mayis|haziran|temmuz|agustos|eylul|ekim|kasim|aralik"
_EMP_REF = re.compile(r"emp-?\d+|sicil no \d+|\d{3,4} numar|numaram \d|personel numaram")
_LV_REF = re.compile(r"lv-?\d")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}[/.]\d{1,2}|" + MONTHS + r"|ceyrek|arasi")
_PERIOD = re.compile(MONTHS + r"|\d{4}-\d{2}|\b20\d\d\b|ceyrek")
_LEAVE_TYPE = re.compile(r"yillik|senelik|mazeret|hastalik|saglik|rapor")
_AMOUNT = re.compile(r"\d{5,}")
_MEDENI = re.compile(r"bekar|evli|\bdul\b|bosan")
_PHONE = re.compile(r"0\d{3}")

# Kullanıcı metninden çıkarılabilirlik kontrolleri. Anahtar yoksa "her zaman var"
# sayılır (serbest metin: gerekce/pozisyon/departman/adres — başka testlerde denetlenir).
_PRESENCE = {
    "employee_id": lambda f: bool(_EMP_REF.search(f)),
    "requester_id": lambda f: bool(_EMP_REF.search(f)),
    "hedef_employee_id": lambda f: bool(_EMP_REF.search(f)),
    "talep_id": lambda f: bool(_LV_REF.search(f)),
    "baslangic_tarihi": lambda f: bool(_DATE.search(f)),
    "bitis_tarihi": lambda f: bool(_DATE.search(f)),
    "donem": lambda f: bool(_PERIOD.search(f)),
    "izin_tipi": lambda f: bool(_LEAVE_TYPE.search(f)),
    "yeni_brut_ucret": lambda f: bool(_AMOUNT.search(f)),
    "medeni_durum": lambda f: bool(_MEDENI.search(f)),
    "telefon": lambda f: bool(_PHONE.search(f)),
}

@pytest.fixture(scope="session")
def intent_expected_decisions(gen):
    direct = {s["intent"] for s in gen.DIRECT_INTENTS}
    cannot = {s["intent"] for s in gen.CANNOT_INTENTS}
    call = (
        {s["intent"] for s in gen.READ_SPECS}
        | {s["intent"] for s in gen.MT_INFO_SPECS}
        | {s["intent"] for s in gen.MULTI_INTENT_SPECS}
        | {s["intent"] for s in gen.MULTI_STEP_SPECS}
    )
    ask = {s["intent"] for s in gen.MISSING_PARAM_SPECS}
    write = {s["intent"] for s in gen.WRITE_SPECS}

    def expected(intent: str) -> set[str]:
        if intent in direct:
            return {"direct"}
        if intent in cannot:
            return {"cannot_answer"}
        out: set[str] = set()
        if intent in call:
            out.add("tool_call")
        if intent in ask:
            out.add("request_for_info")
        if intent in write:
            out |= {"request_for_info", "tool_call"}
        return out

    return expected


def _params_all_present(required, folded_user: str) -> bool:
    for p in required:
        check = _PRESENCE.get(p)
        if check and not check(folded_user):
            return False
    return True


# --------------------------------------------------------------------------
# 1) intent → karar kümesi
# --------------------------------------------------------------------------

def test_every_decision_is_permitted_by_its_intent(all_meta, intent_expected_decisions):
    failures = []
    for m in all_meta:
        allowed = intent_expected_decisions(m["intent"])
        assert allowed, f"{m['id']}: '{m['intent']}' hiçbir üretici havuzuna eşlenemedi"
        if m["decision"] not in allowed:
            failures.append(f"{m['id']}: intent={m['intent']} decision={m['decision']} beklenen∈{sorted(allowed)}")
    assert not failures, "Orakel karar uyuşmazlıkları (etiketleme hatası):\n  " + "\n  ".join(failures[:25])


# --------------------------------------------------------------------------
# 2) tool_call ⇒ parametre yeterliliği
# --------------------------------------------------------------------------

def test_read_tool_calls_have_sufficient_parameters(paired):
    failures = []
    for rec, meta in paired:
        if meta["decision"] != "tool_call" or meta.get("is_write"):
            continue
        f = fold(user_blob(rec["messages"]))
        if not _params_all_present(meta["required_parameters"], f):
            missing = [p for p in meta["required_parameters"]
                       if _PRESENCE.get(p) and not _PRESENCE[p](f)]
            failures.append(f"{meta['id']} ({meta['target_tool']}): {missing} metinden çıkarılamıyor ama tool_call")
    assert not failures, (
        "Yetersiz parametreyle tool_call (request_for_info olmalıydı):\n  " + "\n  ".join(failures[:25])
    )


def test_parameter_insufficiency_never_yields_a_read_call(paired):
    """Katı sınır: bir zorunlu parametre metinde yoksa, WRITE olmayan bir örnek
    ASLA tool_call ile bitmez."""
    for rec, meta in paired:
        if meta["decision"] != "tool_call" or meta.get("is_write"):
            continue
        f = fold(user_blob(rec["messages"]))
        for p in meta["required_parameters"]:
            check = _PRESENCE.get(p)
            assert not (check and not check(f)), (
                f"{meta['id']}: '{p}' yok ama okuma çağrısı yapılmış — halüsinasyon sınırı ihlali"
            )


# --------------------------------------------------------------------------
# 3) request_for_info ⇒ gerçekten bir eksik var
# --------------------------------------------------------------------------

def test_request_for_info_is_justified(paired):
    failures = []
    for rec, meta in paired:
        if meta["decision"] != "request_for_info":
            continue
        f = fold(user_blob(rec["messages"]))
        missing = set(meta["missing_parameters"])
        if missing:
            genuinely_absent = [p for p in missing if _PRESENCE.get(p) and not _PRESENCE[p](f)]
            if not genuinely_absent and any(p in _PRESENCE for p in missing):
                failures.append(f"{meta['id']}: her şey metinde var ama sormuş (missing={sorted(missing)})")
        elif not meta.get("confirmation_required"):
            failures.append(f"{meta['id']}: ne eksik param ne onay-bekleme — gerekçesiz request_for_info")
    assert not failures, "Gerekçesiz request_for_info:\n  " + "\n  ".join(failures[:25])


def test_sufficient_params_only_stall_for_write_confirmation(paired):
    """Parametreler yeterliyken karar `request_for_info` ise, bunun TEK meşru
    nedeni WRITE + onay beklemesidir."""
    for rec, meta in paired:
        if meta["decision"] != "request_for_info" or meta["missing_parameters"]:
            continue
        assert meta.get("confirmation_required") and meta.get("is_write"), (
            f"{meta['id']}: eksik yok ama request_for_info — WRITE onayı da değil"
        )


# --------------------------------------------------------------------------
# 4) WRITE yürütme ⇒ onay turu
# --------------------------------------------------------------------------

def test_write_execution_requires_an_ack_turn(paired):
    for rec, meta in paired:
        if not (meta["decision"] == "tool_call" and meta.get("is_write")):
            continue
        msgs = rec["messages"]
        assert len(msgs) >= 4, f"{meta['id']}: WRITE yürütme {len(msgs)} tur (onay turu için ≥4 gerekir)"
        assert msgs[-2]["role"] == "user", f"{meta['id']}: yürütmeden önce kullanıcı onay turu yok"


# --------------------------------------------------------------------------
# 5) direct / cannot_answer — orakel sınırı: bu intent'ler ASLA tool çağırmaz
# --------------------------------------------------------------------------

def test_knowledge_and_refusal_intents_never_produce_a_tool_call(paired, intent_expected_decisions):
    """DIRECT ve CANNOT havuzundan gelen HİÇBİR intent, hiçbir örnekte tool_call
    üretmemeli — bu intent'ler için üretebilecek bir tool zaten `tools` listesinde
    olsa bile."""
    for rec, meta in paired:
        allowed = intent_expected_decisions(meta["intent"])
        if allowed not in ({"direct"}, {"cannot_answer"}):
            continue
        calls = [obj["name"] for _, obj in iter_tool_calls(rec["messages"])]
        assert not calls, (
            f"{meta['id']} ({meta['intent']}, beklenen {sorted(allowed)}): tool çağrısı yapmış {calls}"
        )


def test_dual_pattern_intents_never_appear_as_direct_or_cannot(paired, intent_expected_decisions):
    """Tool'a hizmet eden intent'ler ASLA direct/cannot_answer olarak etiketlenmemeli."""
    for _, meta in paired:
        allowed = intent_expected_decisions(meta["intent"])
        if allowed in ({"direct"}, {"cannot_answer"}):
            continue
        assert meta["decision"] not in ("direct", "cannot_answer"), (
            f"{meta['id']} ({meta['intent']}): tool intent'i '{meta['decision']}' olarak etiketlenmiş"
        )


# --------------------------------------------------------------------------
# 6) Toplu tutarlılık — orakel tüm sette çalışır
# --------------------------------------------------------------------------

def test_oracle_agreement_is_total(all_meta, paired, intent_expected_decisions):
    """Hepsini tek testte topla: orakel kararı ↔ etiket, %100 uyum."""
    disagree = 0
    for rec, meta in paired:
        allowed = intent_expected_decisions(meta["intent"])
        if meta["decision"] not in allowed:
            disagree += 1
            continue
        f = fold(user_blob(rec["messages"]))
        if meta["decision"] == "tool_call" and not meta.get("is_write"):
            if not _params_all_present(meta["required_parameters"], f):
                disagree += 1
        elif meta["decision"] == "request_for_info" and not meta["missing_parameters"]:
            if not (meta.get("confirmation_required") and meta.get("is_write")):
                disagree += 1
    assert disagree == 0, f"{disagree}/{len(paired)} örnekte orakel etiketle çelişiyor"
