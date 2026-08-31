# -*- coding: utf-8 -*-
"""
test_negative_space_coverage.py — ÜRETİCİ TASARIM UZAYININ tam örneklendiği (§15, §27, §35)
=================================================================================

Diğer testler "üretilen doğru mu" der. Bu dosya "üretilebilecek HER ŞEY
üretilmiş mi" der — kapsama analizi. Tasarım uzayında kör nokta olmamalı:
her tool çağrılmış, her slot havuzu değeri kullanılmış, her ret havuzu
işletilmiş, §27'deki hard-negative senaryolarının HEPSİ mevcut, hiçbir spec
"öksüz" (hiç örnek üretmeyen) değil.

Kapsam
------
* 22 tool'un TAMAMI en az bir kez çağrılmış.
* Her tool bir domain'e (DOMAIN_TOOLS) veya bir spec hedefine bağlı — erişilebilir.
* DATE_RANGES / MONTH_RANGES / DONEMLER / DONEM_YIL yüzeylerinin tamamı kullanılmış.
* GEREKCE_POOL / POZISYONLAR / AMOUNT_POOL / TALEP_IDS / PHONE_POOL / ACIL_KISI_POOL
  değerlerinin tamamı geçiyor.
* 6 REFUSAL_CORE havuzunun (plain/future/privacy/career/financial/unsupported)
  tamamı `cannot_answer` örneklerinde işletilmiş.
* Hiçbir spec öksüz değil (DIRECT / CANNOT / READ / MISSING / WRITE / MT_*).
* register × decision: her hücre dolu (6 × 4 = 24 hücre).
* domain × difficulty: her fonksiyonel hücre dolu (≥ 1 örnek).
* §27 hard-negative senaryoları: tool var ama çağırma / benzer-yanlış tool /
  eksik parametre / kapsam dışı / hassas erişim — hepsi temsil edilmiş.
* ≥ 10 train örneği olan her intent'in ≥ 1 val örneği var.
"""
from __future__ import annotations

from collections import Counter

import pytest

from conftest import iter_tool_calls

DECISIONS = ("tool_call", "direct", "request_for_info", "cannot_answer")
DIFFICULTIES = ("kolay", "orta", "zor", "cok_zor")
FUNCTIONAL_DOMAINS = ("izin_yonetimi", "maas_finans", "puantaj", "organizasyon",
                      "calisan_bilgileri", "ik_islemleri")


@pytest.fixture(scope="session")
def full_text(all_records):
    return " ".join(m["content"] for rec in all_records for m in rec["messages"])


@pytest.fixture(scope="session")
def called_tools(all_records):
    out: set[str] = set()
    for rec in all_records:
        for _, obj in iter_tool_calls(rec["messages"]):
            out.add(obj["name"])
    return out


# --------------------------------------------------------------------------
# Tool kapsaması
# --------------------------------------------------------------------------

def test_every_tool_is_called_at_least_once(called_tools, gen):
    missing = set(gen.TOOLS) - called_tools
    assert not missing, f"hiç çağrılmayan tool(lar): {sorted(missing)}"


def test_every_tool_is_reachable_via_a_domain_or_spec(gen):
    reachable = set()
    for names in gen.DOMAIN_TOOLS.values():
        reachable |= set(names)
    for spec_list in (gen.READ_SPECS, gen.MISSING_PARAM_SPECS, gen.WRITE_SPECS,
                      gen.MT_INFO_SPECS, gen.MULTI_STEP_SPECS):
        for s in spec_list:
            if "tool" in s:
                reachable.add(s["tool"])
    for s in gen.MULTI_INTENT_SPECS:
        reachable |= set(s["tools"])
    unreachable = set(gen.TOOLS) - reachable
    assert not unreachable, f"hiçbir domain/spec'e bağlı olmayan tool(lar): {unreachable}"


# --------------------------------------------------------------------------
# Slot havuzu kapsaması
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pool_name", [
    "DATE_RANGES", "MONTH_RANGES", "DONEMLER", "DONEM_YIL",
])
def test_every_surface_pool_entry_is_used(full_text, gen, pool_name):
    pool = getattr(gen, pool_name)
    surfaces = [row[0] for row in pool]
    unused = [s for s in surfaces if s not in full_text]
    assert not unused, f"{pool_name}: kullanılmayan yüzey(ler) {unused}"


@pytest.mark.parametrize("pool_name", [
    "GEREKCE_POOL", "POZISYONLAR", "AMOUNT_POOL", "TALEP_IDS", "PHONE_POOL", "ACIL_KISI_POOL",
])
def test_every_value_pool_entry_is_used(full_text, gen, pool_name):
    pool = getattr(gen, pool_name)
    unused = [v for v in pool if str(v) not in full_text]
    assert not unused, f"{pool_name}: kullanılmayan değer(ler) {unused}"


# --------------------------------------------------------------------------
# Ret havuzu kapsaması
# --------------------------------------------------------------------------

def test_every_refusal_pool_is_exercised(all_meta, gen):
    pool_by_intent = {s["intent"]: s["pool"] for s in gen.CANNOT_INTENTS}
    used = {pool_by_intent[m["intent"]] for m in all_meta
            if m["decision"] == "cannot_answer" and m["intent"] in pool_by_intent}
    all_pools = set(gen.REFUSAL_CORE)
    assert used == all_pools, f"işletilmeyen ret havuzu: {all_pools - used}"


# --------------------------------------------------------------------------
# Öksüz spec yok
# --------------------------------------------------------------------------

@pytest.mark.parametrize("spec_attr", [
    "DIRECT_INTENTS", "CANNOT_INTENTS", "READ_SPECS", "MISSING_PARAM_SPECS",
    "WRITE_SPECS", "MT_INFO_SPECS", "MULTI_STEP_SPECS", "MT_DIRECT_SPECS", "MULTI_INTENT_SPECS",
])
def test_no_orphan_specs(all_meta, gen, spec_attr):
    present = {m["intent"] for m in all_meta}
    orphans = [s["intent"] for s in getattr(gen, spec_attr) if s["intent"] not in present]
    assert not orphans, f"{spec_attr}: hiç örnek üretmeyen spec(ler) {orphans}"


# --------------------------------------------------------------------------
# Çapraz kapsama
# --------------------------------------------------------------------------

def test_register_by_decision_grid_is_fully_populated(all_meta):
    grid = Counter((m["register"], m["decision"]) for m in all_meta)
    registers = sorted({m["register"] for m in all_meta})
    holes = [f"{r} × {d}" for r in registers for d in DECISIONS if grid[(r, d)] == 0]
    assert not holes, f"register × decision ızgarasında boş hücre: {holes}"


def test_domain_by_difficulty_grid_is_populated(all_meta):
    grid = Counter((m["domain"], m["difficulty"]) for m in all_meta)
    holes = [f"{d} × {x}" for d in FUNCTIONAL_DOMAINS for x in DIFFICULTIES if grid[(d, x)] == 0]
    assert not holes, f"domain × difficulty ızgarasında boş hücre: {holes}"


# --------------------------------------------------------------------------
# §27 hard-negative senaryoları
# --------------------------------------------------------------------------

def test_all_hard_negative_scenarios_are_represented(all_meta, gen):
    future_intents = {s["intent"] for s in gen.CANNOT_INTENTS if s["pool"] == "future"}
    privacy_intents = {s["intent"] for s in gen.CANNOT_INTENTS if s["pool"] == "privacy"}

    scenarios = {
        # "tool var ama çağırma": direct örneklerinde de tools listesi dolu
        "tool_available_but_direct":
            sum(1 for m in all_meta if m["decision"] == "direct"),
        # "benzer fakat yanlış tool": get_izin_gecmisi hedefi (get_izin_bakiyesi ile karışır)
        "similar_but_correct_tool_chosen":
            sum(1 for m in all_meta if m.get("target_tool") == "get_izin_gecmisi" and m["decision"] == "tool_call"),
        # "parametre eksik"
        "missing_required_parameter":
            sum(1 for m in all_meta if m["decision"] == "request_for_info" and "employee_id" in m["missing_parameters"]),
        # "tool kapsamı dışında"
        "out_of_scope_future":
            sum(1 for m in all_meta if m["decision"] == "cannot_answer" and m["intent"] in future_intents),
        # "hassas erişim"
        "sensitive_access_denied":
            sum(1 for m in all_meta if m["decision"] == "cannot_answer" and m["intent"] in privacy_intents),
    }
    weak = {k: v for k, v in scenarios.items() if v < 10}
    assert not weak, f"§27 hard-negative senaryoları zayıf temsil edilmiş: {weak}"


def test_direct_examples_still_carry_a_tool_inventory(paired):
    """§4/§27: tool erişimi var diye çağırmamak öğretilir → direct örneklerinde de
    dolu bir `tools` listesi bulunmalı."""
    for rec, meta in paired:
        if meta["decision"] != "direct":
            continue
        assert len(rec["tools"]) >= 3, f"{meta['id']}: direct örneğinde yalnızca {len(rec['tools'])} tool"


# --------------------------------------------------------------------------
# Train/val kapsama dengesi
# --------------------------------------------------------------------------

def test_sizable_intents_appear_in_validation_split(train_meta, val_meta):
    tr = Counter(m["intent"] for m in train_meta)
    va = Counter(m["intent"] for m in val_meta)
    missing = [i for i, c in tr.items() if c >= 10 and va[i] == 0]
    assert not missing, f"≥10 train örneği olup val'da hiç olmayan intent'ler: {missing}"


def test_every_decision_and_domain_appears_in_both_splits(train_meta, val_meta):
    for key in ("decision", "domain"):
        tr = {m[key] for m in train_meta}
        va = {m[key] for m in val_meta}
        assert tr == va, f"'{key}' train/val arasında farklı: yalnız train {tr - va}, yalnız val {va - tr}"
