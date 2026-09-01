# -*- coding: utf-8 -*-
"""
test_tool_schema.py — TOOL ŞEMA tasarımı ve tutarlılığı (When2Call §14)
=====================================================================

Her tool, gerçek bir kurumsal İK API'sini temsil edecek kadar anlamlı ve
tutarlı bir JSON-Schema alt kümesiyle tanımlanmış olmalı.

Kapsam
------
* Her tool: ``name`` (snake_case), ``description`` (anlamlı), ``parameters``.
* ``parameters.type == "object"``, ``properties`` dict, ``required`` listesi.
* ``required`` ⊆ ``properties`` anahtarları.
* Her property: ``type`` (geçerli JSON tipi) + ``description`` (boş değil).
* ``enum`` alanları: boş olmayan, tekrarsız, string listesi; tip ``string``.
* Tüm tool adları benzersiz; ``data/*_tools.json`` envanteri ile her örneğin
  gömülü tool tanımları BİREBİR aynı (tek doğruluk kaynağı).
* Envanterde 22 tool var, isimler `generate_dataset.TOOLS` ile eşleşiyor.
* Politika sabitleri (`CONFIRMATION_REQUIRED`, `DOMAIN_TOOLS`, `CONFUSABLE`)
  yalnızca tanımlı tool'lara atıfta bulunuyor.
"""
from __future__ import annotations

import re

import pytest

NAME_RE = re.compile(r"^[a-z][a-z0-9_]+$")
JSON_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}
EXPECTED_TOOL_COUNT = 22


# --------------------------------------------------------------------------
# Envanter dosyası
# --------------------------------------------------------------------------

def test_inventory_has_expected_tool_count(tools_inventory):
    assert len(tools_inventory) == EXPECTED_TOOL_COUNT, (
        f"tools.json {len(tools_inventory)} tool içeriyor, {EXPECTED_TOOL_COUNT} bekleniyordu"
    )


def test_inventory_tool_names_unique(tools_inventory):
    names = [t["name"] for t in tools_inventory]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"tools.json içinde tekrar eden tool adı: {dupes}"


def test_inventory_matches_generator_TOOLS(tools_inventory, gen):
    inv = {t["name"]: t for t in tools_inventory}
    assert set(inv) == set(gen.TOOLS), (
        f"tools.json ile generate_dataset.TOOLS farklı:\n"
        f"  yalnız tools.json: {set(inv) - set(gen.TOOLS)}\n"
        f"  yalnız TOOLS     : {set(gen.TOOLS) - set(inv)}"
    )
    for name, tool in inv.items():
        assert tool == gen.TOOLS[name], f"'{name}' tanımı tools.json ile TOOLS arasında farklı"


# --------------------------------------------------------------------------
# Şema geçerliliği
# --------------------------------------------------------------------------

class TestToolSchema:
    """Her tool şeması için ayrı iddialar."""

    @pytest.fixture(autouse=True)
    def _tools(self, tools_inventory):
        self.tools = tools_inventory

    def test_names_are_snake_case(self):
        bad = [t["name"] for t in self.tools if not NAME_RE.match(t["name"])]
        assert not bad, f"snake_case olmayan tool adları: {bad}"

    def test_descriptions_are_meaningful(self):
        for t in self.tools:
            d = t.get("description", "")
            assert isinstance(d, str) and len(d.strip()) >= 20, (
                f"'{t['name']}' açıklaması çok kısa/eksik: {d!r}"
            )

    def test_parameters_block_shape(self):
        for t in self.tools:
            p = t["parameters"]
            assert p.get("type") == "object", f"'{t['name']}': parameters.type != object"
            assert isinstance(p.get("properties"), dict), f"'{t['name']}': properties dict değil"
            assert isinstance(p.get("required", []), list), f"'{t['name']}': required liste değil"

    def test_required_is_subset_of_properties(self):
        for t in self.tools:
            props = set(t["parameters"]["properties"])
            req = set(t["parameters"].get("required", []))
            assert req <= props, (
                f"'{t['name']}': required içinde properties'te olmayan alan(lar) {req - props}"
            )

    def test_every_property_has_type_and_description(self):
        for t in self.tools:
            for pname, pdef in t["parameters"]["properties"].items():
                assert pdef.get("type") in JSON_TYPES, (
                    f"'{t['name']}.{pname}': geçersiz tip {pdef.get('type')!r}"
                )
                assert isinstance(pdef.get("description"), str) and pdef["description"].strip(), (
                    f"'{t['name']}.{pname}': description boş"
                )

    def test_enum_fields_are_clean(self):
        for t in self.tools:
            for pname, pdef in t["parameters"]["properties"].items():
                if "enum" not in pdef:
                    continue
                enum = pdef["enum"]
                assert isinstance(enum, list) and enum, f"'{t['name']}.{pname}': enum boş/liste değil"
                assert all(isinstance(v, str) for v in enum), (
                    f"'{t['name']}.{pname}': enum string olmayan değer içeriyor"
                )
                assert len(enum) == len(set(enum)), f"'{t['name']}.{pname}': enum tekrar içeriyor"
                assert pdef["type"] == "string", f"'{t['name']}.{pname}': enum var ama type != string"

    def test_at_least_one_required_param_per_tool(self):
        # her İK aracı en az bir bağlam parametresi ister (employee_id, departman_adi, talep_id...)
        for t in self.tools:
            assert t["parameters"].get("required"), (
                f"'{t['name']}': hiç zorunlu parametresi yok — muhtemelen tasarım hatası"
            )


# --------------------------------------------------------------------------
# Politika sabitleri tanımlı tool'lara atıfta bulunuyor mu
# --------------------------------------------------------------------------

def test_confirmation_required_tools_exist(gen):
    unknown = gen.CONFIRMATION_REQUIRED - set(gen.TOOLS)
    assert not unknown, f"CONFIRMATION_REQUIRED tanımsız tool'lara atıf yapıyor: {unknown}"


def test_write_tools_all_require_confirmation(gen):
    assert gen.WRITE_TOOLS == gen.CONFIRMATION_REQUIRED, (
        "WRITE_TOOLS ile CONFIRMATION_REQUIRED aynı olmalı (politika: her yazma onay ister)"
    )


def test_write_tool_descriptions_mention_confirmation(gen):
    for name in gen.WRITE_TOOLS:
        desc = gen.TOOLS[name]["description"].lower()
        assert "onay" in desc, f"'{name}' açıklamasında onay gerekliliği belirtilmemiş"


def test_domain_tools_reference_valid_tools(gen):
    for domain, names in gen.DOMAIN_TOOLS.items():
        unknown = set(names) - set(gen.TOOLS)
        assert not unknown, f"DOMAIN_TOOLS['{domain}'] tanımsız tool içeriyor: {unknown}"


def test_confusable_map_references_valid_tools(gen):
    for key, neighbours in gen.CONFUSABLE.items():
        assert key in gen.TOOLS, f"CONFUSABLE anahtarı '{key}' tanımlı bir tool değil"
        unknown = set(neighbours) - set(gen.TOOLS)
        assert not unknown, f"CONFUSABLE['{key}'] tanımsız tool içeriyor: {unknown}"
        assert key not in neighbours, f"CONFUSABLE['{key}'] kendini çeldirici olarak listeliyor"


def test_read_and_write_tools_are_disjoint_by_naming(gen):
    # get_* okuma, create_/update_/cancel_* yazma olmalı
    for name in gen.TOOLS:
        if name.startswith("get_") or name.startswith("check_"):
            assert name not in gen.WRITE_TOOLS, f"'{name}' get_/check_ ama WRITE_TOOLS'ta"
        if name.split("_")[0] in {"create", "update", "cancel"}:
            assert name in gen.WRITE_TOOLS, f"'{name}' değişiklik fiili ama WRITE_TOOLS'ta değil"
