# -*- coding: utf-8 -*-
"""Yapısal geçerlilik + Qwen tool_call biçimi + şema uyumu."""
import json
import re

from conftest import TC_RE


def test_jsonl_roles_and_alternation(train, val, hard_eval):
    for split, recs in (("train", train), ("val", val), ("hard_eval", hard_eval)):
        for i, r in enumerate(recs):
            msgs = r["messages"]
            assert isinstance(r["tools"], list) and r["tools"], f"{split}[{i}] tools boş"
            assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "assistant"
            for j, m in enumerate(msgs):
                assert m["role"] in ("user", "assistant", "tool")
                assert isinstance(m["content"], str) and m["content"].strip()
                if j and m["role"] == "user":
                    assert msgs[j - 1]["role"] != "user", f"{split}[{i}] ardışık user"


def test_tool_message_follows_assistant_call(train):
    for r in train:
        msgs = r["messages"]
        for j, m in enumerate(msgs):
            if m["role"] != "tool":
                continue
            k = j - 1
            while k >= 0 and msgs[k]["role"] == "tool":
                k -= 1
            assert k >= 0 and msgs[k]["role"] == "assistant" and "<tool_call>" in msgs[k]["content"]


def test_toolcall_block_is_strict_and_alone(train, val):
    for recs in (train, val):
        for r in recs:
            for m in r["messages"]:
                if m["role"] != "assistant" or "<tool_call>" not in m["content"]:
                    continue
                assert m["content"].count("<tool_call>") == len(TC_RE.findall(m["content"]))
                assert not m["content"].split("<tool_call>")[0].strip(), "blok öncesi düz metin"
                for b in TC_RE.findall(m["content"]):
                    o = json.loads(b)
                    assert set(o) == {"name", "arguments"} and isinstance(o["arguments"], dict)
                    assert "\n" not in b, "tool_call JSON tek satır olmalı"


def test_schema_conformance(train, val, hard_eval, catalog, calls_of):
    by = {t.name: t for t in catalog}
    for split, recs in (("train", train), ("val", val), ("hard_eval", hard_eval)):
        for r in recs:
            names = {t["name"] for t in r["tools"]}
            for c in calls_of(r):
                assert c["name"] in names, f"{split}: {c['name']} aday listede yok"
                t = by[c["name"]]
                props = {p.name for p in t.params}
                for ak, av in c["arguments"].items():
                    ok = ak in props or re.match(r"(new_)?(start|end)_date$|week_start$", ak)
                    assert ok, f"{split}: {c['name']} bilinmeyen arg {ak}"
                    p = t.param(ak)
                    if p and p.enum:
                        assert str(av) in p.enum, f"{split}: {c['name']}.{ak} enum ihlali {av}"
                for rq in t.required:
                    p = t.param(rq)
                    if p and p.kind == "date_range":
                        continue
                    assert rq in c["arguments"], f"{split}: {c['name']} zorunlu {rq} eksik"


def test_encoding_hygiene():
    from conftest import DATA
    for f in DATA.glob("tool_calling_*.jsonl"):
        b = f.read_bytes()
        assert b[:3] != b"\xef\xbb\xbf" and b"\r\n" not in b and b.endswith(b"\n"), f.name
