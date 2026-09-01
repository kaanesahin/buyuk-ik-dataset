# -*- coding: utf-8 -*-
"""Her tool_call argümanı kullanıcı turundan veya önceki tool sonucundan izlenebilir."""
import importlib

import pytest

V = importlib.import_module("validate_dataset")


def _check(recs, metas):
    bad = []
    for r, m in zip(recs, metas):
        msgs = r["messages"]
        for j, msg in enumerate(msgs):
            if msg["role"] != "assistant" or "<tool_call>" not in msg["content"]:
                continue
            import json
            for b in V.TC_RE.findall(msg["content"]):
                o = json.loads(b)
                t = V.CATALOG.get(o["name"])
                if not t:
                    continue
                ublob = "\n".join(x["content"] for x in msgs[:j] if x["role"] == "user")
                prior = "\n".join(x["content"] for x in msgs[:j] if x["role"] == "tool")
                for ak, av in o["arguments"].items():
                    if isinstance(av, bool):
                        continue
                    if not V.trace_value(ak, av, t, ublob, V.fold(ublob), prior, V.fold(prior)):
                        bad.append((m.get("id"), o["name"], ak, av))
    return bad


def test_train_no_hallucination(train, train_meta):
    bad = _check(train, train_meta)
    assert not bad, f"{len(bad)} halüsinasyon: {bad[:10]}"


def test_val_no_hallucination(val, val_meta):
    bad = _check(val, val_meta)
    assert not bad, f"{len(bad)} halüsinasyon: {bad[:10]}"


def test_hardeval_no_hallucination(hard_eval, hard_eval_meta):
    bad = _check(hard_eval, hard_eval_meta)
    assert not bad, f"{len(bad)} halüsinasyon: {bad[:10]}"


def test_final_answer_after_tool_result_grounded(train):
    """tool sonucu -> asistan nihai yanıtında kaynak-dışı sayı yok."""
    import re
    bad = []
    for r in train:
        msgs = r["messages"]
        for j, m in enumerate(msgs):
            if m["role"] != "tool" or j + 1 >= len(msgs):
                continue
            ans = msgs[j + 1]["content"]
            if msgs[j + 1]["role"] != "assistant" or "<tool_call>" in ans:
                continue
            src = " ".join(x["content"] for x in msgs[:j + 1] if x["role"] in ("tool", "user"))
            src_d = re.sub(r"[.\s]", "", src)
            for num in re.findall(r"(?<![\w.])\d{2,}(?![\w])", re.sub(r"(?<=\d)[.\s](?=\d)", "", ans)):
                if re.sub(r"[.\s]", "", num) not in src_d:
                    bad.append((num, ans[:60]))
    assert not bad, f"{len(bad)} kaynak-dışı sayı: {bad[:8]}"
