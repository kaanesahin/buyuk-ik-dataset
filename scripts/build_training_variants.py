#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kanonik {tools, messages} -> system-turlu ChatML kopyası (opsiyonel).

Qwen 2.5 kullanıyorsanız GEREKMEZ: tokenizer araçları
`apply_chat_template(messages, tools=tools, ...)` ile system istemine kendisi koyar.
Eğitim şablonunuz araçları ayrı bir `tools` alanından değil system turundan
bekliyorsa bu script içerik DEĞİŞTİRMEDEN system-turlu bir kopya üretir:

    data/variants/tool_calling_{split}.chatml_system.jsonl

Kullanım:  python scripts/build_training_variants.py [--style tr|hermes]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "variants"

SYS_TR = (
    "Sen kurumsal sistemlerin araç kullanabilen asistanısın. Aşağıdaki araçları "
    "kullanabilirsin. Bir araç çağırman gerektiğinde yanıtını tam olarak şu biçimde ver:\n"
    "<tool_call>\n{\"name\": <araç-adı>, \"arguments\": <argümanlar>}\n</tool_call>\n"
    "Araç gerekmiyorsa doğrudan yanıtla. Zorunlu bir bilgi eksikse önce onu iste; "
    "değişiklik yaratan işlemleri kullanıcı onayı olmadan yapma; mevcut araçlarla "
    "çözülemeyen istekleri kibarca reddet. Hiçbir bilgiyi uydurma. Araç sonucu geldiğinde "
    "yalnızca o sonuca dayanarak yanıtla."
)
SYS_HERMES = (
    "You are a function calling AI model. You are provided with function signatures "
    "within <tools></tools> XML tags. Call one or more functions to assist with the user "
    "query. For each function call return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags. Do not make up values; ask if a required "
    "argument is missing; require explicit confirmation before state-changing actions."
)


def variant(rec, style):
    tj = json.dumps(rec["tools"], ensure_ascii=False, indent=2)
    sysmsg = (SYS_HERMES if style == "hermes" else SYS_TR) + f"\n<tools>\n{tj}\n</tools>"
    return {"messages": [{"role": "system", "content": sysmsg}] + rec["messages"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["tr", "hermes"], default="tr")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in ("train", "val", "hard_eval"):
        src = DATA / f"tool_calling_{split}.jsonl"
        if not src.exists():
            continue
        recs = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
        dst = OUT / f"tool_calling_{split}.chatml_system.jsonl"
        with dst.open("w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps(variant(r, args.style), ensure_ascii=False) + "\n")
        total += len(recs)
        print(f"[✓] {dst.relative_to(ROOT)}  ({len(recs)})")
    print(f"Toplam {total}. Kanonik dosyalar değişmedi.")


if __name__ == "__main__":
    main()
