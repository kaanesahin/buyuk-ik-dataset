#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Büyük İK dataset — eğitim şablonu varyantları
============================================

Kanonik dosyalar (`data/*.jsonl`) her satırda  {"tools": [...], "messages": [...]}
biçimindedir. Bu, Qwen 2.5 sohbet şablonunun BEKLEDİĞİ biçimdir: tokenizer,
`apply_chat_template(messages, tools=tools, ...)` çağrısında araç tanımlarını
system istemine kendisi yerleştirir. **Çoğu Qwen SFT hattı için ek bir şey
gerekmez; bu script'i çalıştırmanıza gerek yoktur.**

Kullandığınız eğitim şablonu araçları AYRI bir `tools` alanından değil de
doğrudan bir `system` turundan bekliyorsa, bu script içeriği DEĞİŞTİRMEDEN
system-turlu bir kopya üretir:

    data/variants/<prefix>_<split>.chatml_system.jsonl
        -> {"messages": [{"role": "system", "content": "<önsöz>\\n<tools>\\n[...]\\n</tools>"},
                         ... özgün user/assistant turları ...]}
        (ayrı "tools" alanı yoktur; araçlar system turunun içindedir)

Tool çağrı biçimi (`<tool_call>{...}</tool_call>`) ve tüm user/assistant
metinleri birebir korunur.

Kullanım:
    python scripts/build_training_variants.py
    python scripts/build_training_variants.py --style hermes
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
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "variants"
DEFAULT_PREFIX = "buyuk_ik_tool_calling"

SYSTEM_TR = (
    "Sen Büyük İK sisteminin araç kullanabilen asistanısın. Aşağıdaki araçları "
    "kullanabilirsin. Bir aracı çağırman gerektiğinde yanıtını tam olarak şu "
    "biçimde ver:\n<tool_call>\n{\"name\": <araç-adı>, \"arguments\": <argümanlar>}\n</tool_call>\n"
    "Araç gerekmiyorsa doğrudan yanıtla. Zorunlu bir bilgi eksikse önce onu iste; "
    "mevcut araçlarla çözülemeyen istekleri kibarca reddet. Hiçbir bilgiyi uydurma."
)

SYSTEM_HERMES = (
    "You are a function calling AI model. You are provided with function signatures "
    "within <tools></tools> XML tags. Call one or more functions to assist with the "
    "user query. For each function call return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags."
)


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def to_system_variant(rec: dict, style: str) -> dict:
    tools = rec.get("tools", [])
    tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
    if style == "hermes":
        sys_content = f"{SYSTEM_HERMES}\n<tools>\n{tools_json}\n</tools>"
    else:
        sys_content = f"{SYSTEM_TR}\n\n<tools>\n{tools_json}\n</tools>"
    return {"messages": [{"role": "system", "content": sys_content}] + rec["messages"]}


def main():
    ap = argparse.ArgumentParser(description="Büyük İK eğitim şablonu varyant üreticisi")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--style", choices=["tr", "hermes"], default="tr",
                    help="system önsözü: 'tr' (Türkçe, varsayılan) veya 'hermes' (İngilizce, NousResearch biçimi)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in ("train", "val"):
        src = DATA_DIR / f"{args.prefix}_{split}.jsonl"
        if not src.exists():
            raise SystemExit(f"[X] bulunamadı: {src}  (önce scripts/generate_dataset.py)")
        recs = load_jsonl(src)
        out = OUT_DIR / f"{args.prefix}_{split}.chatml_system.jsonl"
        with out.open("w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps(to_system_variant(r, args.style), ensure_ascii=False) + "\n")
        total += len(recs)
        print(f"[✓] {out.relative_to(ROOT)}  ({len(recs)} satır)")
    print(f"\nToplam {total} örnek. Kanonik dosyalar değişmedi.")
    print("Not: Qwen 2.5 kullanıyorsanız muhtemelen bu varyanta ihtiyacınız yok — "
          "kanonik {tools, messages} biçimini apply_chat_template ile kullanın.")


if __name__ == "__main__":
    main()
