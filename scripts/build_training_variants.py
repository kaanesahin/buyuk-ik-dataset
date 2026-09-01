#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kanonik {tools, messages} -> opsiyonel eğitim varyantları.

1) system-turlu ChatML kopyası  (--style tr|hermes)
   Qwen 2.5 kullanıyorsanız GEREKMEZ: tokenizer araçları
   `apply_chat_template(messages, tools=tools, ...)` ile system istemine kendisi
   koyar. Eğitim şablonunuz araçları ayrı bir `tools` alanından değil system
   turundan bekliyorsa bu script içerik DEĞİŞTİRMEDEN system-turlu bir kopya
   üretir:  data/variants/tool_calling_{split}.chatml_system.jsonl

2) küçük-aday-liste (curriculum) kopyası  (--max-candidates N)
   Her kaydın `tools` listesini, ÇAĞRILAN tool'lar + rastgele N-e-kadar çeldirici
   olacak şekilde kırpar (hedef her zaman listede; konum yeniden karışır). Dizi
   uzunluğunu ~3-6× kısaltır -> kayıp yoğunluğu (gradyan-sinyali / işlenen token)
   belirgin artar. Yalnız ISINMA epoch'ları için; tam çeldirici baskısı tam
   dosyada. Ayrıntı: docs/TRAINING_EFFICIENCY.md
       data/variants/tool_calling_{split}.cand{N}.jsonl

Kullanım:
    python scripts/build_training_variants.py --style tr
    python scripts/build_training_variants.py --max-candidates 10
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "variants"
_TC_NAME = re.compile(r'"name":\s*"([^"]+)"')

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


def _called_tools(rec):
    names = set()
    for m in rec["messages"]:
        if m["role"] == "assistant" and "<tool_call>" in m["content"]:
            names |= set(_TC_NAME.findall(m["content"]))
    return names


def trim_candidates(rec, n, rng):
    """`tools` listesini çağrılanlar + rastgele çeldiriciler = en çok n olacak
    şekilde kırp. Hedef her zaman kalır; sıra yeniden karışır."""
    tools = rec["tools"]
    if len(tools) <= n:
        return rec
    called = _called_tools(rec)
    keep = [t for t in tools if t["name"] in called]
    rest = [t for t in tools if t["name"] not in called]
    rng.shuffle(rest)
    keep += rest[: max(0, n - len(keep))]
    rng.shuffle(keep)
    return {**rec, "tools": keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["tr", "hermes"], default="tr")
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="0=kapalı; >0 ise curriculum (küçük-aday-liste) kopyası üret")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in ("train", "val", "hard_eval"):
        src = DATA / f"tool_calling_{split}.jsonl"
        if not src.exists():
            continue
        recs = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
        if args.max_candidates > 0:
            rng = random.Random(20260901)
            dst = OUT / f"tool_calling_{split}.cand{args.max_candidates}.jsonl"
            with dst.open("w", encoding="utf-8", newline="\n") as f:
                for r in recs:
                    t = trim_candidates(r, args.max_candidates, rng)
                    f.write(json.dumps({"tools": t["tools"], "messages": t["messages"]},
                                       ensure_ascii=False) + "\n")
        else:
            dst = OUT / f"tool_calling_{split}.chatml_system.jsonl"
            with dst.open("w", encoding="utf-8", newline="\n") as f:
                for r in recs:
                    f.write(json.dumps(variant(r, args.style), ensure_ascii=False) + "\n")
        total += len(recs)
        print(f"[✓] {dst.relative_to(ROOT)}  ({len(recs)})")
    print(f"Toplam {total}. Kanonik dosyalar değişmedi.")


if __name__ == "__main__":
    main()
