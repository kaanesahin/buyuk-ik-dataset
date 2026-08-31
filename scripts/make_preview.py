#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Büyük İK dataset — insan-okunur önizleme üreticisi
==================================================

Kanonik eğitim dosyaları (`data/*.jsonl`) makine tarafından okunur: her satır
tek bir JSON nesnesidir ve bir IDE'de yatay olarak "uzayıp giden" tek satır
gibi görünür. Bu script, İÇERİĞİ DEĞİŞTİRMEDEN yalnızca gözle okunabilir
türevler üretir:

    preview/DATASET_PREVIEW.md          -> sohbet dökümleri (markdown)
    preview/samples/<decision>.sample.json  -> girintili tam kayıtlar (JSON)
    preview/index.md                    -> ne nerede + dağılım özeti

Kaynak: data/<prefix>_train.meta.jsonl  (+ _val.meta.jsonl)
        (meta dosyaları decision/intent/domain/difficulty/register etiketlerini
         VE tools/messages alanlarını taşır — eğitim girdisi bunları içermez.)

Seçim deterministiktir (sabit seed); tekrar çalıştırınca aynı önizleme çıkar.

Kullanım:
    python scripts/make_preview.py
    python scripts/make_preview.py --per-decision 24 --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Windows konsolu cp1254 olabilir; '✓' gibi karakterler stdout'ta çökmesin.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PREVIEW_DIR = ROOT / "preview"
DEFAULT_PREFIX = "buyuk_ik_tool_calling"

DECISION_ORDER = ["tool_call", "direct", "request_for_info", "cannot_answer"]
DECISION_TR = {
    "tool_call": "TOOL_CALL — tool çağır (tüm zorunlu parametreler mevcut)",
    "direct": "DIRECT — tool gerekmiyor, doğrudan yanıt",
    "request_for_info": "REQUEST_FOR_INFO — eksik bilgi / onay iste",
    "cannot_answer": "CANNOT_ANSWER — mevcut araçlarla cevaplanamaz",
}
ROLE_TR = {"user": "Kullanıcı", "assistant": "Asistan"}

TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


# ---------------------------------------------------------------------------

def load_meta(prefix: str) -> list[dict]:
    rows: list[dict] = []
    for split in ("train", "val"):
        p = DATA_DIR / f"{prefix}_{split}.meta.jsonl"
        if not p.exists():
            raise SystemExit(f"[X] bulunamadı: {p}\n    Önce scripts/generate_dataset.py çalıştırın.")
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                r["_split"] = split
                rows.append(r)
    return rows


def balanced_pick(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    """intent -> register -> difficulty ekseninde olabildiğince yayılmış k örnek."""
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_intent[r["intent"]].append(r)
    for lst in by_intent.values():
        rng.shuffle(lst)

    intents = sorted(by_intent)
    rng.shuffle(intents)

    picked: list[dict] = []
    seen_combo: set[tuple] = set()
    # 1. tur: her intent'ten yeni bir (register, difficulty) kombinasyonu
    while len(picked) < k:
        progressed = False
        for it in intents:
            if len(picked) >= k:
                break
            bucket = by_intent[it]
            for i, cand in enumerate(bucket):
                combo = (it, cand["register"], cand["difficulty"])
                if combo in seen_combo:
                    continue
                picked.append(bucket.pop(i))
                seen_combo.add(combo)
                progressed = True
                break
        if not progressed:
            break
    # 2. tur: kalan kotayı intent döngüsüyle doldur
    while len(picked) < k:
        progressed = False
        for it in intents:
            if len(picked) >= k:
                break
            if by_intent[it]:
                picked.append(by_intent[it].pop())
                progressed = True
        if not progressed:
            break

    picked.sort(key=lambda r: (r["intent"], r["difficulty"], r["register"]))
    return picked


# ---------------------------------------------------------------------------

def fmt_tools(rec_tools: list[dict], targets: list[str]) -> str:
    names = [t["name"] for t in rec_tools]
    tgt = set(targets or [])
    shown = [f"**{n}**" if n in tgt else n for n in names]
    return f"{len(names)} araç — " + ", ".join(shown)


def render_assistant(content: str) -> list[str]:
    """Asistan turunu markdown'a çevir — tool_call bloğu ise fence, değilse alıntı."""
    calls = TOOLCALL_RE.findall(content)
    if calls:
        out = ["```", content.strip(), "```"]
        return out
    # düz metin — satır sonlarını alıntı olarak koru
    return ["> " + line if line else ">" for line in content.strip().split("\n")]


def render_example(r: dict, n: int) -> str:
    meta = r
    tools = meta.get("tools", [])
    msgs = meta.get("messages", [])
    targets = meta.get("target_tools") or ([meta["target_tool"]] if meta.get("target_tool") else [])

    lines: list[str] = []
    lines.append(f"### {n}. `{meta['intent']}`  ·  {meta['decision']}")
    tags = [
        f"**domain** {meta['domain']}",
        f"**difficulty** {meta['difficulty']}",
        f"**register** {meta['register']}",
        f"**turns** {meta.get('turns', len(msgs))}",
        f"**split** {meta['_split']}",
    ]
    if meta.get("is_write"):
        tags.append("**write** ✔")
    if meta.get("confirmation_required"):
        tags.append("**onay gerekir** ✔")
    lines.append("  ·  ".join(tags))
    if meta.get("missing_parameters"):
        lines.append(f"**eksik parametre:** `{', '.join(meta['missing_parameters'])}`")
    lines.append("")
    lines.append(f"_Araçlar:_ {fmt_tools(tools, targets)}")
    lines.append("")

    for m in msgs:
        who = ROLE_TR.get(m["role"], m["role"])
        if m["role"] == "assistant":
            kind = "tool_call" if TOOLCALL_RE.search(m["content"]) else "text"
            lines.append(f"**🤖 {who}**  ·  _{kind}_")
            lines.extend(render_assistant(m["content"]))
        else:
            lines.append(f"**🧑 {who}**")
            lines.extend("> " + ln if ln else ">" for ln in m["content"].strip().split("\n"))
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def build_preview_md(picks: dict[str, list[dict]], stats: dict, per_decision: int) -> str:
    total = sum(stats["decision"].values())
    out: list[str] = []
    out.append("# Büyük İK dataset — okunur önizleme\n")
    out.append(
        "Bu dosya **otomatik üretilir** (`scripts/make_preview.py`) ve salt-okunurdur. "
        "Kanonik veri `data/*.jsonl` içindedir; buradaki örnekler birebir aynı içeriğin "
        "gözle okunur dökümüdür. Örnek seçimi deterministiktir.\n"
    )
    out.append(f"- Kaynak kayıt: **{total}** (train+val)")
    out.append(f"- Önizlenen: her karar sınıfından **{per_decision}** örnek "
               f"(intent / register / difficulty ekseninde yayılmış)\n")

    out.append("## Karar dağılımı (tam veri)\n")
    out.append("| decision | adet | oran |")
    out.append("|---|---:|---:|")
    for d in DECISION_ORDER:
        c = stats["decision"].get(d, 0)
        out.append(f"| `{d}` | {c} | %{100*c/max(total,1):.1f} |")
    out.append("")

    out.append("## İçindekiler\n")
    if picks.get("_showcase"):
        out.append("- [Çok-adımlı zincir ve çok turlu örnekler](#cok-adimli-ve-cok-turlu)")
    for d in DECISION_ORDER:
        out.append(f"- [{d}](#{d.replace('_','-')})")
    out.append("")

    showcase = picks.get("_showcase", [])
    if showcase:
        out.insert(len(out), "")
        out.append("\n<a id=\"cok-adimli-ve-cok-turlu\"></a>\n")
        out.append("## Çok-adımlı zincir ve çok turlu örnekler\n")
        out.append("_6 turlu `tool_call` zinciri (parametre topla → onay iste → uygula), "
                   "çok turlu `direct` (tanım + takip sorusu) ve çok turlu `cannot_answer` "
                   "(ret + kullanıcı ısrarı + kararlı ret)._\n")
        for i, r in enumerate(showcase, 1):
            out.append(render_example(r, i))

    for d in DECISION_ORDER:
        out.append(f"\n<a id=\"{d.replace('_','-')}\"></a>\n")
        out.append(f"## {DECISION_TR[d]}\n")
        rows = picks.get(d, [])
        covered = sorted({r["intent"] for r in rows})
        out.append(f"_Bu bölümde {len(rows)} örnek, {len(covered)} farklı intent._\n")
        for i, r in enumerate(rows, 1):
            out.append(render_example(r, i))
    return "\n".join(out).rstrip() + "\n"


def build_index_md(stats: dict) -> str:
    total = sum(stats["decision"].values())
    out = ["# preview/ — ne nerede\n"]
    out.append("| dosya | içerik |")
    out.append("|---|---|")
    out.append("| `DATASET_PREVIEW.md` | Karar sınıfına göre gruplanmış sohbet dökümleri — **buradan başlayın** |")
    out.append("| `samples/tool_call.sample.json` | `tool_call` — girintili tam kayıtlar (`tools`+`messages`) |")
    out.append("| `samples/direct.sample.json` | `direct` — girintili tam kayıtlar |")
    out.append("| `samples/request_for_info.sample.json` | `request_for_info` — girintili tam kayıtlar |")
    out.append("| `samples/cannot_answer.sample.json` | `cannot_answer` — girintili tam kayıtlar |")
    out.append("")
    out.append("Hepsi `scripts/make_preview.py` ile `data/` üzerinden üretilir; elle düzenlemeyin.\n")

    def blk(title, key):
        rows = [f"\n## {title}\n", "| değer | adet | oran |", "|---|---:|---:|"]
        for k, v in stats[key].most_common():
            rows.append(f"| `{k}` | {v} | %{100*v/max(total,1):.1f} |")
        return "\n".join(rows)

    out.append(blk("Domain", "domain"))
    out.append(blk("Difficulty", "difficulty"))
    out.append(blk("Register", "register"))
    out.append("\n")
    return "\n".join(out)


def sample_json_record(r: dict) -> dict:
    """Önizleme JSON'u: kanonik eğitim kaydı + kısa bir _meta başlığı (okuma kolaylığı)."""
    return {
        "_meta": {
            "id": r.get("id"),
            "split": r["_split"],
            "decision": r["decision"],
            "intent": r["intent"],
            "domain": r["domain"],
            "difficulty": r["difficulty"],
            "register": r["register"],
            "is_write": r.get("is_write", False),
            "confirmation_required": r.get("confirmation_required", False),
            "missing_parameters": r.get("missing_parameters", []),
        },
        "tools": r.get("tools", []),
        "messages": r.get("messages", []),
    }


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Büyük İK dataset okunur önizleme üreticisi")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--per-decision", type=int, default=22,
                    help="DATASET_PREVIEW.md'de karar sınıfı başına örnek (varsayılan 22)")
    ap.add_argument("--per-sample-json", type=int, default=16,
                    help="samples/<decision>.sample.json başına kayıt (varsayılan 16)")
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()

    rows = load_meta(args.prefix)
    rng = random.Random(args.seed)

    stats = {
        "decision": Counter(r["decision"] for r in rows),
        "domain": Counter(r["domain"] for r in rows),
        "difficulty": Counter(r["difficulty"] for r in rows),
        "register": Counter(r["register"] for r in rows),
    }

    by_decision: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_decision[r["decision"]].append(r)

    picks_md: dict[str, list[dict]] = {}
    picks_json: dict[str, list[dict]] = {}
    for d in DECISION_ORDER:
        pool = by_decision.get(d, [])
        picks_md[d] = balanced_pick(pool, min(args.per_decision, len(pool)),
                                    random.Random(args.seed + hash(d) % 1000))
        picks_json[d] = balanced_pick(pool, min(args.per_sample_json, len(pool)),
                                      random.Random(args.seed + 1 + hash(d) % 1000))

    # çok-adımlı / çok turlu vitrini: birkaç zincir + mt_direct + mt_cannot
    srng = random.Random(args.seed + 99)
    chains = [r for r in rows if r.get("chain")]
    mt_dir = [r for r in rows if r["decision"] == "direct" and r["multi_turn"]]
    mt_can = [r for r in rows if r["decision"] == "cannot_answer" and r["multi_turn"]]
    for lst in (chains, mt_dir, mt_can):
        srng.shuffle(lst)
    showcase = chains[:4] + mt_dir[:3] + mt_can[:3]
    picks_md["_showcase"] = showcase

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    (PREVIEW_DIR / "samples").mkdir(parents=True, exist_ok=True)

    (PREVIEW_DIR / "DATASET_PREVIEW.md").write_text(
        build_preview_md(picks_md, stats, args.per_decision), encoding="utf-8", newline="\n")
    (PREVIEW_DIR / "index.md").write_text(build_index_md(stats), encoding="utf-8", newline="\n")

    for d in DECISION_ORDER:
        recs = [sample_json_record(r) for r in picks_json[d]]
        (PREVIEW_DIR / "samples" / f"{d}.sample.json").write_text(
            json.dumps(recs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"[✓] önizleme yazıldı -> {PREVIEW_DIR}")
    print(f"    DATASET_PREVIEW.md          {sum(len(v) for v in picks_md.values())} örnek")
    print(f"    index.md")
    for d in DECISION_ORDER:
        print(f"    samples/{d}.sample.json{'':<{max(0,18-len(d))}} {len(picks_json[d])} kayıt")


if __name__ == "__main__":
    main()
