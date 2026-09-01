#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Büyük İK v2 — şema-güdümlü tool-calling policy dataset üreticisi.

~105 tool / 13 domain kataloğundan (catalog/) örnek üretir. Per-tool cümle
şablonu YOKTUR; her şey tool şemasından + tool-agnostik frame'lerden türetilir.

Çıktı (data/):
    tool_calling_train.jsonl        {"tools":[...], "messages":[...]}
    tool_calling_val.jsonl          (val_seen + val_unseen_tool)
    tool_calling_train.meta.jsonl   aynı sırada; QC/eval alanları
    tool_calling_val.meta.jsonl
    tools_all.json / tools_train.json / tools_val.json / tools_test.json

hard_eval seti ayrı script ile: scripts/build_hard_eval.py

Belirlenimci: aynı --seed -> byte-aynı çıktı.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from catalog import TOOLS, by_name  # noqa: E402
from gen import scenarios as SC  # noqa: E402
from gen.catalog_index import Index, DIRECT_POOL, CANNOT_POOL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_SEED = 20260901
DEFAULT_TODAY = "2026-09-01"
DEFAULT_N = 15000


# --------------------------------------------------------------------------- #
class Idx:
    """scenarios.py'nin beklediği hafif kapsayıcı (today + Index)."""
    def __init__(self, today, tools=TOOLS):
        self.today = today
        self.index = Index(tools)


def norm_sig(msgs):
    us = " || ".join(m["content"] for m in msgs if m["role"] == "user")
    t = us.replace("İ", "i").replace("I", "ı").lower()
    for a, b in (("ı", "i"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("ç", "c")):
        t = t.replace(a, b)
    t = re.sub(r"[a-z]{2,}[-_]?\d+", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^a-z ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class Builder:
    def __init__(self, seed, today):
        self.rng = random.Random(seed)
        self.today = today
        self.idx = Idx(today)
        self.seen = set()
        self.rows = []
        self.skipped = 0

    def _train_tools(self, cat=None):
        ts = [t for t in TOOLS if t.split == "train"]
        if cat == "write":
            ts = [t for t in ts if t.cat in ("write", "action")]
        elif cat:
            ts = [t for t in ts if t.cat == cat]
        return ts

    def add(self, rec: SC.Record, split="train", eval_kind=None):
        if rec is None:
            return False
        sig = (norm_sig(rec.messages), tuple(sorted(rec.meta.get("target_tools", []))),
               rec.meta.get("decision"))
        if sig in self.seen:
            self.skipped += 1
            return False
        self.seen.add(sig)
        rec.meta["split"] = split
        if eval_kind:
            rec.meta["eval_kind"] = eval_kind
        self.rows.append(rec)
        return True

    # --- senaryo koşucuları -------------------------------------------- #
    def run_read(self, n, split="train", tools=None):
        pool = tools or [t for t in self._train_tools() if t.cat == "read"]
        i = 0
        tries = 0
        while i < n and tries < n * 4:
            tries += 1
            t = pool[tries % len(pool)]
            r = SC.gen_read_call(self.rng, self.idx, t)
            if self.add(r, split):
                i += 1

    def run_missing(self, n, split="train", tools=None):
        pool = tools or [t for t in self._train_tools() if t.required]
        i, tries = 0, 0
        while i < n and tries < n * 5:
            tries += 1
            t = pool[tries % len(pool)]
            if self.add(SC.gen_missing_param(self.rng, self.idx, t), split):
                i += 1

    def run_write_confirm(self, n, split="train", tools=None):
        pool = tools or self._train_tools("write")
        i, tries = 0, 0
        while i < n and tries < n * 5:
            tries += 1
            t = pool[tries % len(pool)]
            if self.add(SC.gen_write_confirm(self.rng, self.idx, t), split):
                i += 1

    def run_write_execute(self, n, split="train", tools=None):
        pool = tools or self._train_tools("write")
        i, tries = 0, 0
        while i < n and tries < n * 5:
            tries += 1
            t = pool[tries % len(pool)]
            if self.add(SC.gen_write_execute(self.rng, self.idx, t), split):
                i += 1

    def run_write_chain(self, n, split="train", tools=None):
        pool = [t for t in (tools or self._train_tools("write"))
                if len([p for p in t.params if p.required]) >= 2]
        i, tries = 0, 0
        while i < n and tries < n * 6:
            tries += 1
            t = pool[tries % len(pool)]
            if self.add(SC.gen_write_chain(self.rng, self.idx, t), split):
                i += 1

    def run_multi_parallel(self, n, split="train", tools=None):
        reads = tools or [t for t in self._train_tools() if t.cat == "read"]
        bydom = defaultdict(list)
        for t in reads:
            bydom[t.domain].append(t)
        pairs = []
        for dom, ts in bydom.items():
            for a in ts:
                for b in ts:
                    if a.name < b.name and a.param_kinds() & b.param_kinds():
                        pa = next((p for p in a.params if p.required and p.kind in SC._PRIMARY_KINDS), None)
                        pb = next((p for p in b.params if p.required and p.kind in SC._PRIMARY_KINDS), None)
                        if (pa and pb and pa.kind == pb.kind and pa.name == pb.name
                                and (pa.kind != "id" or pa.prefix == pb.prefix)):
                            pairs.append((a, b))
        self.rng.shuffle(pairs)
        i, tries = 0, 0
        while i < n and pairs and tries < n * 5:
            tries += 1
            a, b = pairs[tries % len(pairs)]
            if self.add(SC.gen_multi_parallel(self.rng, self.idx, a, b), split):
                i += 1

    def run_multi_sequential(self, n, split="train"):
        i, tries = 0, 0
        chains = list(SC.CHAINS)
        while i < n and tries < n * 6:
            tries += 1
            c = chains[tries % len(chains)]
            if self.add(SC.gen_multi_sequential(self.rng, self.idx, c), split):
                i += 1

    def run_direct(self, n, split="train"):
        i, tries = 0, 0
        while i < n and tries < n * 6:
            tries += 1
            e = DIRECT_POOL[tries % len(DIRECT_POOL)]
            if self.add(SC.gen_direct(self.rng, self.idx, e), split):
                i += 1

    def run_cannot(self, n, split="train"):
        i, tries = 0, 0
        while i < n and tries < n * 6:
            tries += 1
            e = CANNOT_POOL[tries % len(CANNOT_POOL)]
            if self.add(SC.gen_cannot_scope(self.rng, self.idx, e), split):
                i += 1

    def run_hn(self, kind, n, split="train", tools=None):
        fn = {
            "kw": SC.gen_hn_keyword_ambiguous,
            "conflict": SC.gen_hn_conflict,
            "absent": SC.gen_hn_tool_absent,
            "wrongname": SC.gen_hn_user_names_wrong_tool,
        }[kind]
        pool = tools or self._train_tools()
        if kind in ("kw", "wrongname"):
            # tool_call üreten HN'ler yalnız READ tool'ları hedefler (onaysız WRITE olmasın)
            pool = [t for t in pool if t.cat == "read"]
        if kind == "kw":
            pool = [t for t in pool if self.idx.index.keyword_siblings(t.name)]
        i, tries = 0, 0
        while i < n and pool and tries < n * 8:
            tries += 1
            t = pool[tries % len(pool)]
            if self.add(fn(self.rng, self.idx, t), split):
                i += 1


# --------------------------------------------------------------------------- #
MIX = {
    "read": 0.345, "missing": 0.115, "write_confirm": 0.065, "write_execute": 0.065,
    "write_chain": 0.035, "multi_parallel": 0.04, "multi_sequential": 0.03,
    "direct": 0.11, "cannot": 0.10, "hn_kw": 0.03, "hn_conflict": 0.02,
    "hn_absent": 0.025, "hn_wrongname": 0.02,
}


def build_train(b: Builder, n):
    q = {k: max(6, round(n * v)) for k, v in MIX.items()}
    b.run_read(q["read"])
    b.run_missing(q["missing"])
    b.run_write_confirm(q["write_confirm"])
    b.run_write_execute(q["write_execute"])
    b.run_write_chain(q["write_chain"])
    b.run_multi_parallel(q["multi_parallel"])
    b.run_multi_sequential(q["multi_sequential"])
    b.run_direct(q["direct"])
    b.run_cannot(q["cannot"])
    b.run_hn("kw", q["hn_kw"])
    b.run_hn("conflict", q["hn_conflict"])
    b.run_hn("absent", q["hn_absent"])
    b.run_hn("wrongname", q["hn_wrongname"])


def build_val(b: Builder, n_seen, n_unseen):
    # val_seen: train tool'ları, alışılmadık yüzey (yüksek oblique zaten frame'de)
    train_reads = [t for t in TOOLS if t.split == "train" and t.cat == "read"]
    train_writes = [t for t in TOOLS if t.split == "train" and t.cat == "write"]
    b.run_read(round(n_seen * 0.5), "val", train_reads)
    b.run_missing(round(n_seen * 0.2), "val", [t for t in TOOLS if t.split == "train" and t.required])
    b.run_write_confirm(round(n_seen * 0.12), "val", train_writes)
    b.run_write_execute(round(n_seen * 0.1), "val", train_writes)
    b.run_direct(round(n_seen * 0.04), "val")
    b.run_cannot(round(n_seen * 0.04), "val")
    for r in b.rows:
        if r.meta["split"] == "val" and "eval_kind" not in r.meta:
            r.meta["eval_kind"] = "val_seen_tool"

    # val_unseen_tool: val_tools HEDEF olarak (eğitimde hiç görülmedi)
    vr = [t for t in TOOLS if t.split == "val" and t.cat == "read"]
    vw = [t for t in TOOLS if t.split == "val" and t.cat == "write"]
    before = len(b.rows)
    b.run_read(round(n_unseen * 0.55), "val", vr)
    b.run_missing(round(n_unseen * 0.2), "val", [t for t in TOOLS if t.split == "val" and t.required])
    b.run_write_confirm(round(n_unseen * 0.13), "val", vw)
    b.run_write_execute(round(n_unseen * 0.12), "val", vw)
    for r in b.rows[before:]:
        r.meta["eval_kind"] = "val_unseen_tool"


# --------------------------------------------------------------------------- #
def write_jsonl(path, objs):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--today", default=DEFAULT_TODAY)
    ap.add_argument("--val-seen", type=int, default=1000)
    ap.add_argument("--val-unseen", type=int, default=1000)
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    today = date.fromisoformat(args.today)
    b = Builder(args.seed, today)
    build_train(b, args.n)
    n_train = len(b.rows)
    build_val(b, args.val_seen, args.val_unseen)

    train = [r for r in b.rows if r.meta["split"] == "train"]
    val = [r for r in b.rows if r.meta["split"] == "val"]
    b.rng.shuffle(train)
    b.rng.shuffle(val)
    for i, r in enumerate(train, 1):
        r.meta["id"] = f"tc_{i:06d}"
    for i, r in enumerate(val, 1):
        r.meta["id"] = f"tc_val_{i:05d}"

    # --- rapor ---
    def dist(rows, key):
        return Counter(r.meta.get(key) for r in rows)

    print(f"seed={args.seed}  today={args.today}")
    print(f"train={len(train)}  val={len(val)}  (atlanan yakın-kopya: {b.skipped})")
    print("train decision:", dict(dist(train, "decision")))
    print("train scenario:", dict(dist(train, "scenario").most_common()))
    print("val eval_kind:", dict(dist(val, "eval_kind")))
    tr_tgt = Counter()
    for r in train:
        for t in r.meta.get("target_tools", []):
            tr_tgt[t] += 1
    print(f"distinct train target tools: {len(tr_tgt)} / {len([t for t in TOOLS if t.split=='train'])}")
    print(f"  min/median/max per tool: {min(tr_tgt.values())}/"
          f"{sorted(tr_tgt.values())[len(tr_tgt)//2]}/{max(tr_tgt.values())}")
    cc = [r.meta["candidate_count"] for r in train]
    cc.sort()
    print(f"candidate_count: p10={cc[len(cc)//10]} p50={cc[len(cc)//2]} p90={cc[len(cc)*9//10]} max={cc[-1]}")
    print(f"tool-result turn: {sum(1 for r in train if r.meta.get('has_tool_result'))}/{len(train)}"
          f"  ({100*sum(1 for r in train if r.meta.get('has_tool_result'))/len(train):.0f}%)")
    print(f"hard-negative: {sum(1 for r in train if r.meta.get('hard_negative'))}/{len(train)}")

    if args.dry_run:
        print("[dry-run] yazılmadı")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "tool_calling_train.jsonl", [{"tools": r.tools, "messages": r.messages} for r in train])
    write_jsonl(out / "tool_calling_val.jsonl", [{"tools": r.tools, "messages": r.messages} for r in val])
    write_jsonl(out / "tool_calling_train.meta.jsonl",
                [{**r.meta, "messages": r.messages} for r in train])
    write_jsonl(out / "tool_calling_val.meta.jsonl",
                [{**r.meta, "messages": r.messages} for r in val])

    def dump_tools(names, path):
        (out / path).write_text(
            json.dumps([by_name(n).schema() for n in names], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n")
    dump_tools([t.name for t in TOOLS], "tools_all.json")
    dump_tools([t.name for t in TOOLS if t.split == "train"], "tools_train.json")
    dump_tools([t.name for t in TOOLS if t.split == "val"], "tools_val.json")
    dump_tools([t.name for t in TOOLS if t.split == "test"], "tools_test.json")
    # split haritası
    (out / "tool_splits.json").write_text(
        json.dumps({t.name: {"domain": t.domain, "cat": t.cat, "split": t.split}
                    for t in TOOLS}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")

    # inceleme örneği (git'te tutulur; tam set deterministik olarak yeniden üretilir)
    sd = out / "sample"
    sd.mkdir(exist_ok=True)
    write_jsonl(sd / "train.sample.jsonl",
               [{"tools": r.tools, "messages": r.messages} for r in train[:400]])
    write_jsonl(sd / "train.sample.meta.jsonl", [dict(r.meta) for r in train[:400]])
    write_jsonl(sd / "val.sample.jsonl",
               [{"tools": r.tools, "messages": r.messages} for r in val[:150]])
    print(f"[✓] -> {out}   (tam jsonl ~{(out/'tool_calling_train.jsonl').stat().st_size//10**6} MB; "
          f"sample/ git'te)")


if __name__ == "__main__":
    main()
