#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Büyük İK v2 — hard_eval seti + policy-probe'lar (POLICY_UYGUNLUK_RAPORU §16).

`data/tool_calling_hard_eval.jsonl` üretir. Her örnek eğitimde GÖRÜLMEYEN bir
durumu ölçer; `meta.probe` alanı hangi testi temsil ettiğini söyler:

  P1_unseen_tool        : hedef tool eğitimde HİÇ yok (split=test)
  P2_seen_intent_new_tool: eğitimde görülen senaryo kalıbı + görülmemiş tool
  P3_category_new_surface: görülen tool KATEGORİSİ + havuz-dışı doğal dil
  P4_same_kw_diff_tool   : aynı yüzey kelimesi birden çok tool'a uyar, doğru olan seçilmeli
  P5_same_tool_new_phrasing: eğitimdeki bir tool, tamamen farklı ifade biçimiyle
  P6_large_candidate_set : 45-90 aday tool arasından seçim

Ayrıca cannot_answer / clarification / tool-result probe'ları.

Kullanım: python scripts/build_hard_eval.py
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from catalog import TOOLS  # noqa: E402
from gen import scenarios as SC  # noqa: E402
from gen.catalog_index import Index, CANNOT_POOL  # noqa: E402
from generate_dataset import Idx, norm_sig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 777
TODAY = date(2026, 9, 1)

# havuz-DIŞI doğal-dil SARMALLARI (P3/P5) — mevcut (grounded) metni sarar, param düşürmez
UNSEEN_WRAPS = [
    ("şuna bir el atar mısın, kafam karıştı — ", ""),
    ("acil değil ama bugün lazım oldu: ", " bu arada"),
    ("patron sordu, net bir şey söyleyebilir miyiz: ", ""),
    ("sistemde bulamadım, buradan deneyeyim; ", " teşekkürler"),
    ("", "  (ilgilenirsen sevinirim, dün de sormuştum)"),
    ("kısaca: ", " — çok detaya girme"),
]


def _pick(rng, split, cat=None):
    ts = [t for t in TOOLS if t.split == split and (cat is None or t.cat == cat)]
    return ts


def build():
    rng = random.Random(SEED)
    idx = Idx(TODAY)
    rows = []
    seen = set()

    def emit(rec, probe):
        if rec is None:
            return
        s = norm_sig(rec.messages)
        if s in seen:
            return
        seen.add(s)
        rec.meta["probe"] = probe
        rec.meta["split"] = "hard_eval"
        rows.append(rec)

    test_reads = _pick(rng, "test", "read")
    test_writes = [t for t in TOOLS if t.split == "test" and t.cat in ("write", "action")]
    val_reads = _pick(rng, "val", "read")
    train_reads = _pick(rng, "train", "read")

    # P1: görülmemiş tool (test) — standart read + write
    for _ in range(70):
        t = rng.choice(test_reads)
        emit(SC.gen_read_call(rng, idx, t, with_result_p=0.35), "P1_unseen_tool")
    for _ in range(25):
        t = rng.choice(test_writes or test_reads)
        emit(SC.gen_write_execute(rng, idx, t) if t.cat != "read"
             else SC.gen_read_call(rng, idx, t), "P1_unseen_tool")

    # P2: görülen senaryo kalıbı (missing-param, multi) + görülmemiş tool
    for _ in range(30):
        t = rng.choice([x for x in test_reads + test_writes if x.required])
        emit(SC.gen_missing_param(rng, idx, t), "P2_seen_intent_new_tool")

    # P3: görülen KATEGORİ + havuz-dışı yüzey (test tool; mevcut grounded metni sar)
    for _ in range(35):
        t = rng.choice(test_reads)
        rec = SC.gen_read_call(rng, idx, t, with_result_p=0.0)
        pre, post = rng.choice(UNSEEN_WRAPS)
        base = rec.messages[0]["content"]
        rec.messages[0]["content"] = (pre + (base[0].lower() + base[1:] if pre else base) + post).strip()
        emit(rec, "P3_category_new_surface")

    # P4: aynı keyword farklı tool (test tool hedef, kardeşleri listede)
    for _ in range(25):
        t = rng.choice(test_reads)
        emit(SC.gen_hn_keyword_ambiguous(rng, idx, t), "P4_same_kw_diff_tool")

    # P5: eğitimdeki tool + havuz-dışı ifade sarmalı
    for _ in range(35):
        t = rng.choice(train_reads)
        rec = SC.gen_read_call(rng, idx, t, with_result_p=0.0)
        pre, post = rng.choice(UNSEEN_WRAPS)
        base = rec.messages[0]["content"]
        rec.messages[0]["content"] = (pre + (base[0].lower() + base[1:] if pre else base) + post).strip()
        emit(rec, "P5_same_tool_new_phrasing")

    # P6: 45-90 aday arasından seçim (test + train tool hedefler)
    for _ in range(40):
        t = rng.choice(test_reads + train_reads)
        rec = SC.gen_read_call(rng, idx, t, with_result_p=0.0)
        big, names = idx.index.candidate_list(rng, [t.name], size=rng.randint(42, 62))
        rec.tools = big
        rec.meta["candidate_count"] = len(names)
        emit(rec, "P6_large_candidate_set")

    # cannot_answer probe (kapsam-dışı + doğru tool listede yok)
    for _ in range(25):
        e = rng.choice(CANNOT_POOL)
        emit(SC.gen_cannot_scope(rng, idx, e), "P7_cannot_answer")
    for _ in range(20):
        t = rng.choice(test_reads + train_reads)
        emit(SC.gen_hn_tool_absent(rng, idx, t), "P7_cannot_answer")

    # clarification probe (çelişkili parametre)
    for _ in range(15):
        t = rng.choice([x for x in test_reads + train_reads if x.required])
        emit(SC.gen_hn_conflict(rng, idx, t), "P8_clarification")

    # tool-result probe (görülmemiş tool, sonucu yorumlama)
    for _ in range(25):
        t = rng.choice(test_reads)
        emit(SC.gen_read_call(rng, idx, t, with_result_p=1.0), "P9_tool_result")

    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r.meta["id"] = f"he_{i:04d}"
    return rows


def main():
    rows = build()
    with (DATA / "tool_calling_hard_eval.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({"tools": r.tools, "messages": r.messages}, ensure_ascii=False) + "\n")
    with (DATA / "tool_calling_hard_eval.meta.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({**r.meta, "messages": r.messages}, ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter(r.meta["probe"] for r in rows)
    print(f"hard_eval: {len(rows)} örnek")
    for k, v in sorted(c.items()):
        print(f"  {k:26s} {v}")
    # sızıntı kontrolü: P1/P2/P3/P4/P9 hedefleri gerçekten test split mi
    from catalog import by_name
    bad = 0
    for r in rows:
        if r.meta["probe"] in ("P1_unseen_tool", "P2_seen_intent_new_tool",
                               "P3_category_new_surface", "P9_tool_result"):
            for t in r.meta.get("target_tools", []):
                if by_name(t).split != "test":
                    bad += 1
    print(f"  [sızıntı] görülmemiş-tool probe'unda test-dışı hedef: {bad}")


if __name__ == "__main__":
    main()
