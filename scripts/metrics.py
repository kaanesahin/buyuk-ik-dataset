#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Büyük İK v2 — dataset istatistikleri + policy/generalization ölçümleri.

Model gerektirmeyen (dataset-seviyesi) tüm ölçümleri hesaplar ve
`docs/DATASET_STATISTICS.md` üretir. Model-seviyesi metrikler (tool-selection
top-1 vb.) eğitim sonrası `hard_eval` üzerinde ölçülür — buradaki `probe`
dağılımı o değerlendirmenin iskeletidir.

Kullanım:  python scripts/metrics.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from catalog import TOOLS, by_name  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load(name):
    p = DATA / name
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def fold(s):
    import unicodedata
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    s = s.replace("İ", "i").replace("I", "ı").lower()
    for a, b in (("ı", "i"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("ç", "c")):
        s = s.replace(a, b)
    return s


def pct(n, d):
    return f"{100*n/d:.1f}%" if d else "—"


def tbl(counter, total, top=None):
    items = counter.most_common(top)
    return "\n".join(f"| `{k}` | {v} | {pct(v, total)} |" for k, v in items)


def main():
    tr = load("tool_calling_train.jsonl")
    trm = load("tool_calling_train.meta.jsonl")
    va = load("tool_calling_val.jsonl")
    vam = load("tool_calling_val.meta.jsonl")
    he = load("tool_calling_hard_eval.jsonl")
    hem = load("tool_calling_hard_eval.meta.jsonl")
    N = len(tr)

    L = []
    A = L.append
    A("# Büyük İK v2 — Dataset İstatistikleri\n")
    A(f"> Otomatik üretilir: `python scripts/metrics.py`. Kaynak: `data/`.\n")

    # --- B. temel sayılar ---
    A("## Temel sayılar\n")
    A("| ölçüt | değer |\n|---|---|")
    A(f"| train örnek | **{len(tr)}** |")
    A(f"| val örnek | **{len(va)}** (val_seen + val_unseen_tool) |")
    A(f"| hard_eval örnek | **{len(he)}** |")
    A(f"| toplam tool (katalog) | **{len(TOOLS)}** |")
    sp = Counter(t.split for t in TOOLS)
    A(f"| train_tools / val_tools / test_tools | **{sp['train']} / {sp['val']} / {sp['test']}** |")
    dom = Counter(t.domain for t in TOOLS)
    A(f"| domain sayısı | **{len(dom)}** ({', '.join(sorted(dom))}) |")
    cat = Counter(t.cat for t in TOOLS)
    A(f"| read / write / action | **{cat['read']} / {cat['write']} / {cat['action']}** |")

    # tool başına örnek dağılımı (train hedef)
    tgt = Counter()
    for m in trm:
        for t in m.get("target_tools", []):
            tgt[t] += 1
    vals = sorted(tgt.values())
    A(f"| train hedef-tool sayısı | {len(tgt)} / {sp['train']} |")
    A(f"| tool başına örnek (min/medyan/maks) | {vals[0]} / {vals[len(vals)//2]} / {vals[-1]} |")

    # --- karar dağılımı ---
    A("\n## Karar dağılımı (train)\n")
    dd = Counter(m["decision"] for m in trm)
    A("| decision | adet | oran |\n|---|---:|---:|")
    A(tbl(dd, N))

    # --- senaryo ---
    A("\n## Senaryo dağılımı (train)\n")
    sc = Counter(m["scenario"] for m in trm)
    A("| senaryo | adet | oran |\n|---|---:|---:|")
    A(tbl(sc, N))

    # --- tek-tool / multi-tool ---
    multi = sum(1 for m in trm if len(m.get("target_tools", [])) > 1)
    seq = sum(1 for m in trm if m.get("sequential"))
    A(f"\n## Tool çağrı yapısı\n")
    A("| ölçüt | adet | oran |\n|---|---:|---:|")
    A(f"| tek-tool tool_call | {dd['tool_call'] - multi} | {pct(dd['tool_call']-multi, N)} |")
    A(f"| çoklu-tool (paralel + sıralı) | {multi} | {pct(multi, N)} |")
    A(f"|  — bunun sıralısı (sonuç→param) | {seq} | {pct(seq, N)} |")
    trr = sum(1 for m in trm if m.get("has_tool_result"))
    A(f"| **tool-sonucu turu içeren** | {trr} | {pct(trr, N)} |")
    A(f"|  — tool_call örnekleri içinde | {trr} | {pct(trr, dd['tool_call'])} |")
    rm = Counter(m.get("tool_result_mode") for m in trm if m.get("has_tool_result"))
    A(f"| tool-sonucu modu | {dict(rm)} |  |")
    wr = sum(1 for m in trm if m.get("is_write"))
    A(f"| WRITE/action örneği | {wr} | {pct(wr, N)} |")
    conf = sum(1 for m in trm if m.get("confirmation"))
    A(f"| onay akışı (confirm) | {conf} | {pct(conf, N)} |")
    chain = sum(1 for m in trm if m.get("chain"))
    A(f"| 6-turlu zincir (eksik→onay→uygula) | {chain} | {pct(chain, N)} |")
    turns = Counter(m["turns"] for m in trm)
    A(f"| tur dağılımı | {dict(sorted(turns.items()))} |  |")

    # --- opsiyonel parametre ---
    optused = sum(1 for m in trm if m.get("optional_params_used"))
    A(f"\n## Parametre davranışı\n")
    A("| ölçüt | adet | oran |\n|---|---:|---:|")
    A(f"| opsiyonel parametre KULLANILAN örnek | {optused} | {pct(optused, N)} |")
    mp = sum(1 for m in trm if m.get("missing_params"))
    A(f"| eksik-parametre (request_for_info) | {mp} | {pct(mp, N)} |")

    # --- aday liste ---
    A("\n## Aday tool listesi boyutu (train)\n")
    cc = [m["candidate_count"] for m in trm]
    b = [sum(1 for c in cc if c <= 12), sum(1 for c in cc if 13 <= c <= 34),
         sum(1 for c in cc if c >= 35)]
    A("| kova | adet | oran | hedef |\n|---|---:|---:|---:|")
    A(f"| ≤12 tool | {b[0]} | {pct(b[0], N)} | ~28% |")
    A(f"| 13–34 tool | {b[1]} | {pct(b[1], N)} | ~56% |")
    A(f"| 35–58 tool | {b[2]} | {pct(b[2], N)} | ~16% |")
    A(f"| medyan / p90 / maks | {sorted(cc)[len(cc)//2]} / {sorted(cc)[int(len(cc)*.9)]} / {max(cc)} |  |  |")
    # hedef tool konumu (ele veriyor mu?)
    posfrac = []
    for d, m in zip(tr, trm):
        tt = m.get("target_tools", [])
        if len(tt) == 1:
            names = [x["name"] for x in d["tools"]]
            if tt[0] in names:
                posfrac.append(names.index(tt[0]) / max(1, len(names) - 1))
    A(f"| hedef tool'un liste-içi konumu (ort., 0=baş 1=son) | {statistics.mean(posfrac):.2f} (uniform ~0.50) |")

    # --- hard-negative ---
    A("\n## Hard-negative örnekleri (train)\n")
    hn = Counter(m.get("hard_negative") for m in trm if m.get("hard_negative"))
    A(f"Toplam **{sum(hn.values())}** ({pct(sum(hn.values()), N)}).\n")
    A("| tür | adet |\n|---|---:|")
    A("\n".join(f"| {k} | {v} |" for k, v in hn.most_common()))

    # --- doğal dil çeşitliliği ---
    A("\n## Doğal dil çeşitliliği (train)\n")
    reg = Counter(m["register"] for m in trm)
    A("| register | adet | oran |\n|---|---:|---:|")
    A(tbl(reg, N))
    u1 = [next(x["content"] for x in d["messages"] if x["role"] == "user") for d in tr]
    uniq1 = len(set(fold(x) for x in u1))
    A(f"\n- benzersiz ilk-kullanıcı-turu (folded): **{uniq1} / {len(u1)}** ({pct(uniq1, len(u1))})")
    # anti-kısayol: keyword -> tool adı korelasyonu (disc_kw)
    kwh = defaultdict(lambda: [0, 0])
    for d, m in zip(tr, trm):
        tt = m.get("target_tools", [])
        if len(tt) != 1 or m["decision"] not in ("tool_call", "request_for_info"):
            continue
        t = by_name(tt[0])
        if not t.disc_kw:
            continue
        uu = fold(" ".join(x["content"] for x in d["messages"] if x["role"] == "user"))
        kwh[t.name][1] += 1
        if any(fold(k) in uu for k in t.disc_kw):
            kwh[t.name][0] += 1
    overall = sum(h for h, _ in kwh.values()) / max(1, sum(t for _, t in kwh.values()))
    A(f"- **ayırt edici yüzey kelimesi → tool korelasyonu: {100*overall:.0f}%** "
      f"(K-1; hedef < 55%; eski sürüm ~97%)")
    hi = sorted(((n, h / t) for n, (h, t) in kwh.items() if t >= 20), key=lambda x: -x[1])[:6]
    A(f"  - en yüksek: " + ", ".join(f"{n.split('_',1)[1]} {100*r:.0f}%" for n, r in hi))

    # --- sızıntı / tekrar ---
    A("\n## Sızıntı ve tekrar\n")
    tr_sig = set(fold(" || ".join(x["content"] for x in d["messages"] if x["role"] == "user")) for d in tr)
    va_sig = set(fold(" || ".join(x["content"] for x in d["messages"] if x["role"] == "user")) for d in va)
    A(f"- train↔val kullanıcı-turu imza kesişimi: **{len(tr_sig & va_sig)}**")
    # tool sızıntısı
    leak = [t for t in tgt if by_name(t).split != "train"]
    A(f"- train hedefinde val/test tool'u: **{len(leak)}**")
    vu_tools = set()
    for m in vam:
        if m.get("eval_kind") == "val_unseen_tool":
            vu_tools |= set(m.get("target_tools", []))
    A(f"- val_unseen_tool hedefleri (hepsi split=val): "
      f"**{sum(1 for t in vu_tools if by_name(t).split=='val')}/{len(vu_tools)}**")
    he_t = set()
    for m in hem:
        if m.get("probe") in ("P1_unseen_tool", "P9_tool_result"):
            he_t |= set(m.get("target_tools", []))
    A(f"- hard_eval P1/P9 hedefleri (hepsi split=test): "
      f"**{sum(1 for t in he_t if by_name(t).split=='test')}/{len(he_t)}**")

    # --- tool coverage ---
    A("\n## Tool kapsama\n")
    A(f"- 75 train tool'unun tamamı hedef: **{'EVET' if len(tgt)==sp['train'] else 'HAYIR'}**")
    A(f"- dağılım eğriliği (maks/min): **{vals[-1]/vals[0]:.1f}×**")
    dc = defaultdict(Counter)
    for m in trm:
        for t in m.get("target_tools", []):
            dc[by_name(t).domain][m["decision"]] += 1
    A("\n| domain | tool_call | request_for_info | (train tool) |\n|---|---:|---:|---:|")
    for dm in sorted(dc):
        A(f"| {dm} | {dc[dm]['tool_call']} | {dc[dm]['request_for_info']} | "
          f"{sum(1 for t in TOOLS if t.domain==dm and t.split=='train')} |")

    # --- hard_eval probe ---
    A("\n## hard_eval probe dağılımı\n")
    pr = Counter(m["probe"] for m in hem)
    A("| probe | ne ölçer | adet |\n|---|---|---:|")
    desc = {
        "P1_unseen_tool": "eğitimde hiç görülmemiş tool'a doğru çağrı",
        "P2_seen_intent_new_tool": "bilinen senaryo kalıbı + yeni tool",
        "P3_category_new_surface": "bilinen kategori + havuz-dışı doğal dil",
        "P4_same_kw_diff_tool": "aynı kelime → doğru tool ayrımı",
        "P5_same_tool_new_phrasing": "bilinen tool + yepyeni ifade",
        "P6_large_candidate_set": "36–58 aday arasından seçim",
        "P7_cannot_answer": "uygun tool yok → kibar ret",
        "P8_clarification": "çelişkili parametre → netleştirme",
        "P9_tool_result": "görülmemiş tool sonucunu yorumlama",
    }
    for k, v in sorted(pr.items()):
        A(f"| `{k}` | {desc.get(k,'')} | {v} |")

    A("\n## Eğitim sonrası ölçülecek metrikler (hard_eval üzerinde)\n")
    A("| metrik | nasıl |\n|---|---|")
    A("| 4-karar doğruluğu | `meta.decision` vs model kararı, 4×4 matris |")
    A("| tool-selection top-1 / top-3 | `meta.target_tools[0]` vs model; P1/P5 ayrı raporla |")
    A("| argüman tam-eşleşme | tool_call arguments == altın (P1/P6) |")
    A("| halüsinasyon oranı | model argümanı kullanıcı/tool metninden izlenemiyorsa |")
    A("| yetkisiz WRITE | onay turu olmadan write/action tool_call sayısı (0 olmalı) |")
    A("| clarification doğruluğu | P8: model netleştirme sordu mu |")
    A("| tool-result özet doğruluğu | P9: yanıt yalnız sonuca dayanıyor mu, sayı uydurma yok |")
    A("| **genelleme farkı** | top-1(P5, seen tool) − top-1(P1, unseen tool); küçük fark = policy taşınıyor |")

    out = "\n".join(L) + "\n"
    (ROOT / "docs" / "DATASET_STATISTICS.md").write_text(out, encoding="utf-8", newline="\n")
    print(out)
    print("[✓] docs/DATASET_STATISTICS.md")


if __name__ == "__main__":
    main()
