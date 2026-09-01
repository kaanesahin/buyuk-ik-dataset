#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Büyük İK v2 — şema-güdümlü dataset için bağımsız kalite kontrol.

Kontroller (POLICY_UYGUNLUK_RAPORU.md §15):
  * JSON / Qwen mesaj yapısı / rol akışı
  * tool_call JSON geçerliliği, tool aday listede mi, zorunlu/enum/bilinmeyen arg
  * HALÜSİNASYON: her arg değeri kullanıcı turundan VEYA önceki tool sonucundan izlenebilir mi
  * WRITE onaysız çağrı yok
  * tool sonucu doğru bağlanmış (tool mesajı yalnız asistan tool_call'undan sonra)
  * nihai yanıt (tool sonrası) uydurma sayı/varlık içermiyor
  * train ↔ val/test TOOL sızıntısı (val/test tool'u train'de hedef olamaz)
  * birebir / aşırı-benzer tekrar
  * tool dağılımı, aday-liste kova dağılımı
  * keyword -> tool adı korelasyonu (K-1) — tool başına ve genel
  * karar tutarlılığı

Kullanım:  python scripts/validate_dataset.py
Çıkış kodu 1 => HATA var.
"""
from __future__ import annotations

import json
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
from gen import resolve as RSV  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TODAY = date(2026, 9, 1)

TC_RE = re.compile(r"<tool_call>\n(\{.*?\})\n</tool_call>", re.S)
_MONTHS = ("ocak|şubat|subat|mart|nisan|mayıs|mayis|haziran|temmuz|ağustos|agustos|"
           "eylül|eylul|ekim|kasım|kasim|aralık|aralik")
CATALOG = {t.name: t for t in TOOLS}
WRITE_TOOLS = {t.name for t in TOOLS if t.cat in ("write", "action")}


import unicodedata as _ud


def fold(s):
    s = _ud.normalize("NFKD", str(s))
    s = "".join(c for c in s if not _ud.combining(c))  # birleşen aksan (İ->i̇) temizle
    s = s.replace("İ", "i").replace("I", "ı").lower()
    for a, b in (("ı", "i"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("ç", "c")):
        s = s.replace(a, b)
    return s


def loose(s):
    return re.sub(r"[^a-z0-9]+", " ", fold(s)).strip()


class Rep:
    def __init__(self):
        self.err = []
        self.warn = []
        self.info = []

    def E(self, m):
        self.err.append(m)

    def W(self, m):
        self.warn.append(m)


# --------------------------------------------------------------------------- #
def trace_value(key, val, tool, ublob, ublob_f, priorblob, priorblob_f):
    """Arg değeri kullanıcı metninden veya önceki tool sonucundan izlenebilir mi?"""
    sval = str(val)
    p = tool.param(key) if tool else None
    kind = p.kind if p else None
    _u = (kind is None)  # kind yoksa yüzey biçiminden tahmin et; varsa kind otoritedir

    # önceki tool sonucunda birebir geçiyorsa (sıralı zincir) -> OK
    if sval and sval in priorblob:
        return True

    if kind in ("emp_id", "id") or (_u and re.fullmatch(r"[A-Z]{2,5}-\d+", sval)):
        num = sval.split("-")[-1]
        if fold(sval) in ublob_f:
            return True
        if re.search(rf"(?<!\d){re.escape(num)}(?!\d)", ublob):
            return True
        return False

    if kind in ("date", "future_date", "past_date") or (_u and re.fullmatch(r"\d{4}-\d{2}-\d{2}", sval)):
        cand_re = (rf"\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}[./]\d{{1,2}}[./]\d{{4}}|"
                   rf"\d{{1,2}}\s+(?:{_MONTHS})(?:\s+\d{{4}})?|"
                   rf"(?:bug[üu]n|yar[ıi]n|[öo]b[üu]r g[üu]n|d[üu]n|\d+\s+g[üu]n\s+sonra|"
                   rf"\d+\s+g[üu]n\s+[öo]nce|\d+\s+hafta\s+sonra|"
                   rf"(?:[öo]n[üu]m[üu]zdeki|gelecek|bu|haftaya)\s+\w+|ay\s+sonu)")
        for blob in (ublob, ublob_f):
            for m in re.finditer(cand_re, blob, re.I):
                if RSV.resolve_date(m.group(0), TODAY) == sval:
                    return True
        # tarih ARALIĞI yüzeyleri (start/end tek ifadeden)
        range_re = (rf"[^;()]*?(?:ba[şs]lang[ıi][çc]l[ıi]\s+\d+\s+g[üu]nl[üu]k|"
                    rf"\d{{4}}-\d{{2}}-\d{{2}}\s*/\s*\d{{4}}-\d{{2}}-\d{{2}}|"
                    rf"\d{{1,2}}\s*[-–]\s*\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}|"
                    rf"\d{{1,2}}(?:\s+(?:{_MONTHS}))?\s+\d{{0,4}}\s*[-–]\s*\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}|"
                    rf"\d{{1,2}}[./]\d{{1,2}}[./]\d{{4}}\s*(?:ile|arası|[-–])\s*\d{{1,2}}[./]\d{{1,2}}[./]\d{{4}})")
        for blob in (ublob, ublob_f):
            for m in re.finditer(range_re, blob, re.I):
                st, en = RSV.resolve_range(m.group(0), TODAY)
                if sval in (st, en):
                    return True
        return False

    if kind == "period" or (_u and re.fullmatch(r"\d{4}-\d{2}", sval)):
        pr = (rf"\d{{4}}-\d{{2}}\b|\d{{1,2}}/\d{{4}}|(?:{_MONTHS})\s+\d{{4}}|"
              rf"\d{{4}}\s+(?:{_MONTHS})|bu ay|ge[çc]en ay|iki ay [öo]nce|[öo]nceki ay")
        for blob in (ublob, ublob_f):
            for m in re.finditer(pr, blob, re.I):
                if RSV.resolve_period(m.group(0), TODAY) == sval:
                    return True
        return False

    if kind == "year" or (_u and re.fullmatch(r"20\d{2}", sval)):
        yr = r"20\d{2}|bu y[ıi]l|ge[çc]en y[ıi]l|gelecek y[ıi]l|bu sene|ge[çc]en sene|seneye"
        for blob in (ublob, ublob_f):
            for m in re.finditer(yr, blob, re.I):
                if RSV.resolve_year(m.group(0), TODAY) == sval:
                    return True
        return False

    if kind in ("amount", "count", "weight", "hours", "pct", "minutes", "duration") or \
            (_u and re.fullmatch(r"-?\d+(\.\d+)?", sval)):
        digits = re.sub(r"[.\s]", "", sval)
        ub = re.sub(r"(?<=\d)[.\s](?=\d)", "", ublob)
        for tok in re.findall(r"\d[\d.\s]*\d|\d", ub):
            if re.sub(r"[.\s]", "", tok) == digits:
                return True
        # "76 bin", "1,5 milyon" gibi çarpanlı ifadeler
        for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*(bin|milyon|milyar)", ublob_f):
            base = float(m.group(1).replace(",", "."))
            mult = {"bin": 1e3, "milyon": 1e6, "milyar": 1e9}[m.group(2)]
            if abs(base * mult - float(digits)) < 1:
                return True
        # sözel süre ("yarım saat" = 30, "1,5 saat" = 90)
        if kind == "minutes":
            words = {"çeyrek saat": 15, "yarım saat": 30, "bir saat": 60, "1 saat": 60,
                     "1,5 saat": 90, "1.5 saat": 90, "iki saat": 120, "2 saat": 120,
                     "45 dakika": 45, "45 dk": 45}
            for w, mn in words.items():
                if fold(w) in ublob_f and str(mn) == digits:
                    return True
        return re.search(rf"(?<!\d){re.escape(digits)}(?!\d)", ublob) is not None

    if kind == "enum" and p:
        cands = [sval] + (p.smap.get(sval, []) if p.smap else [])
        return any(fold(c) in ublob_f for c in cands)

    # serbest metin / isim / yer / uygulama / sorgu
    return loose(sval) in loose(ublob) or loose(sval)[:12] in loose(priorblob)


def check_record(i, rec, meta, rep, split, seen_sigs):
    ctx = f"[{split}:{meta.get('id', i)}]"
    tools = rec.get("tools")
    msgs = rec.get("messages")
    if not isinstance(tools, list) or not isinstance(msgs, list) or not msgs:
        rep.E(f"{ctx} tools/messages yapısı bozuk")
        return
    tnames = {t["name"] for t in tools}

    # rol akışı
    roles = [m["role"] for m in msgs]
    if roles[0] != "user":
        rep.E(f"{ctx} ilk mesaj user değil")
    if roles[-1] != "assistant":
        rep.E(f"{ctx} son mesaj assistant değil")
    for j, m in enumerate(msgs):
        if m["role"] not in ("user", "assistant", "tool"):
            rep.E(f"{ctx} geçersiz rol {m['role']}")
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            rep.E(f"{ctx} boş içerik mesaj[{j}]")
        if m["role"] == "tool":
            k = j - 1
            while k >= 0 and msgs[k]["role"] == "tool":
                k -= 1
            if k < 0 or msgs[k]["role"] != "assistant" or "<tool_call>" not in msgs[k]["content"]:
                rep.E(f"{ctx} tool mesajı[{j}] öncesinde asistan tool_call'u yok")
        if m["role"] == "user" and j > 0 and msgs[j - 1]["role"] == "user":
            rep.E(f"{ctx} ardışık user mesajı[{j}]")

    decision = meta.get("decision")
    # tüm tool_call'lar + hangi turda
    calls = []  # (turn_idx, name, args)
    for j, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        raw = m["content"].count("<tool_call>")
        blocks = TC_RE.findall(m["content"])
        if raw != len(blocks):
            rep.E(f"{ctx} bozuk <tool_call> bloğu (mesaj[{j}])")
        for b in blocks:
            try:
                o = json.loads(b)
            except json.JSONDecodeError:
                rep.E(f"{ctx} tool_call JSON parse hatası")
                continue
            calls.append((j, o.get("name"), o.get("arguments", {})))
        # tool_call bloğundan önce düz metin olmamalı (yalnız blok)
        if blocks and m["content"].split("<tool_call>")[0].strip():
            rep.E(f"{ctx} tool_call öncesi düz metin var (mesaj[{j}])")

    if decision in ("direct", "cannot_answer") and calls:
        rep.E(f"{ctx} decision={decision} ama tool_call var")
    if decision == "tool_call" and not calls:
        rep.E(f"{ctx} decision=tool_call ama tool_call yok")
    if decision == "request_for_info" and calls:
        rep.E(f"{ctx} decision=request_for_info ama tool_call var")
    if decision == "request_for_info":
        last = fold(msgs[-1]["content"])
        if not any(k in last for k in ("?", "misin", "musun", "misiniz", "musunuz", "nedir",
                                       "hangi", "ihtiyac", "paylas", "netlestir", "onayl",
                                       "iletir", "belirt", "verirsen", "soyler", "devam edemem",
                                       "alamam", "gerekiyor")):
            rep.W(f"{ctx} request_for_info son turu soru/onay gibi değil")

    # her tool_call: şema + halüsinasyon
    for turn, name, args in calls:
        if name not in tnames:
            rep.E(f"{ctx} çağrılan '{name}' aday listede yok")
        tool = CATALOG.get(name)
        if not tool:
            rep.E(f"{ctx} '{name}' katalogda yok")
            continue
        # train'de val/test tool'u hedef olamaz
        if split == "train" and tool.split != "train":
            rep.E(f"{ctx} SIZINTI: train örneğinde {tool.split} tool'u hedef ('{name}')")
        props = {p.name for p in tool.params}
        req = set(tool.required)
        # date_range iki alana açıldığı için required'ı esnet
        rangepairs = {}
        for p in tool.params:
            if p.kind == "date_range":
                pass
        for ak in args:
            if ak not in props and not re.match(r"(new_)?(start|end)_date$|week_start$", ak):
                rep.E(f"{ctx} '{name}' bilinmeyen arg '{ak}'")
        for rq in req:
            rp = tool.param(rq)
            if rq not in args and not (rp and rp.kind == "date_range"):
                rep.E(f"{ctx} '{name}' zorunlu arg '{rq}' eksik")
        for ak, av in args.items():
            p = tool.param(ak)
            if p and p.enum and str(av) not in p.enum:
                rep.E(f"{ctx} '{name}.{ak}' enum ihlali: {av}")

        # halüsinasyon: bu tool_call'a kadarki user + tool içerikleri
        ublob = "\n".join(m["content"] for m in msgs[:turn] if m["role"] == "user")
        priorblob = "\n".join(m["content"] for m in msgs[:turn] if m["role"] == "tool")
        ub_f, pb_f = fold(ublob), fold(priorblob)
        for ak, av in args.items():
            if isinstance(av, bool):
                continue
            if not trace_value(ak, av, tool, ublob, ub_f, priorblob, pb_f):
                rep.E(f"{ctx} HALÜSİNASYON '{name}.{ak}'={av!r} — kullanıcı/tool metninde yok "
                      f"| user: {ublob[:120]!r}")

    # WRITE onayı: write tool_call'undan önce (aynı konuşmada) confirm + ack olmalı
    for turn, name, args in calls:
        if name in WRITE_TOOLS:
            pre_asst = [m["content"] for m in msgs[:turn] if m["role"] == "assistant"]
            pre_user = [m["content"] for m in msgs[:turn] if m["role"] == "user"]
            confirmed = any(re.search(r"onay|devam edeyim mi|uygun mu|üzereyim|gerçekleştireceğim",
                                      fold(a)) for a in pre_asst)
            acked = any(re.search(r"evet|onayl|devam|olur|uygun|tabii|tamam", fold(u)) for u in pre_user[1:])
            if not (confirmed and acked):
                rep.E(f"{ctx} WRITE '{name}' onaysız çağrılmış (confirm={confirmed} ack={acked})")

    # tool sonrası nihai yanıt: uydurma sayı yok
    for j, m in enumerate(msgs):
        if m["role"] == "tool" and j + 1 < len(msgs) and msgs[j + 1]["role"] == "assistant":
            ans = msgs[j + 1]["content"]
            if "<tool_call>" in ans:
                continue
            src = " ".join(x["content"] for x in msgs[:j + 1] if x["role"] in ("tool", "user"))
            src_d = re.sub(r"[.\s]", "", src)
            for num in re.findall(r"(?<![\w.])\d{2,}(?![\w])", re.sub(r"(?<=\d)[.\s](?=\d)", "", ans)):
                if re.sub(r"[.\s]", "", num) not in src_d:
                    rep.W(f"{ctx} tool-sonrası yanıtta kaynak dışı sayı '{num}'")

    # near-dup
    sig = (loose(" || ".join(m["content"] for m in msgs if m["role"] == "user")),
           tuple(sorted(c[1] for c in calls)), decision)
    if sig in seen_sigs:
        seen_sigs["_dups"] = seen_sigs.get("_dups", 0) + 1
    else:
        seen_sigs[sig] = 1


# --------------------------------------------------------------------------- #
def aggregate_checks(train, val, tmeta, vmeta, rep):
    # tool sızıntı (dosya düzeyi zaten kayıt kontrolünde; burada özet)
    tr_targets = Counter()
    for m in tmeta:
        for t in m.get("target_tools", []):
            tr_targets[t] += 1
    leak = [t for t in tr_targets if CATALOG[t].split != "train"]
    if leak:
        rep.E(f"train hedef tool sızıntısı: {leak}")
    rep.info.append(f"train distinct target tools: {len(tr_targets)} / "
                    f"{len([t for t in TOOLS if t.split=='train'])}")
    if tr_targets:
        vals = sorted(tr_targets.values())
        rep.info.append(f"  per-tool örnek: min {vals[0]} / medyan {vals[len(vals)//2]} / max {vals[-1]}")
        if vals[0] < 15:
            rep.W(f"bazı train tool'ları <15 örnek: "
                  f"{[t for t,c in tr_targets.items() if c < 15][:8]}")

    # decision dağılımı
    dd = Counter(m["decision"] for m in tmeta)
    rep.info.append(f"train decision: {dict(dd)}")

    # aday-liste kova dağılımı
    cc = [m["candidate_count"] for m in tmeta]
    b1 = sum(1 for c in cc if c <= 12) / len(cc)
    b2 = sum(1 for c in cc if 13 <= c <= 34) / len(cc)
    b3 = sum(1 for c in cc if c >= 35) / len(cc)
    rep.info.append(f"aday-liste kovaları: ≤12 %{100*b1:.0f} | 13-34 %{100*b2:.0f} | 35+ %{100*b3:.0f} "
                    f"(medyan {sorted(cc)[len(cc)//2]}, maks {max(cc)})")
    if b3 < 0.10 or max(cc) < 45:
        rep.W("büyük aday liste (35+) payı / tavanı düşük — kalabalık-katalog sinyali zayıf")

    # tool-result oranı
    trr = sum(1 for m in tmeta if m.get("has_tool_result"))
    tc = sum(1 for m in tmeta if m["decision"] == "tool_call")
    rep.info.append(f"tool-result turu: {trr}/{len(tmeta)} (%{100*trr/len(tmeta):.0f}) "
                    f"| tool_call içinde %{100*trr/max(tc,1):.0f}")

    # KEYWORD -> TOOL ADI korelasyonu (K-1)
    kw_hit = defaultdict(lambda: [0, 0])
    for d, m in zip(train, tmeta):
        tgts = m.get("target_tools", [])
        if len(tgts) != 1 or m["decision"] not in ("tool_call", "request_for_info"):
            continue
        tool = CATALOG[tgts[0]]
        if not tool.disc_kw:
            continue
        u = fold(" ".join(x["content"] for x in d["messages"] if x["role"] == "user"))
        kw_hit[tool.name][1] += 1
        if any(fold(k) in u for k in tool.disc_kw):
            kw_hit[tool.name][0] += 1
    rates = {n: h / t for n, (h, t) in kw_hit.items() if t >= 10}
    if rates:
        overall = sum(h for h, _ in kw_hit.values()) / sum(t for _, t in kw_hit.values())
        hi = sorted(rates.items(), key=lambda x: -x[1])[:8]
        rep.info.append(f"keyword->tool korelasyonu: genel %{100*overall:.0f} "
                        f"(hedef < %55) | en yüksek: " +
                        ", ".join(f"{n.split('_',1)[1]} %{100*r:.0f}" for n, r in hi))
        if overall > 0.55:
            rep.W(f"keyword->tool korelasyonu yüksek (%{100*overall:.0f}) — K-1 riski")

    # register
    rr = Counter(m["register"] for m in tmeta)
    rep.info.append(f"register: {dict(rr)}")

    # val eval_kind
    vk = Counter(m.get("eval_kind") for m in vmeta)
    rep.info.append(f"val eval_kind: {dict(vk)}")
    # val_unseen: hedefler val_tools olmalı
    for d, m in zip(val, vmeta):
        if m.get("eval_kind") == "val_unseen_tool":
            for t in m.get("target_tools", []):
                if CATALOG[t].split != "val":
                    rep.E(f"[val] val_unseen örneğinde hedef '{t}' split={CATALOG[t].split}")


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    rep = Rep()
    files = {k: DATA / f"tool_calling_{k}.jsonl" for k in ("train", "val")}
    for k, f in files.items():
        if not f.exists():
            rep.E(f"yok: {f}")
    if rep.err:
        print("\n".join(rep.err)); sys.exit(1)

    train = load(files["train"]); val = load(files["val"])
    tmeta = load(DATA / "tool_calling_train.meta.jsonl")
    vmeta = load(DATA / "tool_calling_val.meta.jsonl")
    if len(train) != len(tmeta) or len(val) != len(vmeta):
        rep.E("meta/data satır sayısı uyuşmuyor")

    # kodlama
    for k, f in files.items():
        b = f.read_bytes()
        if b[:3] == b"\xef\xbb\xbf" or b"\r\n" in b or not b.endswith(b"\n"):
            rep.E(f"{k}: BOM/CRLF/son-satır hijyeni")

    seen = {}
    for i, (d, m) in enumerate(zip(train, tmeta), 1):
        check_record(i, d, m, rep, "train", seen)
    for i, (d, m) in enumerate(zip(val, vmeta), 1):
        check_record(i, d, m, rep, "val", seen)
    hep = DATA / "tool_calling_hard_eval.jsonl"
    if hep.exists():
        he = load(hep)
        hem = load(DATA / "tool_calling_hard_eval.meta.jsonl")
        for i, (d, m) in enumerate(zip(he, hem), 1):
            check_record(i, d, m, rep, "hard_eval", seen)
        rep.info.append(f"hard_eval: {len(he)} örnek doğrulandı")
    dups = seen.get("_dups", 0)
    total = len(train) + len(val)
    rep.info.append(f"aşırı-benzer (collapsed) kayıt: {dups} / {total} (%{100*dups/total:.2f})")
    if dups / total > 0.02:
        rep.E(f"aşırı-benzer kayıt oranı yüksek: %{100*dups/total:.1f}")
    elif dups / total > 0.005:
        rep.W(f"aşırı-benzer kayıt oranı: %{100*dups/total:.2f}")

    aggregate_checks(train, val, tmeta, vmeta, rep)

    out = ["# Büyük İK v2 — doğrulama raporu\n",
           f"- train {len(train)} | val {len(val)}",
           f"- HATA {len(rep.err)} | UYARI {len(rep.warn)}\n", "## Bilgi\n```"]
    out += rep.info
    out.append("```")
    if rep.err:
        out.append("\n## Hatalar\n" + "\n".join(f"- {e}" for e in rep.err[:120]))
        if len(rep.err) > 120:
            out.append(f"- ... (+{len(rep.err) - 120})")
    if rep.warn:
        out.append("\n## Uyarılar\n" + "\n".join(f"- {w}" for w in rep.warn[:60]))
        if len(rep.warn) > 60:
            out.append(f"- ... (+{len(rep.warn) - 60})")
    txt = "\n".join(out) + "\n"
    (ROOT / "docs" / "validation_report.md").write_text(txt, encoding="utf-8", newline="\n")
    print(txt)
    if rep.err:
        print(f"[X] {len(rep.err)} HATA")
        sys.exit(1)
    print(f"[OK] ({len(rep.warn)} uyarı)")


if __name__ == "__main__":
    main()
