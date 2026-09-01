#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Büyük İK tool-calling dataset — bağımsız kalite kontrol (When2Call §31)
=====================================================================

`generate_dataset.py` çıktısını yeniden yükleyip doğrular:

  * yapısal geçerlilik (her satır JSON, tools + messages)
  * mesaj biçimi (rol sırası, boş içerik, alternasyon)
  * tool şeması tutarlılığı
  * çağrılan tool tanımlı mı
  * argüman anahtarları şema ile uyumlu mu
  * zorunlu parametre eksik mi
  * enum ihlali var mı
  * karar (decision) davranış tutarlılığı
  * HALÜSİNASYON: uydurulmuş employee_id / tarih / tutar / talep_id
  * karar dağılımı hedefe yakın mı
  * intent / yüzey çeşitliliği

Kullanım:
    python scripts/validate_dataset.py --dir ./data --prefix buyuk_ik_tool_calling
    python scripts/validate_dataset.py     # varsayılan: depo kökündeki data/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Windows konsolu cp1254 olabilir; Türkçe/simge karakterler stdout'ta çökmesin.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# generate_dataset.py bu dosyayla aynı klasörde (scripts/); hangi dizinden
# çağrılırsa çağrılsın import edilebilsin.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_dataset as G

TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
EMP_RE = re.compile(r"EMP-\d+", re.IGNORECASE)
LV_RE = re.compile(r"LV-\d[\d-]*", re.IGNORECASE)
ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
ISO_PERIOD_RE = re.compile(r"\b\d{4}-\d{2}\b")

TOL = 0.035  # karar dağılımı toleransı (±3.5 puan)


# ---- ters yüzey haritaları (halüsinasyon izleme için) ---------------------

def _fold(s: str) -> str:
    return G.tr_fold(s)


DATE_SURFACES: dict[str, list[str]] = defaultdict(list)
for surf, b, e in G.DATE_RANGES:
    DATE_SURFACES[b].append(surf)
    DATE_SURFACES[e].append(surf)
for surf, b, e in G.MONTH_RANGES:
    DATE_SURFACES[b].append(surf)
    DATE_SURFACES[e].append(surf)

PERIOD_SURFACES: dict[str, list[str]] = defaultdict(list)
for surf, canon in G.DONEMLER:
    PERIOD_SURFACES[canon].append(surf)
for surf, canon in G.DONEM_YIL:
    PERIOD_SURFACES[canon].append(surf)

IZIN_CANON_SURF: dict[str, list[str]] = defaultdict(list)
for surf, canon in G.IZIN_TIPI_YUZEY.items():
    IZIN_CANON_SURF[canon].append(surf)

MEDENI_CANON_SURF: dict[str, list[str]] = defaultdict(list)
for surf, canon in G.MEDENI_YUZEY.items():
    MEDENI_CANON_SURF[canon].append(surf)

OGRENIM_CANON_SURF: dict[str, list[str]] = defaultdict(list)
for surf, canon in G.OGRENIM_YUZEY.items():
    OGRENIM_CANON_SURF[canon].append(surf)

DEPT_CANON_SURF: dict[str, list[str]] = defaultdict(list)
for surf, canon in G.DEPARTMAN_YUZEY.items():
    DEPT_CANON_SURF[canon].append(surf)
for d in G.DEPARTMANLAR:
    DEPT_CANON_SURF[d].append(d)

DURUM_SURF = {
    "aktif": ["aktif", "calisan"],
    "izinli": ["izin", "izinde", "izinli"],
    "ayrildi": ["ayril", "isten cik", "ayrilmis"],
}
TUR_SURF = {"net": ["net", "elime", "eline"], "brut": ["brut", "brüt"]}


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def err(self, m):
        self.errors.append(m)

    def warn(self, m):
        self.warnings.append(m)


def load_jsonl(path: Path):
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        rows.append((i, json.loads(line)))
    return rows


# ---------------------------------------------------------------------------

def check_tool_schema(tool: dict, rep: Report, ctx: str):
    for k in ("name", "description", "parameters"):
        if k not in tool:
            rep.err(f"{ctx}: tool '{tool.get('name','?')}' -> '{k}' alanı yok")
    params = tool.get("parameters", {})
    if params.get("type") != "object":
        rep.err(f"{ctx}: tool '{tool.get('name')}' parameters.type != object")
    props = params.get("properties", {})
    for r in params.get("required", []):
        if r not in props:
            rep.err(f"{ctx}: tool '{tool.get('name')}' required '{r}' properties içinde yok")


def parse_tool_calls(text: str):
    out = []
    for m in TOOLCALL_RE.finditer(text):
        out.append(json.loads(m.group(1)))
    return out


def check_messages(msgs, rep: Report, ctx: str):
    if not msgs:
        rep.err(f"{ctx}: messages boş")
        return
    if msgs[0]["role"] != "user":
        rep.err(f"{ctx}: ilk mesaj user değil")
    if msgs[-1]["role"] != "assistant":
        rep.err(f"{ctx}: son mesaj assistant değil")
    for j, m in enumerate(msgs):
        if m.get("role") not in ("user", "assistant"):
            rep.err(f"{ctx}: geçersiz rol '{m.get('role')}'")
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            rep.err(f"{ctx}: mesaj[{j}] içeriği boş")
        if j > 0 and m["role"] == msgs[j - 1]["role"]:
            rep.err(f"{ctx}: mesaj[{j}] rol alternasyonu bozuk")


def _loose(s: str) -> str:
    """fold + noktalama/boşluk normalizasyonu — gevşek altdizi eşleşmesi için."""
    return re.sub(r"[^a-z0-9]+", " ", _fold(s)).strip()


def value_in_blob(val: str, blob_folded: str) -> bool:
    return _loose(str(val)) in _loose(blob_folded)


def trace_arg(key: str, val, prop_schema: dict, user_blob: str, user_blob_folded: str, rep: Report, ctx: str):
    """Argüman değeri kullanıcı turlarından izlenebiliyor mu (halüsinasyon kontrolü)."""
    sval = str(val)

    if EMP_RE.fullmatch(sval):
        num = sval.split("-")[1]
        if sval.lower() in user_blob.lower():
            return
        if re.search(rf"(?<!\d){re.escape(num)}(?!\d)", user_blob):
            return
        rep.err(f"{ctx}: employee_id '{sval}' kullanıcı mesajında geçmiyor (HALÜSİNASYON)")
        return

    if LV_RE.fullmatch(sval):
        if sval.lower() not in user_blob.lower():
            rep.err(f"{ctx}: talep_id '{sval}' kullanıcı mesajında yok (HALÜSİNASYON)")
        return

    if key in ("donem",) or ISO_DATE_RE.fullmatch(sval) or ISO_PERIOD_RE.fullmatch(sval):
        if _fold(sval) in user_blob_folded:
            return
        surfaces = DATE_SURFACES.get(sval, []) + PERIOD_SURFACES.get(sval, [])
        if any(_fold(s) in user_blob_folded for s in surfaces):
            return
        if key == "donem":
            # "bu yıl", "geçen ay" gibi göreli ifadeler
            if re.search(r"bu (yil|ay)|gecen (yil|ay|sene)|onceki (ay|yil)", user_blob_folded):
                return
        rep.err(f"{ctx}: tarih/dönem '{sval}' kullanıcı metninden türetilemiyor (HALÜSİNASYON)")
        return

    enum = prop_schema.get("enum")
    if enum:
        if key == "izin_tipi":
            ok = any(_fold(s) in user_blob_folded for s in IZIN_CANON_SURF.get(sval, [sval]))
        elif key == "kaynak_tipi":
            ok = _fold(G.KAYNAK_SURF.get(sval, sval)) in user_blob_folded or _fold(sval) in user_blob_folded
        elif key == "durum":
            ok = any(s in user_blob_folded for s in DURUM_SURF.get(sval, [sval]))
        elif key == "tur":
            ok = any(s in user_blob_folded for s in TUR_SURF.get(sval, [sval]))
        elif key == "medeni_durum":
            ok = any(_fold(s) in user_blob_folded for s in MEDENI_CANON_SURF.get(sval, [sval]))
        elif key == "ogrenim_durumu":
            ok = any(_fold(s) in user_blob_folded for s in OGRENIM_CANON_SURF.get(sval, [sval]))
        else:
            ok = _fold(sval) in user_blob_folded
        if not ok:
            rep.warn(f"{ctx}: enum arg {key}='{sval}' kullanıcı metninde açıkça görünmüyor")
        return

    if prop_schema.get("type") == "number":
        if re.search(rf"(?<!\d){re.escape(sval)}(?!\d)", user_blob.replace(".", "")):
            return
        rep.err(f"{ctx}: sayısal arg {key}={sval} kullanıcı metninde yok (HALÜSİNASYON)")
        return

    if key == "departman_adi":
        if any(_fold(s) in user_blob_folded for s in DEPT_CANON_SURF.get(sval, [sval])):
            return
        rep.warn(f"{ctx}: departman '{sval}' kullanıcı metninde birebir görünmüyor")
        return

    # serbest metin (gerekce / adres / email / telefon / pozisyon / aciklama)
    if not value_in_blob(sval, user_blob_folded):
        rep.warn(f"{ctx}: arg {key}='{sval}' kullanıcı metninde birebir yok")


def check_record(idx, rec, meta, rep: Report):
    ctx = f"satır {idx}" + (f" [{meta.get('id')}]" if meta else "")
    tools = rec.get("tools")
    msgs = rec.get("messages")
    if not isinstance(tools, list) or not isinstance(msgs, list):
        rep.err(f"{ctx}: tools/messages liste değil")
        return

    tool_names = set()
    for t in tools:
        check_tool_schema(t, rep, ctx)
        tool_names.add(t.get("name"))
    tool_by_name = {t.get("name"): t for t in tools}

    check_messages(msgs, rep, ctx)

    user_turns = [m["content"] for m in msgs if m["role"] == "user"]
    asst_turns = [m["content"] for m in msgs if m["role"] == "assistant"]
    user_blob = "\n".join(user_turns)
    user_blob_folded = _fold(user_blob)

    all_calls = []
    for k, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        calls = TOOLCALL_RE.findall(m["content"])
        is_last = (k == len(msgs) - 1)
        if calls and not is_last:
            rep.err(f"{ctx}: tool_call son olmayan assistant mesajında (mesaj[{k}])")
        for raw in calls:
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                rep.err(f"{ctx}: tool_call JSON parse edilemedi")
                continue
            all_calls.append(obj)

    # ---- decision tutarlılığı ----
    decision = meta.get("decision") if meta else None
    last_has_call = bool(TOOLCALL_RE.search(msgs[-1]["content"])) if msgs else False
    any_call = bool(all_calls)

    if decision == "direct":
        if any_call:
            rep.err(f"{ctx}: decision=direct ama tool_call var")
    elif decision == "tool_call":
        if not last_has_call:
            rep.err(f"{ctx}: decision=tool_call ama son mesajda tool_call yok")
    elif decision == "request_for_info":
        if any_call:
            rep.err(f"{ctx}: decision=request_for_info ama tool_call var")
        last = _fold(msgs[-1]["content"])
        req_lemmas = ("?", "misiniz", "musunuz", "paylasir", "paylasirsan", "iletir", "iletirsen",
                      "belirt", "verir mi", "ihtiyacim var", "gerekiyor", "hangi ", "onayl")
        if not any(x in last for x in req_lemmas):
            rep.warn(f"{ctx}: request_for_info son mesajı bilgi/onay isteği gibi görünmüyor")
    elif decision == "cannot_answer":
        if any_call:
            rep.err(f"{ctx}: decision=cannot_answer ama tool_call var")

    # ---- tool_call doğrulaması ----
    for obj in all_calls:
        name = obj.get("name")
        args = obj.get("arguments", {})
        if name not in tool_names:
            rep.err(f"{ctx}: çağrılan tool '{name}' tools listesinde yok")
            continue
        if not isinstance(args, dict):
            rep.err(f"{ctx}: '{name}' arguments dict değil")
            continue
        schema = tool_by_name[name]["parameters"]
        props = schema.get("properties", {})
        for ak in args:
            if ak not in props:
                rep.err(f"{ctx}: '{name}' bilinmeyen argüman '{ak}'")
        for req in schema.get("required", []):
            if req not in args:
                rep.err(f"{ctx}: '{name}' zorunlu argüman '{req}' eksik")
        for ak, av in args.items():
            ps = props.get(ak, {})
            if ps.get("enum") and str(av) not in ps["enum"]:
                rep.err(f"{ctx}: '{name}.{ak}' enum ihlali: '{av}' ∉ {ps['enum']}")
            if ps.get("type") == "string" and not isinstance(av, str):
                rep.err(f"{ctx}: '{name}.{ak}' string olmalı")
            if ps.get("type") == "number" and not isinstance(av, (int, float)):
                rep.err(f"{ctx}: '{name}.{ak}' number olmalı")
            trace_arg(ak, av, ps, user_blob, user_blob_folded, rep, f"{ctx} ({name})")

    # ---- assistant DÜZ METNİNDE (tool_call bloğu hariç) uydurulmuş kimlik ----
    for a in asst_turns:
        prose = TOOLCALL_RE.sub(" ", a)
        for mm in EMP_RE.findall(prose):
            num = mm.split("-")[1]
            if mm.lower() in user_blob.lower():
                continue
            if re.search(rf"(?<!\d){re.escape(num)}(?!\d)", user_blob):
                continue
            rep.err(f"{ctx}: assistant düz metninde uydurulmuş employee_id '{mm}'")
        for mm in LV_RE.findall(prose):
            if mm.lower() not in user_blob.lower():
                rep.err(f"{ctx}: assistant düz metninde uydurulmuş talep_id '{mm}'")
        if decision in ("request_for_info", "cannot_answer"):
            for num in re.findall(r"\b(\d[\d.]*)\s*(?:gün|gun|TL|₺|saat)\b", a):
                if _fold(num) not in user_blob_folded:
                    rep.err(f"{ctx}: {decision} yanıtında kullanıcı vermediği sayı '{num}' (HALÜSİNASYON)")

    # meta ↔ record adı uyumu
    if meta:
        tgt = meta.get("target_tool")
        if decision == "tool_call" and tgt and not meta.get("target_tools", []):
            pass
        called_names = {o.get("name") for o in all_calls}
        if decision == "tool_call" and tgt and tgt not in called_names and not meta.get("target_tools"):
            rep.warn(f"{ctx}: meta.target_tool='{tgt}' ama çağrılan {called_names}")


def check_distribution(metas, rep: Report):
    n = len(metas)
    dec = Counter(m["decision"] for m in metas)
    rep.info.append("Karar dağılımı:")
    for k, target in G.TARGET_MIX.items():
        frac = dec[k] / n if n else 0
        flag = "" if abs(frac - target) <= TOL else "  <-- HEDEFTEN SAPMA"
        rep.info.append(f"  {k:18s} {dec[k]:5d}  %{100*frac:5.1f}  (hedef %{100*target:.0f}){flag}")
        if abs(frac - target) > TOL:
            rep.warn(f"karar dağılımı '{k}' hedeften sapıyor: %{100*frac:.1f} vs %{100*target:.0f}")

    # her cannot_answer domain'e yayılmış mı (§17)
    cannot_domains = Counter(m["domain"] for m in metas if m["decision"] == "cannot_answer")
    if len(cannot_domains) < 4:
        rep.warn(f"cannot_answer yalnızca {len(cannot_domains)} domain'e yayılmış: {dict(cannot_domains)}")
    rep.info.append(f"cannot_answer domain yayılımı: {dict(cannot_domains)}")
    for d in ("puantaj", "ik_islemleri"):
        if cannot_domains.get(d, 0) == 0:
            rep.warn(f"cannot_answer '{d}' alanında hiç örnek yok (§17)")

    # tur yapısı
    turns = Counter(m.get("turns") for m in metas)
    chains = sum(1 for m in metas if m.get("chain"))
    mt = sum(1 for m in metas if m.get("multi_turn"))
    dif = Counter(m["difficulty"] for m in metas)
    rep.info.append(f"tur dağılımı: {dict(sorted(turns.items()))}  |  çok turlu: {mt}  |  6-turlu zincir: {chains}")
    rep.info.append(f"zorluk dağılımı: {dict(dif)}")
    # 6-turlu zincir örnekleri gerçekten 6 tur olmalı
    for meta in metas:
        if meta.get("chain") and meta.get("turns") != 6:
            rep.err(f"chain örneği {meta.get('id')} 6 tur değil ({meta.get('turns')})")


def check_diversity(recs_metas, rep: Report):
    by_intent = defaultdict(list)
    first_users = []
    full_sigs = []
    for rec, meta in recs_metas:
        users = [m["content"] for m in rec["messages"] if m["role"] == "user"]
        first_users.append(users[0])
        full_sigs.append(G.norm_sig(" || ".join(users)))
        by_intent[(meta["decision"], meta["intent"])].append(G.norm_sig(users[0]))

    dup = [t for t, c in Counter(full_sigs).items() if c > 1]
    if dup:
        rep.err(f"{len(dup)} adet birebir tekrar eden konuşma (tüm kullanıcı turları aynı)")

    low = []
    for (dec, intent), sigs in by_intent.items():
        if len(sigs) >= 12:
            ratio = len(set(sigs)) / len(sigs)
            if ratio < 0.6:
                low.append(f"{dec}/{intent} (%{100*ratio:.0f}, n={len(sigs)})")
    if low:
        rep.warn("düşük yüzey çeşitliliği: " + ", ".join(low))

    # aşırı sık ilk-4-kelime öneki (stil öneklerini hariç tut)
    style_pref = ("sayın", "ilgili", "bilgi", "merhaba,", "ya", "abi", "şuna", "bir", "pardon",
                  "önümüzdeki", "yöneticimle", "muhasebeyle", "kafam", "bu", "aylık")
    pref = Counter()
    for fu in first_users:
        toks = _fold(fu).split()
        if toks and toks[0] in style_pref:
            continue
        pref[" ".join(toks[:4])] += 1
    n = len(first_users)
    for p, c in pref.most_common(5):
        if c / n > 0.05:
            rep.warn(f"çok sık ilk-4-kelime öneki: '{p}' (%{100*c/n:.1f})")


def run(dirpath: Path, prefix: str, split: str, rep: Report):
    data_p = dirpath / f"{prefix}_{split}.jsonl"
    meta_p = dirpath / f"{prefix}_{split}.meta.jsonl"
    if not data_p.exists():
        rep.err(f"dosya yok: {data_p}")
        return []
    data = load_jsonl(data_p)
    metas = load_jsonl(meta_p) if meta_p.exists() else []
    if metas and len(metas) != len(data):
        rep.err(f"{split}: meta ({len(metas)}) ve data ({len(data)}) satır sayısı farklı")
    meta_by_line = {i: m for i, m in metas}

    recs_metas = []
    for (i, rec) in data:
        meta = meta_by_line.get(i, {})
        check_record(i, rec, meta, rep)
        recs_metas.append((rec, meta))
    return recs_metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--prefix", default=G.DEFAULT_PREFIX)
    ap.add_argument("--report", default=None,
                    help="markdown rapor yolu (varsayılan: depo kökündeki docs/validation_report.md)")
    args = ap.parse_args()

    default_report = Path(__file__).resolve().parent.parent / "docs" / "validation_report.md"

    d = Path(args.dir)
    rep = Report()

    all_rm = []
    for split in ("train", "val"):
        all_rm += run(d, args.prefix, split, rep)

    metas = [m for _, m in all_rm if m]
    if metas:
        check_distribution(metas, rep)
    check_diversity(all_rm, rep)

    # --- özet ---
    lines = ["# Büyük İK dataset — doğrulama raporu\n"]
    lines.append(f"- Toplam örnek: **{len(all_rm)}**")
    lines.append(f"- HATA: **{len(rep.errors)}**   |   UYARI: **{len(rep.warnings)}**\n")
    if rep.info:
        lines.append("## Bilgi\n```")
        lines += rep.info
        lines.append("```")
    if rep.errors:
        lines.append("\n## Hatalar\n")
        lines += [f"- {e}" for e in rep.errors[:200]]
        if len(rep.errors) > 200:
            lines.append(f"- ... (+{len(rep.errors)-200} daha)")
    if rep.warnings:
        lines.append("\n## Uyarılar\n")
        lines += [f"- {w}" for w in rep.warnings[:120]]
        if len(rep.warnings) > 120:
            lines.append(f"- ... (+{len(rep.warnings)-120} daha)")
    out = "\n".join(lines) + "\n"

    print(out)
    report_path = Path(args.report) if args.report else default_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(out, encoding="utf-8", newline="\n")

    if rep.errors:
        print(f"[X] {len(rep.errors)} HATA — dataset geçersiz")
        sys.exit(1)
    print(f"[OK] yapısal/anlamsal kontroller geçti ({len(rep.warnings)} uyarı)")


if __name__ == "__main__":
    main()
