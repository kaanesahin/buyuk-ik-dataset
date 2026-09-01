# -*- coding: utf-8 -*-
"""Senaryo üreticileri — hepsi TOOL ŞEMASINDAN türetir (per-tool şablon yok).

Her üretici bir `Record` döndürür: {tools, messages, meta}. `meta` yalnız QC /
istatistik / eval içindir; eğitim dosyasına yazılmaz.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from catalog import Param
from . import frames as F
from . import synth as S
from .synth import Slot

TOOLCALL_OPEN, TOOLCALL_CLOSE = "<tool_call>", "</tool_call>"


def tc_block(name, args):
    return f"{TOOLCALL_OPEN}\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + f"\n{TOOLCALL_CLOSE}"


@dataclass
class Record:
    tools: list
    messages: list
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        # tek chokepoint: 'İ'.lower() kaynaklı birleşen-nokta (U+0307) artefaktını
        # her mesajdan temizle (K-minor: NFC pass).
        for m in self.messages:
            c = m.get("content")
            if isinstance(c, str):
                m["content"] = F.denorm(c)


# --------------------------------------------------------------------------- #
#  Yardımcılar
# --------------------------------------------------------------------------- #
_PRIMARY_KINDS = ("emp_id", "id", "name")


def _synth_param(p, rng, today, direction=0):
    return S.synth(p.kind, rng, today, enum=p.enum, surface_map=p.smap,
                   id_prefix=p.prefix, id_digits=p.digits, direction=direction)


def _param_to_args(p, slot: Slot, args: dict):
    """Bir Slot'u tool_call argümanına yaz (date_range -> iki alan)."""
    if p.kind == "date_range":
        cs, ce = slot.canonical
        # ad kalıbından iki alanı bul
        keys = [p.name]
        args[p.name] = cs  # geçici; çağıran düzeltir
        return ("range", cs, ce)
    args[p.name] = slot.canonical
    return None


def _disp(slot: Slot):
    if slot.aux and slot.aux.get("disp"):
        return slot.aux["disp"]
    return slot.surface


def _human_phrase(p, slot: Slot):
    return f"{p.human}: {_disp(slot)}"


def _fill_user(rng, tool, must_surfaces, subj_slot, *, allow_oblique=True,
               oblique_p=0.62, kind="tool_call"):
    """must_surfaces: [(param, slot)] kullanıcı metninde GEÇMESİ gereken parametreler
       subj_slot: (param, slot) veya None — özneye gömülecek birincil kimlik"""
    subj = ""
    if subj_slot:
        p, sl = subj_slot
        if p.kind == "emp_id" or (p.kind == "id"):
            subj = rng.choice(F.SUBJ_EMP).format(emp=sl.surface)
        elif p.kind == "name":
            subj = rng.choice(F.SUBJ_NAME).format(name=sl.surface)
        else:
            subj = ""
    else:
        subj = rng.choice(F.SUBJ_SELF)

    plist_phrase = ", ".join(_human_phrase(p, s) for p, s in must_surfaces)
    plist_bare = ", ".join(_disp(s) for _, s in must_surfaces)
    has_extra = bool(must_surfaces)
    syn = rng.choice(tool.syn) if tool.syn else None

    if kind == "missing":
        pool = F.USER_FRAMES_MISSING
    elif kind == "chain":
        pool = F.USER_FRAMES_CHAIN_START
    elif kind == "write":
        pool = F.USER_FRAMES_WRITE
    else:
        pool = (F.USER_FRAMES_OBLIQUE if (allow_oblique and syn and rng.random() < oblique_p)
                else F.USER_FRAMES_DIRECT)

    ctx = {
        "subj": subj,
        "subj_cap": (_cap(subj)) if subj else "",
        "subj_tail": rng.choice(F.SUBJ_TAIL),
        "obj": tool.obj, "obj_nom": tool.obj_nom,
        "verb": rng.choice(tool.verbs),
        "verb2": rng.choice(tool.verbs),
        "syn": syn or "",
        "plist_phrase": plist_phrase, "plist_bare": plist_bare,
        "q_nom": rng.choice(F.Q_NOM),
    }
    txt = None
    rng.shuffle(pool := list(pool))
    for fr in pool:
        needs_plist = ("{plist_phrase}" in fr) or ("{plist_bare}" in fr)
        needs_syn = "{syn}" in fr
        if needs_syn and not syn:
            continue
        if has_extra and not needs_plist:
            continue
        if (not has_extra) and needs_plist:
            continue
        try:
            txt = fr.format(**ctx)
        except (KeyError, IndexError):
            txt = None
            continue
        break
    if txt is None:
        txt = f"{subj}{tool.obj} {ctx['verb']}"
        if has_extra:
            txt += f": {plist_phrase}"
    txt = re.sub(r"\s+", " ", txt).strip(" ,;")
    txt = _cap(txt)

    # ÖNCE stillendir (kısa/yazım-hatası kaydı içeriği kısaltabilir) SONRA garanti et
    styled, reg = F.style(rng, txt)
    needed = []
    if subj_slot:
        p, sl = subj_slot
        needed.append((f"{p.human}", sl))
    needed += [(p.human, s) for p, s in must_surfaces]
    styled = _ensure_grounded(styled, needed)
    return styled, reg, _kw_hit(tool, styled)


def _ensure_grounded(txt, needed):
    """needed: [(label, slot)] — hepsi metinde bulunmalı; eksikleri parantezle ekle."""
    def present(sl):
        t = _fold(txt)
        surf = _fold(str(sl.surface))
        if surf.isdigit():                       # çıplak sayı: sınır ara
            if re.search(rf"(?<!\d){re.escape(surf)}(?!\d)", re.sub(r"[.\s](?=\d)", "", t)):
                return True
        elif surf and surf in t:
            return True
        if isinstance(sl.canonical, tuple):      # date_range: iki tarih de görünmeli
            return all(_fold(x) in t for x in sl.canonical)
        num = re.sub(r"\D", "", str(sl.canonical))
        if num and len(num) >= 3 and re.search(rf"(?<!\d){num}(?!\d)",
                                               re.sub(r"(?<=\d)[.\s](?=\d)", "", t)):
            return True
        return False
    miss = [f"{lbl}: {_disp(s)}" for lbl, s in needed if not present(s)]
    if miss:
        txt = txt.rstrip(" .?!,;") + " (" + ", ".join(miss) + ")"
    return txt


def _fold(s):
    return F._fold(s)


def _cap(s):
    """Türkçe-güvenli ilk harf büyütme (i -> İ)."""
    return F.tr_upper(s[:1]) + s[1:]


def _kw_hit(tool, text):
    """Ayırt edici yüzey sözcüğü metinde geçiyor mu (K-1 ölçümü için)."""
    f = _fold(text)
    return any(_fold(k) in f for k in tool.disc_kw)


def _display(v):
    if isinstance(v, (int, float)):
        return f"{v:,}".replace(",", ".")
    return str(v)


_COMPANIES = ("Aksu Tekstil A.Ş.", "Delta Lojistik", "Marmara Gıda", "Ege Yazılım",
              "Nova Enerji", "Batı İnşaat", "Anadolu Kimya", "Zirve Otomotiv",
              "Kuzey Denizcilik", "Star Ambalaj", "PELİKAN Kırtasiye", "Meridyen Danışmanlık")
_INDUSTRIES = ("Perakende", "Üretim", "Bilişim", "Lojistik", "Sağlık", "İnşaat",
               "Enerji", "Finans", "Tarım", "Turizm", "Otomotiv", "Telekom")
_TERMS = ("30 gün vadeli", "60 gün vadeli", "peşin", "45 gün vadeli", "kapıda ödeme")
_LOCATIONS = ("Gebze Deposu", "Hadımköy Antrepo", "yolda - Ankara yakını", "İzmir Şube",
              "Merkez Depo", "gümrükte", "dağıtım merkezinde")
_STATES = ("pending", "approved", "in_progress", "resolved", "completed", "rejected", "on_hold")
_CURRENCIES = ("TRY", "EUR", "USD")
_ASSET_TYPES = ("dizüstü bilgisayar", "masaüstü bilgisayar", "cep telefonu", "monitör",
                "tablet", "yazıcı", "docking istasyonu")
_PRODUCT_NAMES = ("A4 fotokopi kağıdı", "toner kartuş", "arşiv klasörü", "termal etiket rulosu",
                  "koli bandı", "USB bellek 32GB", "kablo kanalı", "beyaz tahta kalemi")
_REPORT_SUMMARIES = ("hedeflerin çoğu tutturuldu, önceki döneme göre artış var",
                     "kayıt sayısı düştü, ortalama çözüm süresi iyileşti",
                     "üç kalemde bütçe aşımı görünüyor, kalan çeyrekte önlem gerekli",
                     "sonuçlar yatay seyrediyor, en yüksek katkı kurumsal segmentten",
                     "gecikme oranı azaldı, müşteri memnuniyeti hafif yükseldi")
_ESCALATION_TEAMS = ("2. kademe destek", "kıdemli mühendislik", "ürün ekibi",
                     "saha operasyon", "müşteri başarı ekibi")
_CHURN_FACTORS = ("fiyat", "hizmet kalitesi", "rakip teklifi", "kullanım düşüşü", "destek gecikmesi")

# isim / şirket adına işaret eden anahtar kökleri
_PERSON_KEYS = {"full_name", "manager", "manager_name", "owner", "assignee", "assigned_to",
                "decided_by", "approver", "signer", "reporter", "created_by", "requested_by",
                "contact_name", "employee_name", "rep_name"}
_ORG_KEYS = {"account", "account_name", "company", "company_name", "vendor_name", "customer_name"}


def _synth_result_value(key, kind, rng, today, tool=None):
    k = key.lower()
    dom = tool.domain if tool else ""

    if k in ("status", "state"):
        return rng.choice(_STATES)
    if k in ("score", "nps", "utilization", "health") or k.endswith("_score"):
        return rng.randint(35, 98)
    if k == "breached":
        return rng.choice([0, 0, 0, 1])
    if k in ("sub_teams", "payments", "open_deals", "open_cases", "records_count"):
        return rng.randint(0, 8)          # küçük sayılar (birim başı alt-ekip vb.)
    if k in ("currency", "currency_code", "para_birimi"):
        return rng.choice(_CURRENCIES)

    # --- şirket / hesap / tedarikçi adı (kişiden ÖNCE kontrol) ---
    if k in _ORG_KEYS or ("name" in k and any(x in k for x in ("vendor", "company", "account", "customer"))):
        return rng.choice(_COMPANIES)
    if k == "name" and kind == "title":            # tedarikçi/ürün kaydının adı
        return rng.choice(_PRODUCT_NAMES) if dom == "inventory" else rng.choice(_COMPANIES)

    # --- kişi adı ---
    if kind == "name" or k in _PERSON_KEYS:
        return S.gen_name(rng, full=True).canonical

    if "industry" in k or "sector" in k:
        return rng.choice(_INDUSTRIES)
    if "term" in k or k == "payment_terms":
        return rng.choice(_TERMS)
    if k in ("type", "asset_type", "device_type", "item_type"):
        return rng.choice(_ASSET_TYPES)
    if k in ("summary", "highlights", "headline", "note", "notes"):
        return rng.choice(_REPORT_SUMMARIES)
    if k == "escalated_to":
        return rng.choice(_ESCALATION_TEAMS)
    if k == "top_factor":
        return rng.choice(_CHURN_FACTORS)
    if k == "title":                               # unvan mı belge başlığı mı — domaine göre
        return rng.choice(S.DOC_TITLES) if dom in ("documents", "reporting") else rng.choice(S.TITLES)
    if k in ("manager_title", "unit", "category", "plan", "sla_tier", "trend", "carrier"):
        pool = {"unit": S.ORG_UNITS, "category": ("Kırtasiye", "Elektronik", "Ambalaj", "Gıda"),
                "plan": ("Standart", "Kurumsal", "Premium"), "sla_tier": ("Gümüş", "Altın", "Platin"),
                "trend": ("yükseliyor", "sabit", "düşüyor"), "carrier": ("Aras", "MNG", "Yurtiçi", "UPS"),
                "manager_title": S.TITLES}
        return rng.choice(pool.get(k, S.TITLES))
    if k == "location":
        return rng.choice(_LOCATIONS)

    if kind == "amount":
        return S.gen_amount(rng).canonical
    if kind in ("count", "hours", "pct", "duration"):
        return S.synth(kind, rng, today).canonical
    if kind in ("past_date", "future_date", "date"):
        return S.synth(kind, rng, today).canonical
    if kind == "period":
        return S.gen_period(rng, today).canonical
    if kind == "id":
        return S.gen_id(rng, "REF").canonical
    if kind == "email":
        return S.gen_email(rng).canonical
    if kind == "enum":
        return rng.choice(_STATES)
    if kind == "title":
        return rng.choice(S.TITLES)
    # son çare — yalnız gerçekten sayısal ölçüt kalır (isim/başlık artık buraya düşmez)
    return rng.randint(1, 90)


_RESULT_LABELS = {
    "annual_left": "yıllık kalan", "excuse_left": "mazeret kalan", "sick_left": "hastalık kalan",
    "amount": "tutar", "net": "net", "gross": "brüt", "total": "toplam", "status": "durum",
    "state": "durum", "hours": "saat", "pay": "karşılık", "count": "adet", "score": "skor",
    "eta": "tahmini varış", "location": "konum", "manager_name": "yönetici",
    "full_name": "ad", "unit": "birim", "title": "unvan", "hire_date": "işe giriş",
    "worked_days": "çalışılan gün", "absences": "devamsızlık", "remaining": "kalan",
    "spent": "harcanan", "allocated": "ayrılan", "on_hand": "eldeki", "reserved": "rezerve",
    "records_count": "kayıt sayısı", "total_days": "toplam gün", "matches": "eşleşme",
    "manager_title": "yönetici unvanı", "close_date": "kapanış tarihi", "stage": "aşama",
    "open_deals": "açık fırsat", "industry": "sektör", "owner": "sorumlu", "carrier": "taşıyıcı",
    "response_due": "yanıt son tarihi", "breached": "ihlal", "target": "hedef", "achieved": "gerçekleşen",
    "pct": "oran", "payment_terms": "ödeme koşulu", "name": "ad", "due_date": "vade",
    "capacity": "kapasite", "utilization": "doluluk", "list_price": "liste fiyatı", "discount": "indirim",
    "allocated_": "ayrılan", "assignee": "atanan", "since": "başlangıç", "sub_teams": "alt ekip",
    "headcount": "kişi sayısı", "plan": "paket", "open_cases": "açık kayıt", "sla_tier": "SLA seviyesi",
    "trend": "eğilim", "cumulative_base": "kümülatif matrah", "tax_paid": "ödenen vergi",
    "bracket": "dilim", "severance": "kıdem", "notice": "ihbar", "years": "yıl", "items": "kalem",
    "shifts": "vardiya", "late_count": "geç kalma", "absent_days": "devamsız gün", "sgk": "SGK",
    "income_tax": "gelir vergisi", "stamp": "damga", "deductions": "kesinti", "delivery_date": "teslim",
    "valid_until": "geçerlilik", "rows": "satır", "generated_at": "üretim zamanı", "revenue": "ciro",
    "nps": "NPS", "free_slots": "boş aralık", "slots": "uygun aralık", "top_factor": "başlıca etken",
    "escalated_to": "yükseltilen ekip", "ready_date": "hazır tarih", "paid_date": "ödeme tarihi",
    "version": "sürüm", "adjustment": "düzeltme", "transit_days": "transit gün", "price": "fiyat",
    "unit_price": "birim fiyat", "decided_by": "karar veren", "currency": "para birimi",
    "manager": "yönetici", "assigned_to": "zimmetli", "type": "tür", "account": "bağlı hesap",
    "summary": "özet", "category": "kategori", "daily_cap": "günlük tavan",
    "needs_receipt": "fiş gerekli", "payments": "ödeme adedi", "row_count": "satır sayısı",
    "email": "e-posta",
}


def _label(k, dom=None):
    if k == "title" and dom in ("documents", "reporting"):
        return "başlık"
    return _RESULT_LABELS.get(k, k.replace("_", " "))


def _harmonize_result(res, rng):
    """tool sonucunun iç tutarlılığı: net ≤ brüt, kesinti = brüt − net, kalan = ayrılan − harcanan."""
    base = res.get("gross") if isinstance(res.get("gross"), (int, float)) else res.get("total")
    if isinstance(base, (int, float)) and isinstance(res.get("net"), (int, float)):
        if res["net"] > base:
            res["net"] = int(base * rng.uniform(0.62, 0.85) / 500) * 500
        if isinstance(res.get("deductions"), (int, float)):
            res["deductions"] = max(0, base - res["net"])
    alloc = res.get("allocated")
    if isinstance(alloc, (int, float)) and isinstance(res.get("spent"), (int, float)):
        res["spent"] = min(res["spent"], alloc)
        if isinstance(res.get("remaining"), (int, float)):
            res["remaining"] = alloc - res["spent"]
    return res


def synth_result(tool, rng, today, mode="ok", echo=None):
    """mode: ok | empty | error | partial"""
    if mode == "error":
        err = rng.choice(["zaman aşımı", "servis geçici olarak kapalı", "yetki hatası", "bağlantı hatası"])
        return {"error": err}, err
    fields = list(tool.result)
    if mode == "empty":
        return ({"records": [], "count": 0} if any(k in ("count", "matches", "rows")
                for k, _ in fields) else {"result": None}), None
    if mode == "partial" and len(fields) > 1:
        fields = fields[: max(1, len(fields) // 2)]
    res = {}
    for k, kind in fields:
        res[k] = _synth_result_value(k, kind, rng, today, tool)
    _harmonize_result(res, rng)
    if echo:
        res.update(echo)
    return res, None


def result_phrase(res, tool=None):
    dom = tool.domain if tool else None
    parts = []
    for k, v in res.items():
        if k in ("error", "matches"):
            continue
        parts.append(f"{_label(k, dom)} {_display(v)}")
    return "; ".join(parts) if parts else "kayıt döndü"


def final_answer(rng, tool, res, mode, today):
    if mode == "error":
        return rng.choice(F.RESULT_ERROR).format(
            obj_nom_cap=_cap(tool.obj_nom),
            obj_nom=tool.obj_nom, err=res.get("error", "hata"))
    if mode == "empty":
        return rng.choice(F.RESULT_EMPTY).format(
            obj_nom_cap=_cap(tool.obj_nom), obj_nom=tool.obj_nom)
    ph = result_phrase(res, tool)
    tmpl = F.RESULT_PARTIAL if mode == "partial" else F.RESULT_OK
    return rng.choice(tmpl).format(
        obj_nom_cap=_cap(tool.obj_nom), obj_nom=tool.obj_nom,
        result_phrase=ph, subj_res="")


# --------------------------------------------------------------------------- #
#  SENARYO: tekli READ  (tool_call, tüm zorunlu param var)  [+ opsiyonel tool sonucu]
# --------------------------------------------------------------------------- #
def _date_pairs(tool):
    """('start_date','end_date') gibi tarih çiftlerini döndür (tutarlı aralık için)."""
    dnames = [p.name for p in tool.params if p.kind in ("date", "future_date", "past_date")]
    pairs = []
    for s in dnames:
        for a, b in (("start", "end"), ("_from", "_to")):
            if a in s:
                e = s.replace(a, b)
                if e in dnames and (s, e) not in pairs:
                    pairs.append((s, e))
    return pairs


def _apply_date_pairs(tool, rng, today, args, must, want_names, direction=0):
    """want_names: doldurulacak param adları. Çift olan tarihleri tutarlı aralık yap."""
    handled = set()
    for s_name, e_name in _date_pairs(tool):
        if s_name not in want_names and e_name not in want_names:
            continue
        if direction == 0 and tool.param(s_name).kind == "future_date":
            direction = 1
        sl = S.gen_date_range(rng, today, direction=direction)
        cs, ce = sl.canonical
        args[s_name] = cs
        args[e_name] = ce
        must.append((Param("tarih aralığı", "date_range", True, human="tarih aralığı"), sl))
        handled |= {s_name, e_name}
    return handled


def gen_read_call(rng, idx, tool, with_result_p=0.52):
    today = idx.today
    args = {}
    must = []          # (param, slot) kullanıcı metninde geçecek (subj hariç)
    subj_slot = None
    req_names = {p.name for p in tool.params if p.required}
    handled = _apply_date_pairs(tool, rng, today, args, must, req_names)
    # birincil kimlik / özne
    for p in tool.params:
        if not p.required or p.name in handled:
            continue
        sl = _synth_param(p, rng, today)
        if p.kind in _PRIMARY_KINDS and subj_slot is None:
            subj_slot = (p, sl)
            if p.kind == "date_range":
                pass
            args[p.name] = sl.canonical
        elif p.kind == "date_range":
            cs, ce = sl.canonical
            ln = _range_fields(tool, p)
            args[ln[0]] = cs
            args[ln[1]] = ce
            must.append((p, sl))
        else:
            args[p.name] = sl.canonical
            must.append((p, sl))
    # opsiyonel param'lar (D-10): %35 dahil et
    opt_meta = []
    for p in tool.params:
        if p.required:
            continue
        if rng.random() < 0.42:
            sl = _synth_param(p, rng, today)
            if p.kind == "date_range":
                ln = _range_fields(tool, p)
                cs, ce = sl.canonical
                args[ln[0]] = cs; args[ln[1]] = ce
            else:
                args[p.name] = sl.canonical
            must.append((p, sl))
            opt_meta.append(p.name)

    utext, reg, kw = _fill_user(rng, tool, must, subj_slot, oblique_p=0.72)

    msgs = [{"role": "user", "content": utext},
            {"role": "assistant", "content": tc_block(tool.name, args)}]

    has_result = rng.random() < with_result_p
    rmode = "ok"
    if has_result:
        rmode = rng.choices(["ok", "ok", "ok", "empty", "error", "partial"],
                            weights=[55, 0, 0, 18, 15, 12])[0]
        res, _ = synth_result(tool, rng, today, rmode)
        msgs.append({"role": "tool", "content": json.dumps(res, ensure_ascii=False)})
        msgs.append({"role": "assistant", "content": final_answer(rng, tool, res, rmode, today)})

    tools, names = idx.index.candidate_list(rng, [tool.name])
    return Record(tools, msgs, {
        "decision": "tool_call", "scenario": "read_call", "target_tools": [tool.name],
        "domain": tool.domain, "category": tool.cat, "register": reg,
        "turns": len(msgs), "candidate_count": len(names), "has_tool_result": has_result,
        "tool_result_mode": rmode if has_result else None,
        "keyword_in_prompt": kw, "optional_params_used": opt_meta,
        "missing_params": [], "is_write": False, "hard_negative": None,
    })


def _range_fields(tool, p):
    """date_range param için tool şemasındaki iki alan adını bul."""
    names = [x.name for x in tool.params]
    base = p.name
    cands = [n for n in names if n != base and ("date" in n or "start" in n or "end" in n)]
    # tipik: start_date/end_date, new_start_date/new_end_date
    starts = [n for n in names if "start" in n]
    ends = [n for n in names if "end" in n]
    if starts and ends:
        return (starts[0], ends[0])
    return (base, cands[0] if cands else base + "_end")


# --------------------------------------------------------------------------- #
#  SENARYO: eksik parametre  (request_for_info)
# --------------------------------------------------------------------------- #
def gen_missing_param(rng, idx, tool):
    today = idx.today
    req = [p for p in tool.params if p.required]
    if not req:
        return None
    n_missing = 1 if len(req) <= 2 or rng.random() < 0.7 else 2
    missing = rng.sample(req, min(n_missing, len(req)))
    present = [p for p in req if p not in missing]

    args_present = {}
    must = []
    subj_slot = None
    for p in present:
        sl = _synth_param(p, rng, today)
        if p.kind in _PRIMARY_KINDS and subj_slot is None:
            subj_slot = (p, sl)
        elif p.kind == "date_range":
            must.append((p, sl))
        else:
            must.append((p, sl))
    utext, reg, kw = _fill_user(rng, tool, must, subj_slot, kind="missing", oblique_p=0.5)

    if len(missing) == 1:
        ask = rng.choice(F.ASK_MISSING).format(
            human=missing[0].human, human_cap=_cap(missing[0].human))
    else:
        hl = " ve ".join(m.human for m in missing)
        ask = rng.choice(F.ASK_MISSING_MULTI).format(human_list=hl)

    tools, names = idx.index.candidate_list(rng, [tool.name])
    return Record(tools, [
        {"role": "user", "content": utext},
        {"role": "assistant", "content": ask},
    ], {
        "decision": "request_for_info", "scenario": "missing_param", "target_tools": [tool.name],
        "domain": tool.domain, "category": tool.cat, "register": reg, "turns": 2,
        "candidate_count": len(names), "has_tool_result": False,
        "missing_params": [m.name for m in missing], "is_write": tool.cat == "write",
        "keyword_in_prompt": kw, "hard_negative": None,
    })


# --------------------------------------------------------------------------- #
#  SENARYO: WRITE onay iste  (request_for_info)  — tüm param var
# --------------------------------------------------------------------------- #
def _write_args_and_summary(rng, idx, tool, include_optional=True):
    today = idx.today
    args = {}
    surfaces = []
    subj_slot = None
    want = {p.name for p in tool.params if p.required or
            (include_optional and rng.random() < 0.3)}
    handled = _apply_date_pairs(tool, rng, today, args, surfaces, want, direction=1)
    for p in tool.params:
        if p.name not in want or p.name in handled:
            continue
        if p.kind == "name":
            sl = S.gen_name(rng, full=True)
        else:
            sl = _synth_param(p, rng, today, direction=1 if p.kind == "future_date" else 0)
        if p.kind == "date_range":
            ln = _range_fields(tool, p)
            cs, ce = sl.canonical
            args[ln[0]] = cs; args[ln[1]] = ce
        else:
            args[p.name] = sl.canonical
        if p.kind in _PRIMARY_KINDS and subj_slot is None:
            subj_slot = (p, sl)
        surfaces.append((p, sl))
    summ = _summary_text(tool, surfaces)
    return args, surfaces, subj_slot, summ


def _summary_text(tool, surfaces):
    bits = [(s.canonical if p.kind in ("emp_id", "id") else s.surface)
            for p, s in surfaces if p.kind in _PRIMARY_KINDS]
    detail = [f"{p.human}: {_disp(s)}" for p, s in surfaces if p.kind not in _PRIMARY_KINDS]
    head = (str(bits[0]) + " için ") if bits else ""
    return f"{head}{tool.obj_nom}" + (f" ({', '.join(detail)})" if detail else "")


def gen_write_confirm(rng, idx, tool):
    args, surfaces, subj_slot, summ = _write_args_and_summary(rng, idx, tool)
    must = [(p, s) for p, s in surfaces if not (subj_slot and p is subj_slot[0])]
    utext, reg, kw = _fill_user(rng, tool, must, subj_slot, kind="write", oblique_p=0.4)
    ask = rng.choice(F.CONFIRM_ASK).format(
        summary=summ, summary_cap=_cap(summ))
    tools, names = idx.index.candidate_list(rng, [tool.name])
    return Record(tools, [
        {"role": "user", "content": utext},
        {"role": "assistant", "content": ask},
    ], {
        "decision": "request_for_info", "scenario": "write_confirm", "target_tools": [tool.name],
        "domain": tool.domain, "category": tool.cat, "register": reg, "turns": 2,
        "candidate_count": len(names), "has_tool_result": False, "missing_params": [],
        "is_write": True, "confirmation": True, "keyword_in_prompt": kw, "hard_negative": None,
    })


# --------------------------------------------------------------------------- #
#  SENARYO: WRITE yürüt  (tool_call, 4 tur: iste -> onay -> çağır) [+ done]
# --------------------------------------------------------------------------- #
def gen_write_execute(rng, idx, tool, with_result_p=0.5):
    today = idx.today
    args, surfaces, subj_slot, summ = _write_args_and_summary(rng, idx, tool)
    must = [(p, s) for p, s in surfaces if not (subj_slot and p is subj_slot[0])]
    utext, reg, kw = _fill_user(rng, tool, must, subj_slot, kind="write", oblique_p=0.4)
    ask = rng.choice(F.CONFIRM_ASK).format(summary=summ, summary_cap=_cap(summ))
    msgs = [
        {"role": "user", "content": utext},
        {"role": "assistant", "content": ask},
        {"role": "user", "content": rng.choice(F.ACK)},
        {"role": "assistant", "content": tc_block(tool.name, args)},
    ]
    has_result = rng.random() < with_result_p
    if has_result:
        res, _ = synth_result(tool, rng, today, "ok")
        ref = next((v for k, v in res.items()
                    if isinstance(v, str) and re.fullmatch(r"[A-Z]{2,5}-\d+", v)), None)
        if ref is None:                      # sonuçta kimlik yoksa sonuca EKLE (grounding)
            ref = S.gen_id(rng, "REQ").canonical
            res["reference"] = ref
        msgs.append({"role": "tool", "content": json.dumps(res, ensure_ascii=False)})
        msgs.append({"role": "assistant", "content": rng.choice(F.WRITE_DONE).format(
            summary=summ, summary_cap=_cap(summ), ref=ref)})
    tools, names = idx.index.candidate_list(rng, [tool.name])
    return Record(tools, msgs, {
        "decision": "tool_call", "scenario": "write_execute", "target_tools": [tool.name],
        "domain": tool.domain, "category": tool.cat, "register": reg, "turns": len(msgs),
        "candidate_count": len(names), "has_tool_result": has_result, "missing_params": [],
        "is_write": True, "confirmation": True, "keyword_in_prompt": kw, "hard_negative": None,
    })


# --------------------------------------------------------------------------- #
#  SENARYO: WRITE zinciri  (6 tur: eksik param -> ver -> onay -> ver -> çağır)
# --------------------------------------------------------------------------- #
def gen_write_chain(rng, idx, tool):
    today = idx.today
    req = [p for p in tool.params if p.required]
    if len(req) < 2:
        return None
    pair_names = {n for pr in _date_pairs(tool) for n in pr}
    missing = rng.choice([p for p in req if p.kind not in ("emp_id",) and p.name not in pair_names]
                         or [p for p in req if p.kind not in ("emp_id",)] or req)
    present = [p for p in req if p is not missing]
    args = {}
    surfaces = []
    subj_slot = None
    handled = _apply_date_pairs(tool, rng, today, args, surfaces, {p.name for p in present}, direction=1)
    for p in present:
        if p.name in handled:
            continue
        sl = _synth_param(p, rng, today, direction=1 if p.kind == "future_date" else 0)
        args[p.name] = sl.canonical
        if p.kind in _PRIMARY_KINDS and subj_slot is None:
            subj_slot = (p, sl)
        surfaces.append((p, sl))
    must = [(p, s) for p, s in surfaces if not (subj_slot and p is subj_slot[0])]
    u1, reg, kw = _fill_user(rng, tool, must, subj_slot, kind="chain")
    a1 = rng.choice(F.CHAIN_ASK_PARAM).format(
        human=missing.human, human_cap=_cap(missing.human))
    # eksik param'ı ver
    msl = _synth_param(missing, rng, today, direction=1 if missing.kind == "future_date" else 0)
    if missing.kind == "date_range":
        ln = _range_fields(tool, missing); cs, ce = msl.canonical
        args[ln[0]] = cs; args[ln[1]] = ce
    else:
        args[missing.name] = msl.canonical
    u2 = rng.choice(["{s}", "{s} olsun", "Şöyle: {s}", "{s} deyelim"]).format(s=msl.surface)
    surfaces.append((missing, msl))
    summ = _summary_text(tool, surfaces)
    a2 = rng.choice(F.CONFIRM_ASK).format(summary=summ, summary_cap=_cap(summ))
    u3 = rng.choice(F.ACK)
    a3 = tc_block(tool.name, args)
    tools, names = idx.index.candidate_list(rng, [tool.name], size=rng.choice([None, None, 12]))
    return Record(tools, [
        {"role": "user", "content": u1}, {"role": "assistant", "content": a1},
        {"role": "user", "content": u2}, {"role": "assistant", "content": a2},
        {"role": "user", "content": u3}, {"role": "assistant", "content": a3},
    ], {
        "decision": "tool_call", "scenario": "write_chain", "target_tools": [tool.name],
        "domain": tool.domain, "category": tool.cat, "register": reg, "turns": 6,
        "candidate_count": len(names), "has_tool_result": False,
        "missing_params": [missing.name], "is_write": True, "confirmation": True,
        "chain": True, "keyword_in_prompt": kw, "hard_negative": None,
    })


# --------------------------------------------------------------------------- #
#  SENARYO: paralel çoklu-tool  (aynı varlık, 2 READ, tek asistan turu)
# --------------------------------------------------------------------------- #
def gen_multi_parallel(rng, idx, tool_a, tool_b):
    today = idx.today
    # ortak birincil param (emp_id / aynı id kind)
    pa = next((p for p in tool_a.params if p.required and p.kind in _PRIMARY_KINDS), None)
    pb = next((p for p in tool_b.params if p.required and p.kind in _PRIMARY_KINDS), None)
    if not pa or not pb or pa.kind != pb.kind:
        return None
    if pa.kind == "id" and pa.prefix != pb.prefix:
        return None
    if pa.name != pb.name and pa.kind == "id":
        return None
    sl = _synth_param(pa, rng, today)
    args_a = {pa.name: sl.canonical}
    args_b = {pb.name: sl.canonical}
    # ek zorunlu param'ları da doldur (varsa) — aynı yüzeyle
    extra_sfc = []
    for tool, args in ((tool_a, args_a), (tool_b, args_b)):
        for p in tool.params:
            if not p.required or p.name in args:
                continue
            s2 = _synth_param(p, rng, today)
            if p.kind == "date_range":
                ln = _range_fields(tool, p); cs, ce = s2.canonical
                args[ln[0]] = cs; args[ln[1]] = ce
            else:
                args[p.name] = s2.canonical
            extra_sfc.append((p, s2))
    subj = rng.choice(F.SUBJ_EMP).format(emp=sl.surface) if pa.kind != "name" \
        else rng.choice(F.SUBJ_NAME).format(name=sl.surface)
    conj = rng.choice(["ve", "ile birlikte", "bir de"])
    detail = (" — " + ", ".join(_human_phrase(p, s) for p, s in extra_sfc)) if extra_sfc else ""
    u = f"{subj}{tool_a.obj} {conj} {tool_b.obj} {rng.choice(tool_a.verbs)}{detail}"
    u = _cap(u)
    u, reg = F.style(rng, u)
    u = _ensure_grounded(u, [(pa.human, sl)] + [(p.human, s) for p, s in extra_sfc])
    call = tc_block(tool_a.name, args_a) + "\n" + tc_block(tool_b.name, args_b)
    msgs = [{"role": "user", "content": u}, {"role": "assistant", "content": call}]
    if rng.random() < 0.5:
        ra, _ = synth_result(tool_a, rng, today, "ok")
        rb, _ = synth_result(tool_b, rng, today, "ok")
        msgs.append({"role": "tool", "content": json.dumps(ra, ensure_ascii=False)})
        msgs.append({"role": "tool", "content": json.dumps(rb, ensure_ascii=False)})
        msgs.append({"role": "assistant", "content":
                     f"{_cap(tool_a.obj_nom)}: {result_phrase(ra, tool_a)}. "
                     f"{_cap(tool_b.obj_nom)}: {result_phrase(rb, tool_b)}."})
    tools, names = idx.index.candidate_list(rng, [tool_a.name, tool_b.name])
    return Record(tools, msgs, {
        "decision": "tool_call", "scenario": "multi_parallel",
        "target_tools": [tool_a.name, tool_b.name], "domain": tool_a.domain,
        "category": "read", "register": reg, "turns": len(msgs),
        "candidate_count": len(names), "has_tool_result": len(msgs) > 2,
        "missing_params": [], "is_write": False, "multi_tool": True, "hard_negative": None,
    })


# --------------------------------------------------------------------------- #
#  SENARYO: sıralı çoklu-tool  (A sonucu -> B parametresi)
# --------------------------------------------------------------------------- #
# İlişki spec'leri: (resolver, verilen_kind, üretilen_key, üretilen_kind, tüketici, tüketici_param)
# tüketici tool'un TEK zorunlu param'ı zincirlenen param olmalı (aksi -> uydurma)
CHAINS = [
    ("crm_search_contacts", "name", "contact_id", "id", "crm_get_contact", "contact_id"),
    ("crm_search_contacts", "name", "account_id", "id", "crm_list_deals", "account_id"),
    ("crm_search_contacts", "name", "account_id", "id", "crm_get_account", "account_id"),
    ("finance_get_invoice", "id:INV", "vendor_id", "id", "finance_get_vendor", "vendor_id"),
    ("sales_get_order", "id:ORD", "account_id", "id", "crm_get_account", "account_id"),
    ("support_list_open_cases", "id:CUS", "case_id", "id", "support_get_sla_status", "case_id"),
    ("it_get_ticket", "id:TIC", "asset_id", "id", "it_get_asset", "asset_id"),
]


def gen_multi_sequential(rng, idx, chain):
    today = idx.today
    r_name, given_kind, prod_key, prod_kind, c_name, c_param = chain
    resolver = idx.index.by_name.get(r_name)
    consumer = idx.index.by_name.get(c_name)
    if not resolver or not consumer:
        return None
    # yalnız hedefler train ise üret (val/test hedef olmasın)
    if resolver.split != "train" or consumer.split != "train":
        return None
    # resolver argümanı — verilen kind'e uyan param, yoksa ilk param
    if given_kind == "name":
        gsl = S.gen_name(rng, full=True)
        gp = next((p for p in resolver.params if p.kind == "name"), resolver.params[0])
    else:
        pref = given_kind.split(":")[1]
        gsl = S.gen_id(rng, pref)
        gp = next((p for p in resolver.params if p.kind in ("id", "emp_id")), resolver.params[0])
    rargs = {gp.name: gsl.canonical}
    subj = ""
    given_surface = gsl.surface
    # üretilen id
    prod_val = S.gen_id(rng, {"contact_id": "CNT", "account_id": "ACC", "vendor_id": "VEN",
                              "case_id": "CASE", "asset_id": "AST"}.get(prod_key, "REF")).canonical
    r_res = {prod_key: prod_val, "matches": 1}
    c_args = {c_param: prod_val}
    # tüketicinin zincirlenmeyen zorunlu param'ı varsa bu zincir geçersiz (uydurma olur)
    if any(p.required and p.name not in c_args for p in consumer.params):
        return None
    c_res, _ = synth_result(consumer, rng, today, "ok")

    u = (f"{given_surface} için {consumer.obj} {rng.choice(consumer.verbs)}"
         if given_kind == "name" else
         f"{given_surface} kaydından yola çıkıp {consumer.obj} {rng.choice(consumer.verbs)}")
    u = _cap(u)
    u, reg = F.style(rng, u)
    u = _ensure_grounded(u, [(gp.human, gsl)])
    msgs = [
        {"role": "user", "content": u},
        {"role": "assistant", "content": tc_block(resolver.name, rargs)},
        {"role": "tool", "content": json.dumps(r_res, ensure_ascii=False)},
        {"role": "assistant", "content": tc_block(consumer.name, c_args)},
        {"role": "tool", "content": json.dumps(c_res, ensure_ascii=False)},
        {"role": "assistant", "content":
         f"{given_surface} ({prod_val}) için {consumer.obj_nom}: {result_phrase(c_res, consumer)}."},
    ]
    tools, names = idx.index.candidate_list(rng, [resolver.name, consumer.name])
    return Record(tools, msgs, {
        "decision": "tool_call", "scenario": "multi_sequential",
        "target_tools": [resolver.name, consumer.name], "domain": consumer.domain,
        "category": "read", "register": reg, "turns": 6, "candidate_count": len(names),
        "has_tool_result": True, "missing_params": [], "is_write": False,
        "multi_tool": True, "sequential": True, "hard_negative": None,
    })


# --------------------------------------------------------------------------- #
#  SENARYO: direct  (tool gerekmiyor)
# --------------------------------------------------------------------------- #
def gen_direct(rng, idx, entry, four_turn_p=0.22):
    qs, cores, dom = entry
    q = rng.choice(qs)
    q, reg = F.style(rng, q)
    core = rng.choice(cores)
    a = rng.choice(F.DIRECT_WRAP).format(core=core, core_low=F.tr_lower(core[:1]) + core[1:])
    msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    four = rng.random() < four_turn_p and len(cores) > 1 and dom != "meta"
    if four:
        q2 = rng.choice(F.MT_DIRECT_FOLLOWUP_A)
        a2 = rng.choice([c for c in cores if c != core])
        msgs += [{"role": "user", "content": q2}, {"role": "assistant", "content": a2}]
    tools, names = idx.index.candidate_list(rng, [], size=rng.choice([None, None, 8]))
    return Record(tools, msgs, {
        "decision": "direct", "scenario": "direct", "target_tools": [],
        "domain": dom, "category": None, "register": reg, "turns": len(msgs),
        "candidate_count": len(names), "has_tool_result": False, "missing_params": [],
        "is_write": False, "hard_negative": None,
    })


# --------------------------------------------------------------------------- #
#  SENARYO: cannot_answer  (kapsam / gelecek / gizlilik / yetki)
# --------------------------------------------------------------------------- #
def gen_cannot_scope(rng, idx, entry, four_turn_p=0.2):
    qs, cat = entry
    q = rng.choice(qs)
    q, reg = F.style(rng, q)
    pool = {"out_of_scope": "out_of_scope", "future": "future", "privacy": "privacy",
            "unauthorized": "unauthorized", "advice": "advice"}[cat]
    doms = "İK, finans, CRM, envanter, satış, lojistik, takvim, destek"
    a = rng.choice(F.CANNOT[pool]).format(domains=doms) + rng.choice(F.CANNOT_REDIRECT)
    msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a.strip()}]
    if rng.random() < four_turn_p:
        msgs += [{"role": "user", "content": rng.choice(F.CANNOT_PUSH)},
                 {"role": "assistant", "content": rng.choice(F.CANNOT_HOLD)}]
    tools, names = idx.index.candidate_list(rng, [], size=rng.choice([None, None, 10]))
    return Record(tools, msgs, {
        "decision": "cannot_answer", "scenario": "cannot_scope", "target_tools": [],
        "domain": {"out_of_scope": "kapsanmayan", "future": "kapsanmayan",
                   "privacy": "gizlilik", "unauthorized": "yetki", "advice": "kapsanmayan"}[cat],
        "category": None, "register": reg, "turns": len(msgs), "candidate_count": len(names),
        "has_tool_result": False, "missing_params": [], "is_write": False,
        "hard_negative": None, "cannot_reason": cat,
    })


# --------------------------------------------------------------------------- #
#  HARD-NEGATIVE senaryoları
# --------------------------------------------------------------------------- #
def gen_hn_keyword_ambiguous(rng, idx, tool):
    """A: aynı keyword'ü paylaşan kardeşler arasında doğru tool seçimi.
       Kullanıcı ifadesi tool'un YÜZEY kelimesini içerir ama syn ile ayrışır."""
    sibs = idx.index.keyword_siblings(tool.name)
    if not sibs:
        return None
    today = idx.today
    args = {}
    must = []
    subj_slot = None
    for p in tool.params:
        if not p.required:
            continue
        sl = _synth_param(p, rng, today)
        if p.kind in _PRIMARY_KINDS and subj_slot is None:
            subj_slot = (p, sl); args[p.name] = sl.canonical
        elif p.kind == "date_range":
            ln = _range_fields(tool, p); cs, ce = sl.canonical
            args[ln[0]] = cs; args[ln[1]] = ce; must.append((p, sl))
        else:
            args[p.name] = sl.canonical; must.append((p, sl))
    if not args:  # yalnız opsiyonel param'lı tool -> bir tanesini ekle
        opts = [p for p in tool.params if not p.required and p.kind not in ("date_range",)]
        if opts:
            p = rng.choice(opts)
            sl = _synth_param(p, rng, today)
            args[p.name] = sl.canonical
            if p.kind in _PRIMARY_KINDS:
                subj_slot = (p, sl)
            else:
                must.append((p, sl))
    syn = rng.choice(tool.syn) if tool.syn else tool.obj_nom
    kwtok = rng.choice(tool.kw)
    subj = ""
    if subj_slot:
        p, sl = subj_slot
        subj = (rng.choice(F.SUBJ_EMP).format(emp=sl.surface) if p.kind != "name"
                else rng.choice(F.SUBJ_NAME).format(name=sl.surface))
    detail = (" — " + ", ".join(_human_phrase(p, s) for p, s in must)) if must else ""
    frame = rng.choice([
        "{subj}{kw} konusu: {syn}{detail}",
        "{subj}{syn}{detail} ({kw} tarafında)",
        "{subj}{kw} derken tam olarak şunu istiyorum: {syn}{detail}",
    ])
    u = frame.format(subj=subj, kw=kwtok, syn=syn, detail=detail)
    u = _cap(u)
    u, reg = F.style(rng, u)
    _need = ([(subj_slot[0].human, subj_slot[1])] if subj_slot else []) + [(p.human, s) for p, s in must]
    u = _ensure_grounded(u, _need)
    # aday listede kardeşler MUTLAKA olsun
    sib_pick = rng.sample(sibs, min(len(sibs), rng.randint(1, 3)))
    tools, names = idx.index.candidate_list(rng, [tool.name])
    have = {t["name"] for t in tools}
    for s in sib_pick:
        if s not in have:
            tools.append(idx.index.by_name[s].schema()); names.append(s)
    rng.shuffle(tools)
    return Record(tools, [
        {"role": "user", "content": u},
        {"role": "assistant", "content": tc_block(tool.name, args)},
    ], {
        "decision": "tool_call", "scenario": "hn_keyword_ambiguous", "target_tools": [tool.name],
        "domain": tool.domain, "category": tool.cat, "register": reg, "turns": 2,
        "candidate_count": len(names), "has_tool_result": False, "missing_params": [],
        "is_write": tool.cat == "write", "keyword_in_prompt": True,
        "hard_negative": "A_keyword_ambiguous", "siblings_in_list": sib_pick,
    })


def gen_hn_conflict(rng, idx, tool):
    """E: çelişkili parametre -> request_for_info (netleştirme)."""
    today = idx.today
    # çelişki türü: tarih aralığı ters, ya da iki farklı id
    dr = next((p for p in tool.params if p.kind == "date_range"), None)
    subj_slot = None
    must = []
    for p in tool.params:
        if p.required and p.kind in _PRIMARY_KINDS and subj_slot is None:
            sl = _synth_param(p, rng, today); subj_slot = (p, sl)
    if dr:
        start = today.replace()
        s1 = S.gen_future_date(rng, today, 40, 80)
        s2 = S.gen_future_date(rng, today, 2, 20)  # bitiş başlangıçtan ÖNCE
        conflict = f"başlangıç {s1.surface}, bitiş {s2.surface} (bitiş başlangıçtan önce)"
    else:
        idp = next((p for p in tool.params if p.required and p.kind in ("id", "emp_id")), None)
        if not idp:
            return None
        a, b = S.gen_id(rng, idp.prefix), S.gen_id(rng, idp.prefix)
        conflict = f"bir yerde {a.canonical}, başka yerde {b.canonical} yazıyor"
    subj = ""
    if subj_slot:
        p, sl = subj_slot
        subj = (rng.choice(F.SUBJ_EMP).format(emp=sl.surface) if p.kind != "name"
                else rng.choice(F.SUBJ_NAME).format(name=sl.surface))
    u = f"{subj}{tool.obj} {rng.choice(tool.verbs)}; {conflict}"
    u = _cap(u)
    u, reg = F.style(rng, u)
    ask = rng.choice(F.CONFLICT_ASK).format(
        conflict=conflict, conflict_cap=_cap(conflict))
    tools, names = idx.index.candidate_list(rng, [tool.name])
    return Record(tools, [
        {"role": "user", "content": u}, {"role": "assistant", "content": ask},
    ], {
        "decision": "request_for_info", "scenario": "hn_conflict", "target_tools": [tool.name],
        "domain": tool.domain, "category": tool.cat, "register": reg, "turns": 2,
        "candidate_count": len(names), "has_tool_result": False, "missing_params": [],
        "is_write": tool.cat == "write", "hard_negative": "E_conflict",
    })


def gen_hn_tool_absent(rng, idx, tool):
    """F: doğru tool aday listede YOK -> cannot_answer."""
    today = idx.today
    must = []
    subj_slot = None
    for p in tool.params:
        if not p.required:
            continue
        sl = _synth_param(p, rng, today)
        if p.kind in _PRIMARY_KINDS and subj_slot is None:
            subj_slot = (p, sl)
        elif p.kind == "date_range":
            must.append((p, sl))
        else:
            must.append((p, sl))
    u, reg, kw = _fill_user(rng, tool, must, subj_slot, oblique_p=0.5)
    tools, names = idx.index.candidate_list(rng, [tool.name], exclude_targets=True)
    doms = "İK, finans, CRM, envanter, satış, lojistik, takvim, destek"
    a = rng.choice(F.CANNOT["no_tool"]).format(domains=doms) + rng.choice(F.CANNOT_REDIRECT)
    return Record(tools, [
        {"role": "user", "content": u}, {"role": "assistant", "content": a.strip()},
    ], {
        "decision": "cannot_answer", "scenario": "hn_tool_absent", "target_tools": [],
        "would_need_tool": tool.name, "domain": tool.domain, "category": None, "register": reg,
        "turns": 2, "candidate_count": len(names), "has_tool_result": False,
        "missing_params": [], "is_write": False, "hard_negative": "F_tool_absent",
    })


def gen_hn_user_names_wrong_tool(rng, idx, tool):
    """D: kullanıcı yanlış tool'un yüzeyini söyler ama aslında bunu ister."""
    sibs = idx.index.keyword_siblings(tool.name) or idx.index.ranked_distractors(tool.name)[:4]
    wrong = idx.index.by_name[rng.choice(sibs)]
    today = idx.today
    args = {}
    must = []
    subj_slot = None
    for p in tool.params:
        if not p.required:
            continue
        sl = _synth_param(p, rng, today)
        if p.kind in _PRIMARY_KINDS and subj_slot is None:
            subj_slot = (p, sl); args[p.name] = sl.canonical
        elif p.kind == "date_range":
            ln = _range_fields(tool, p); cs, ce = sl.canonical
            args[ln[0]] = cs; args[ln[1]] = ce; must.append((p, sl))
        else:
            args[p.name] = sl.canonical; must.append((p, sl))
    subj = ""
    if subj_slot:
        p, sl = subj_slot
        subj = (rng.choice(F.SUBJ_EMP).format(emp=sl.surface) if p.kind != "name"
                else rng.choice(F.SUBJ_NAME).format(name=sl.surface))
    syn = rng.choice(tool.syn) if tool.syn else tool.obj_nom
    detail = ("; " + ", ".join(_human_phrase(p, s) for p, s in must)) if must else ""
    u = f"{subj}{wrong.obj_nom} gibi düşün ama tam o değil — aslında {syn}{detail}"
    u = _cap(u)
    u, reg = F.style(rng, u)
    _need = ([(subj_slot[0].human, subj_slot[1])] if subj_slot else []) + [(p.human, s) for p, s in must]
    u = _ensure_grounded(u, _need)
    tools, names = idx.index.candidate_list(rng, [tool.name])
    if wrong.name not in names:
        tools.append(wrong.schema()); names.append(wrong.name); rng.shuffle(tools)
    return Record(tools, [
        {"role": "user", "content": u},
        {"role": "assistant", "content": tc_block(tool.name, args)},
    ], {
        "decision": "tool_call", "scenario": "hn_user_names_wrong_tool", "target_tools": [tool.name],
        "domain": tool.domain, "category": tool.cat, "register": reg, "turns": 2,
        "candidate_count": len(names), "has_tool_result": False, "missing_params": [],
        "is_write": tool.cat == "write", "hard_negative": "D_user_names_wrong_tool",
        "named_wrong_tool": wrong.name,
    })
