# -*- coding: utf-8 -*-
"""Prosedürel parametre / değer sentezi.

Her `kind` için: rastgele bir KANONİK değer üret + o değeri kullanıcı metnine
gömülebilecek bir veya daha çok YÜZEY biçiminde döndür. Sabit havuz yoktur;
değerler üretim anında sentezlenir (D-6). Kanonik <-> yüzey ilişkisi
`resolve.py` ile doğrulanabilir olacak şekilde kurulur.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from . import resolve as R


@dataclass
class Slot:
    canonical: object          # tool_call argümanına yazılacak değer
    surface: str               # kullanıcı metnine gömülecek ifade
    kind: str
    aux: dict = None           # ek yüzeyler / gösterim biçimleri


# --------------------------------------------------------------------------- #
#  Kimlikler
# --------------------------------------------------------------------------- #
_ID_STYLES = [
    lambda pre, n: f"{pre}-{n}",
    lambda pre, n: f"{pre}{n}",
    lambda pre, n: f"{pre.lower()}_{n}",
    lambda pre, n: f"#{n}",
    lambda pre, n: f"{n}",
]
_ID_REF_TR = {
    "EMP": ["{v} numaralı çalışan", "{v} numaralı personel", "sicil no {v}",
            "personel {v}", "çalışan {v}", "{v} kodlu personel", "{v}"],
    "CUS": ["{v} numaralı müşteri", "müşteri {v}", "{v} kodlu müşteri hesabı",
            "hesap {v}", "{v}"],
    "TCK": ["{v} numaralı talep", "talep {v}", "{v} kodlu kayıt", "{v}"],
    "*":   ["{v}", "{v} numaralı kayıt", "{v} kodlu kayıt", "kayıt {v}"],
}


def gen_id(rng, prefix="EMP", digits=4):
    n = rng.randint(10 ** (digits - 1), 10 ** digits - 1)
    canon = f"{prefix}-{n}"
    style = rng.choice(_ID_STYLES)
    raw = style(prefix, n)
    refs = _ID_REF_TR.get(prefix, _ID_REF_TR["*"])
    surf = rng.choice(refs).format(v=raw)
    return Slot(canon, surf, "id", {"num": str(n), "raw": raw, "prefix": prefix})


# --------------------------------------------------------------------------- #
#  Tarih / dönem / yıl
# --------------------------------------------------------------------------- #
def _fmt_date_surfaces(d: date, today: date, rng):
    """Bir tarihin çeşitli yüzey biçimleri (hepsi resolve_date ile çözülür)."""
    out = [
        d.isoformat(),
        f"{d.day:02d}/{d.month:02d}/{d.year}",
        f"{d.day:02d}.{d.month:02d}.{d.year}",
        f"{d.day} {R.TR_MONTHS[d.month]} {d.year}",
    ]
    if d.year == today.year:
        out.append(f"{d.day} {R.TR_MONTHS[d.month]}")
    delta = (d - today).days
    if delta == 1:
        out.append("yarın")
    elif delta == 2:
        out += ["öbür gün", "2 gün sonra"]
    elif 3 <= delta <= 21:
        out.append(f"{delta} gün sonra")
        if delta % 7 == 0:
            out.append(f"{delta // 7} hafta sonra")
    elif delta == 0:
        out.append("bugün")
    if 0 < delta <= 13:
        out.append(f"{'önümüzdeki' if delta > (6 - today.weekday()) else 'bu'} {R.TR_WEEKDAYS[d.weekday()]}")
    return list(dict.fromkeys(out))


def gen_date(rng, today: date, lo=-120, hi=200):
    d = today + timedelta(days=rng.randint(lo, hi))
    surfaces = _fmt_date_surfaces(d, today, rng)
    surf = rng.choice(surfaces)
    # doğrula: seçilen yüzey kanonike çözülmeli, değilse ISO'ya düş
    if R.resolve_date(surf, today) != d.isoformat():
        surf = d.isoformat()
    return Slot(d.isoformat(), surf, "date", {"iso": d.isoformat(), "surfaces": surfaces})


def gen_future_date(rng, today, lo=2, hi=180):
    return gen_date(rng, today, lo, hi)


def gen_past_date(rng, today, lo=-365, hi=-2):
    return gen_date(rng, today, lo, hi)


def gen_date_range(rng, today, min_days=1, max_days=25, direction=0):
    if direction > 0:
        start = today + timedelta(days=rng.randint(2, 150))
    elif direction < 0:
        start = today - timedelta(days=rng.randint(20, 320))
    else:
        start = today + timedelta(days=rng.randint(-200, 150))
    length = rng.randint(min_days, max_days)
    end = start + timedelta(days=length)
    s_iso, e_iso = start.isoformat(), end.isoformat()
    forms = [
        (f"{start.day} {R.TR_MONTHS[start.month]} {start.year} - {end.day} {R.TR_MONTHS[end.month]} {end.year}", (s_iso, e_iso)),
        (f"{start.day}-{end.day} {R.TR_MONTHS[end.month]} {end.year}" if start.month == end.month
         else f"{start.day} {R.TR_MONTHS[start.month]} - {end.day} {R.TR_MONTHS[end.month]} {end.year}", (s_iso, e_iso)),
        (f"{start.day:02d}/{start.month:02d}/{start.year} ile {end.day:02d}/{end.month:02d}/{end.year} arası", (s_iso, e_iso)),
        (f"{s_iso} / {e_iso}", (s_iso, e_iso)),
        (f"{start.day} {R.TR_MONTHS[start.month]} {start.year} başlangıçlı {length + 1} günlük", (s_iso, e_iso)),
    ]
    surf, (cs, ce) = rng.choice(forms)
    return Slot((cs, ce), surf, "date_range",
               {"start": cs, "end": ce, "length": length})


def gen_period(rng, today, lo=-14, hi=3):
    d = R.add_months(today.replace(day=1), rng.randint(lo, hi))
    canon = f"{d:%Y-%m}"
    forms = [canon, f"{d.month:02d}/{d.year}", f"{R.TR_MONTHS[d.month]} {d.year}",
             f"{d.year} {R.TR_MONTHS[d.month]}"]
    diff = (today.year - d.year) * 12 + (today.month - d.month)
    if diff == 0:
        forms.append("bu ay")
    elif diff == 1:
        forms.append("geçen ay")
    elif diff == 2:
        forms.append("iki ay önce")
    surf = rng.choice(forms)
    if R.resolve_period(surf, today) != canon:
        surf = canon
    return Slot(canon, surf, "period", {"canon": canon})


def gen_year(rng, today, lo=-3, hi=1):
    y = today.year + rng.randint(lo, hi)
    forms = [str(y)]
    if y == today.year:
        forms.append("bu yıl")
    elif y == today.year - 1:
        forms.append("geçen yıl")
    elif y == today.year + 1:
        forms.append("gelecek yıl")
    surf = rng.choice(forms)
    if R.resolve_year(surf, today) != str(y):
        surf = str(y)
    return Slot(str(y), surf, "year", {})


# --------------------------------------------------------------------------- #
#  Sayısal
# --------------------------------------------------------------------------- #
def _num_surfaces(n, unit_tr, rng):
    """Küçük sayılarda çıplak rakam KULLANILMAZ (uzun ID'lerin içinde substring
    eşleşmesine yol açar); birim daima eklenir. Yalnız 'TL' (büyük) çıplak kalabilir."""
    grp = f"{n:,}".replace(",", ".")
    if unit_tr == "TL":
        out = [str(n), f"{grp} TL", f"{n} lira", f"₺{grp}", f"{n} TL"]
        if n % 1000 == 0:
            out.append(f"{n // 1000} bin TL")
        return out
    if unit_tr == "adet":
        return [f"{n} adet", f"{n} birim", f"{n} tane"]
    if unit_tr == "%":
        return [f"%{n}", f"yüzde {n}", f"{n}%"]
    if unit_tr == "saat":
        return [f"{n} saat", f"{n} sa"]
    if unit_tr == "gün":
        return [f"{n} gün", f"{n} günlük", f"{n} iş günü"]
    return [str(n)]


def gen_amount(rng, lo=1500, hi=250000, step=500):
    n = rng.randrange(lo, hi, step)
    surfs = _num_surfaces(n, "TL", rng)
    return Slot(n, rng.choice(surfs), "amount", {"surfaces": surfs,
                "disp": f"{n:,}".replace(",", ".") + " TL"})


def gen_count(rng, lo=1, hi=40):
    n = rng.randint(lo, hi)
    surfs = _num_surfaces(n, "adet", rng)
    return Slot(n, rng.choice(surfs), "count", {"surfaces": surfs, "disp": f"{n} adet"})


def gen_hours(rng, lo=1, hi=60):
    n = rng.randint(lo, hi)
    return Slot(n, rng.choice(_num_surfaces(n, "saat", rng)), "hours", {"disp": f"{n} saat"})


def gen_pct(rng, lo=3, hi=45):
    n = rng.randint(lo, hi)
    return Slot(n, rng.choice(_num_surfaces(n, "%", rng)), "pct", {"disp": f"%{n}"})


def gen_duration_days(rng, lo=1, hi=20):
    n = rng.randint(lo, hi)
    return Slot(n, rng.choice(_num_surfaces(n, "gün", rng)), "duration", {"disp": f"{n} gün"})


def gen_minutes(rng):
    n = rng.choice([15, 30, 45, 60, 90, 120])
    surfs = {15: ["15 dk", "15 dakika", "çeyrek saat"], 30: ["30 dk", "30 dakika", "yarım saat"],
             45: ["45 dk", "45 dakika"], 60: ["1 saat", "60 dakika", "bir saat"],
             90: ["1,5 saat", "90 dakika"], 120: ["2 saat", "120 dakika"]}[n]
    return Slot(n, rng.choice(surfs), "minutes", {"disp": f"{n} dakika"})


# --------------------------------------------------------------------------- #
#  Enum
# --------------------------------------------------------------------------- #
def gen_enum(rng, enum, surface_map=None):
    canon = rng.choice(list(enum))
    if surface_map and canon in surface_map and surface_map[canon]:
        surf = rng.choice(surface_map[canon])
    else:
        surf = canon
    return Slot(canon, surf, "enum", {"enum": list(enum)})


# --------------------------------------------------------------------------- #
#  Serbest metin havuzları  (kanonik = yüzey; büyük ve çeşitli)
# --------------------------------------------------------------------------- #
FIRST_NAMES = ("Ahmet Mehmet Ayşe Elif Can Deniz Zeynep Burak Seda Emre Merve Kaan "
               "Selin Onur Gizem Barış Ece Tolga Pınar Serkan Büşra Yusuf Nazlı Cem "
               "Hakan İrem Kerem Aslı Murat Derya Sinan Ceren Volkan Melis Uğur Nil "
               "Efe Duru Arda Yağmur Baran Ada Mert Lara Kuzey Zehra").split()
LAST_NAMES = ("Yılmaz Demir Şahin Çelik Yıldız Yıldırım Öztürk Aydın Özdemir Arslan "
              "Doğan Kılıç Aslan Çetin Kara Koç Kurt Özkan Şimşek Erdoğan Aksoy Polat").split()

REASONS = (
    "piyasa koşullarına uyum", "yıllık performans değerlendirmesi", "yeni sorumluluklar",
    "terfi", "enflasyon revizyonu", "ek proje yükü", "ekip liderliğine geçiş",
    "sertifikasyon sonrası kademe", "elde tutma", "görev tanımının genişlemesi",
    "acil aile durumu", "sağlık nedeni", "taşınma", "eğitim programı", "evlilik",
    "yurt dışı görevlendirme", "müşteri şikayeti", "sistem arızası", "veri hatası",
    "yanlış giriş düzeltmesi", "bütçe kısıtı", "tedarikçi gecikmesi", "stok fazlası",
    "kampanya hazırlığı", "yıl sonu kapanışı", "denetim talebi", "sözleşme yenileme",
)
TITLES = (
    "Kıdemli Yazılım Mühendisi", "Takım Lideri", "Ürün Yöneticisi", "Veri Analisti",
    "Proje Yöneticisi", "Müşteri Başarı Uzmanı", "Satış Temsilcisi", "İş Analisti",
    "Kıdemli Muhasebe Uzmanı", "Operasyon Sorumlusu", "Bölge Müdürü", "Teknik Lider",
    "İK İş Ortağı", "Finansal Kontrolör", "Lojistik Koordinatörü", "Depo Sorumlusu",
)
EVENT_TITLES = (
    "sprint planlama", "müşteri sunumu", "bütçe toplantısı", "birebir görüşme",
    "işe alım mülakatı", "tedarikçi görüşmesi", "ekip retrospektifi", "eğitim oturumu",
    "yönetim kurulu", "proje kickoff", "performans değerlendirmesi",
)
DOC_TITLES = (
    "2027 bütçe planı", "Q1 satış raporu", "tedarik sözleşmesi", "gizlilik politikası",
    "işe alım prosedürü", "envanter sayım tutanağı", "müşteri memnuniyet anketi",
    "proje kapanış raporu", "denetim bulguları", "seyahat politikası",
)
CITIES = ("İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Kocaeli", "Konya", "Adana",
          "Gaziantep", "Mersin", "Samsun", "Eskişehir")
STREETS = ("Çınar", "Lale", "Menekşe", "Gül", "Zambak", "Papatya", "Yasemin", "Manolya")
ORG_UNITS = ("Satış", "Pazarlama", "Yazılım Geliştirme", "Bilgi Teknolojileri", "İnsan Kaynakları",
             "Muhasebe", "Finans", "Operasyon", "Lojistik", "Satın Alma", "Kalite Güvence",
             "Müşteri Destek", "Ar-Ge", "Hukuk", "İç Denetim", "Üretim", "Depo", "Saha Ekibi")
PLACES = tuple(list(CITIES) + [f"{c} Deposu" for c in ("Gebze", "Hadımköy", "Kemalpaşa", "İkitelli", "Esenyurt")]
               + ["Merkez Depo", "Ana Antrepo", "Bölge Deposu"])
APPS = ("SAP ERP", "Salesforce CRM", "Jira", "Confluence", "Office 365", "GitHub",
        "Tableau", "SharePoint", "Zendesk", "internal-portal", "Power BI", "SuccessFactors")
QUERY_TOPICS = ("bütçe onay akışı", "izin politikası", "tedarikçi listesi", "fiyat güncellemesi",
                "sözleşme şablonu", "seyahat kuralları", "performans dönemi", "stok sayımı",
                "müşteri şikayeti", "güvenlik prosedürü", "yıl sonu kapanış", "işe alım süreci")


def gen_name(rng, full=None):
    if full is None:
        full = rng.random() < 0.72
    fn = rng.choice(FIRST_NAMES)
    if not full:
        surf = rng.choice([fn, f"{fn} Bey", f"{fn} Hanım"])
        return Slot(fn, surf, "name", {})
    name = f"{fn} {rng.choice(LAST_NAMES)}"
    return Slot(name, name, "name", {})


def gen_reason(rng):
    v = rng.choice(REASONS)
    return Slot(v, v, "text", {})


def gen_title(rng, pool=TITLES):
    v = rng.choice(pool)
    return Slot(v, v, "text", {})


def _tr_lower(s):
    # 'İ'.lower() Python'da 'i̇' üretir; onu kullanmadan Türkçe küçült.
    return s.replace("İ", "i").replace("I", "ı").lower()


def gen_org_name(rng):
    v = rng.choice(ORG_UNITS)
    surf = rng.choice([v, f"{v} birimi", f"{v} ekibi", f"{v} departmanı", _tr_lower(v)])
    return Slot(v, surf, "org_name", {})


def gen_place(rng):
    v = rng.choice(PLACES)
    return Slot(v, v, "place", {})


def gen_app(rng):
    v = rng.choice(APPS)
    return Slot(v, v, "app_name", {})


def gen_weight(rng, lo=5, hi=1200):
    n = rng.randrange(lo, hi, 5)
    surfs = [str(n), f"{n} kg", f"{n} kilo", f"{n}kg"]
    return Slot(n, rng.choice(surfs), "weight", {"surfaces": surfs})


def gen_query(rng):
    if rng.random() < 0.4:
        nm = gen_name(rng, full=True)
        return Slot(nm.canonical, nm.surface, "query", {})
    v = rng.choice(QUERY_TOPICS)
    return Slot(v, v, "query", {})


def gen_email(rng):
    fn = _tr_lower(rng.choice(FIRST_NAMES))
    ln = _tr_lower(rng.choice(LAST_NAMES))
    fold = str.maketrans("çğıöşü", "cgiosu")
    dom = rng.choice(["ornek.com", "sirket.com.tr", "firma.io", "mail.com"])
    v = f"{fn}.{ln}".translate(fold) + f"@{dom}"
    return Slot(v, v, "email", {})


def gen_phone(rng):
    v = f"0{rng.choice(['532','541','555','505','533','542'])} {rng.randint(100,999)} {rng.randint(10,99)} {rng.randint(10,99)}"
    return Slot(v, v, "phone", {})


def gen_address(rng):
    v = (f"{rng.choice(STREETS)} Mah. {rng.randint(1,120)}. Sk. No:{rng.randint(1,60)} "
         f"D:{rng.randint(1,20)}, {rng.choice(CITIES)}")
    return Slot(v, v, "address", {})


# --------------------------------------------------------------------------- #
#  Dispatch
# --------------------------------------------------------------------------- #
def synth(kind: str, rng, today: date, *, enum=None, surface_map=None,
          id_prefix="EMP", id_digits=4, title_pool=None, direction=0):
    if kind == "emp_id":
        return gen_id(rng, "EMP", 4)
    if kind == "id":
        return gen_id(rng, id_prefix, id_digits)
    if kind == "date":
        return gen_date(rng, today)
    if kind == "future_date":
        return gen_future_date(rng, today)
    if kind == "past_date":
        return gen_past_date(rng, today)
    if kind == "date_range":
        return gen_date_range(rng, today, direction=direction)
    if kind == "period":
        return gen_period(rng, today)
    if kind == "year":
        return gen_year(rng, today)
    if kind == "amount":
        return gen_amount(rng)
    if kind == "count":
        return gen_count(rng)
    if kind == "hours":
        return gen_hours(rng)
    if kind == "pct":
        return gen_pct(rng)
    if kind == "duration":
        return gen_duration_days(rng)
    if kind == "minutes":
        return gen_minutes(rng)
    if kind == "enum":
        return gen_enum(rng, enum, surface_map)
    if kind == "name":
        return gen_name(rng)
    if kind == "reason":
        return gen_reason(rng)
    if kind == "title":
        return gen_title(rng, title_pool or TITLES)
    if kind == "org_name":
        return gen_org_name(rng)
    if kind == "event_title":
        v = rng.choice(EVENT_TITLES)
        return Slot(v, rng.choice([v, f"{v} toplantısı", f"{v} görüşmesi"]), "event_title", {})
    if kind == "doc_title":
        v = rng.choice(DOC_TITLES)
        return Slot(v, v, "doc_title", {})
    if kind == "place":
        return gen_place(rng)
    if kind == "app_name":
        return gen_app(rng)
    if kind == "weight":
        return gen_weight(rng)
    if kind == "query":
        return gen_query(rng)
    if kind == "email":
        return gen_email(rng)
    if kind == "phone":
        return gen_phone(rng)
    if kind == "address":
        return gen_address(rng)
    raise ValueError(f"bilinmeyen kind: {kind}")
