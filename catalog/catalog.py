# -*- coding: utf-8 -*-
"""~100 tool'luk şema kataloğu — 13 domain.

Bu dosya, veri üretiminin TEK kaynağıdır. Yeni bir tool eklemek = bu listeye bir
`T(...)` satırı eklemek. Üretici (scripts/gen/scenarios.py) örnekleri buradaki
şemadan türetir; per-tool cümle şablonu YOKTUR.

Tool alanları:
    name      : benzersiz ad (İngilizce snake_case, domain öneki)
    domain    : 13 alandan biri
    cat       : "read" | "write" | "action"
    desc      : şema açıklaması (tools[] içinde modele verilir) — 1 cümle TR
    obj       : nesne öbeği, "-i hâli" ("{obj} {verb}" -> "faturayı getir")
    obj_nom   : yalın hâl ("{obj_nom} nedir")
    kw        : bu tool'un yüzey sözlüğünü tanımlayan kökler (hard-negative
                gruplaması + anti-korelasyon denetimi için)
    verbs     : eylem fiilleri
    params    : Param listesi
    result    : (anahtar, kind) listesi — tool-sonucu turu için sentez şeması
    syn       : baş sözcüğü İÇERMEYEN dolaylı ifadeler (anti-kısayol, D-4)
                — tam cümlecik; frame bunları tek başına kullanır
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class Param:
    name: str
    kind: str
    required: bool = True
    desc: str = ""
    human: str = ""            # request_for_info sorusunda kullanılacak insan ifadesi
    enum: tuple | None = None
    smap: dict | None = None   # enum kanonik -> [yüzey eşanlamlıları]
    prefix: str = "EMP"        # kind == "id" için
    digits: int = 4

    def schema(self) -> dict:
        d = {"type": "number" if self.kind in ("amount", "count", "hours", "pct", "duration") else "string",
             "description": self.desc}
        if self.enum:
            d["enum"] = list(self.enum)
        return d


@dataclass
class Tool:
    name: str
    domain: str
    cat: str
    desc: str
    obj: str
    obj_nom: str
    kw: tuple
    verbs: tuple
    params: list
    result: list = field(default_factory=list)
    syn: tuple = ()
    split: str = "train"
    disc_kw: tuple = ()

    # --- türetilmiş ---
    @property
    def required(self):
        return [p.name for p in self.params if p.required]

    @property
    def optional(self):
        return [p.name for p in self.params if not p.required]

    def param(self, name):
        for p in self.params:
            if p.name == name:
                return p
        return None

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.desc,
            "parameters": {
                "type": "object",
                "properties": {p.name: p.schema() for p in self.params},
                "required": self.required,
            },
        }

    def param_kinds(self) -> frozenset:
        return frozenset(p.kind for p in self.params)


# --------------------------------------------------------------------------- #
#  Kısa yapıcılar
# --------------------------------------------------------------------------- #
def P(name, kind, req=True, desc="", human="", enum=None, smap=None, prefix="EMP", digits=4):
    return Param(name, kind, req, desc, human or _default_human(name, kind), enum, smap, prefix, digits)


def _default_human(name, kind):
    table = {
        "emp_id": "personel numarası", "date": "tarih", "future_date": "tarih",
        "past_date": "tarih", "date_range": "tarih aralığı", "period": "dönem (ay)",
        "year": "yıl", "amount": "tutar", "count": "adet", "hours": "saat",
        "pct": "oran", "duration": "süre", "name": "ad", "reason": "gerekçe",
        "title": "unvan/başlık", "email": "e-posta", "phone": "telefon",
        "address": "adres", "enum": "seçenek", "org_name": "birim adı",
        "place": "konum", "app_name": "uygulama adı", "weight": "ağırlık (kg)",
        "query": "aranacak ifade", "event_title": "etkinlik başlığı", "doc_title": "belge başlığı",
        "minutes": "süre (dakika)", "weight_kg": "ağırlık",
    }
    return table.get(kind, name.replace("_", " "))


# kw alanından ayıklanacak jenerik eylem/soru sözcükleri — bunlar "yüzey sözlüğü"
# sayılmaz; korelasyon ölçümünü ve hard-negative gruplamasını kirletirler.
_KW_STOP = {
    "liste", "getir", "göster", "goster", "kontrol", "çıkar", "cikar", "bul", "ara", "aç", "ac",
    "oluştur", "olustur", "gir", "başlat", "baslat", "kaydet", "güncelle", "guncelle",
    "değiştir", "degistir", "düzelt", "duzelt", "ver", "kaldır", "kaldir", "sil", "planla",
    "ayarla", "gönder", "gonder", "ilet", "ekle", "yeni", "özetle", "ozetle", "hesapla",
    "sor", "bak", "dök", "dok", "topla", "say", "yenile", "kim", "kimler", "hangi", "kayıt",
    "kayit", "at", "tut", "geri", "vazgeç", "vazgec",
}


def T(name, domain, cat, desc, obj, obj_nom, kw, verbs, params, result, syn):
    kw = tuple(k for k in kw if k.lower() not in _KW_STOP)
    return Tool(name, domain, cat, desc, obj, obj_nom, kw, tuple(verbs),
                list(params), list(result), tuple(syn))


# yaygın parametreler
EMP = lambda req=True: P("employee_id", "emp_id", req, "Personel kimliği, 'EMP-1234' biçiminde", "personel numarası")
def ID(name, prefix, human, req=True, desc=None):
    return P(name, "id", req, desc or f"{human.capitalize()} kimliği, '{prefix}-1234' biçiminde", human, prefix=prefix)


# --------------------------------------------------------------------------- #
#  KATALOG
# --------------------------------------------------------------------------- #
TOOLS: list[Tool] = []
def _add(*ts): TOOLS.extend(ts)


# ---- 1. hr / İnsan Kaynakları --------------------------------------------- #
_LEAVE_ENUM = ("annual", "excuse", "sick", "unpaid")
_LEAVE_SMAP = {"annual": ["yıllık", "senelik", "yıllık ücretli"], "excuse": ["mazeret", "idari"],
               "sick": ["hastalık", "rapor", "sağlık", "istirahat"], "unpaid": ["ücretsiz"]}
_add(
    T("hr_get_employee_profile", "hr", "read",
      "Bir çalışanın temel özlük kaydını (ad, birim, unvan, işe giriş tarihi, yönetici) getirir.",
      "özlük kaydını", "özlük kaydı", ["özlük", "personel", "çalışan", "sicil"],
      ["getir", "göster", "aç", "çıkar"],
      [EMP()],
      [("full_name", "name"), ("unit", "title"), ("title", "title"), ("hire_date", "past_date")],
      ["bu kişi kim, ne iş yapıyor", "hangi birimde ve ne zamandır çalışıyor", "kayıtta ne yazıyor"]),
    T("hr_get_employment_status", "hr", "read",
      "Bir çalışanın güncel çalışma durumunu (aktif, izinli, ücretsiz izinde, ayrıldı) döndürür.",
      "çalışma durumunu", "çalışma durumu", ["durum", "aktif", "izinli", "ayrıldı"],
      ["kontrol et", "söyle", "getir", "bak"],
      [EMP()],
      [("status", "enum")],
      ["hâlâ şirkette mi", "şu an işbaşında mı yoksa izinde mi", "bu kişi ayrıldı mı"]),
    T("hr_get_manager", "hr", "read",
      "Bir çalışanın bağlı olduğu yöneticinin bilgisini getirir.",
      "bağlı olduğu yöneticiyi", "yönetici", ["yönetici", "amir", "müdür", "bağlı"],
      ["söyle", "getir", "göster", "bul"],
      [EMP()],
      [("manager_name", "name"), ("manager_title", "title")],
      ["bu kişi kime rapor veriyor", "kimin ekibinde", "üstü kim"]),
    T("hr_get_leave_balance", "hr", "read",
      "Bir çalışanın izin türüne göre kalan izin bakiyesini getirir.",
      "izin bakiyesini", "izin bakiyesi", ["izin", "bakiye", "hak"],
      ["göster", "getir", "kontrol et", "hesapla"],
      [EMP(), P("leave_type", "enum", False, "İzin türü", "izin türü", _LEAVE_ENUM, _LEAVE_SMAP)],
      [("annual_left", "count"), ("excuse_left", "count"), ("sick_left", "count")],
      ["kaç günüm kaldı", "daha ne kadar dinlenme hakkım var", "bu yıl ne kadar kullanabilirim daha",
       "yıllık hakkımdan geriye ne kaldı"]),
    T("hr_get_leave_history", "hr", "read",
      "Bir çalışanın geçmişte kullandığı izin kayıtlarını tarih aralığına göre listeler.",
      "izin geçmişini", "izin geçmişi", ["izin", "geçmiş", "kullandığı"],
      ["listele", "çıkar", "getir", "dök"],
      [EMP(), P("start_date", "past_date", False, "Aralık başlangıcı, YYYY-AA-GG"),
       P("end_date", "past_date", False, "Aralık bitişi, YYYY-AA-GG")],
      [("records_count", "count"), ("total_days", "count")],
      ["daha önce ne zaman izne çıktım", "hangi tarihlerde tatil yaptım", "geçen dönem izin kullanımım nasıldı"]),
    T("hr_get_leave_request_status", "hr", "read",
      "Bir çalışanın izin talebinin onay durumunu (beklemede, onaylandı, reddedildi) getirir.",
      "talebin durumunu", "talep durumu", ["talep", "onay", "başvuru", "durum"],
      ["kontrol et", "sor", "bak", "getir"],
      [EMP(), P("request_id", "id", False, "Talep kimliği, 'LR-1234' biçiminde", "talep numarası", prefix="LR")],
      [("state", "enum"), ("decided_by", "name")],
      ["başvurum kabul edildi mi", "onay geldi mi", "hâlâ bekliyor mu"]),
    T("hr_create_leave_request", "hr", "write",
      "Bir çalışan için yeni izin talebi oluşturur. Onay gerektirir.",
      "izin talebini", "izin talebi", ["izin", "talep", "başvuru"],
      ["oluştur", "aç", "gir", "başlat", "kaydet"],
      [EMP(), P("leave_type", "enum", True, "İzin türü", "izin türü", _LEAVE_ENUM, _LEAVE_SMAP),
       P("start_date", "future_date", True, "Başlangıç, YYYY-AA-GG"),
       P("end_date", "future_date", True, "Bitiş, YYYY-AA-GG"),
       P("note", "reason", False, "İsteğe bağlı açıklama")],
      [("request_id", "id"), ("state", "enum")],
      ["birkaç gün dinlenmek istiyorum, kaydını aç", "tatile çıkacağım, gerekeni yap",
       "işe gelemeyeceğim günler için başvuru gir"]),
    T("hr_cancel_leave_request", "hr", "write",
      "Mevcut bir izin talebini iptal eder. Onay gerektirir.",
      "izin talebini", "izin talebi iptali", ["iptal", "talep", "geri", "vazgeç"],
      ["iptal et", "geri çek", "kaldır", "sil"],
      [P("request_id", "id", True, "İptal edilecek talebin kimliği", "talep numarası", prefix="LR")],
      [("state", "enum")],
      ["girdiğim başvurudan vazgeçtim", "o kaydı geri al", "artık o günleri istemiyorum"]),
    T("hr_update_leave_request", "hr", "write",
      "Mevcut bir izin talebinin tarihlerini günceller. Onay gerektirir.",
      "izin talebinin tarihlerini", "talep güncelleme", ["güncelle", "tarih", "değiştir", "kaydır"],
      ["güncelle", "değiştir", "revize et", "kaydır"],
      [P("request_id", "id", True, "Talep kimliği", "talep numarası", prefix="LR"),
       P("new_start_date", "future_date", True, "Yeni başlangıç"),
       P("new_end_date", "future_date", True, "Yeni bitiş")],
      [("state", "enum")],
      ["o günleri başka tarihe alalım", "planım değişti, kaydı yeni tarihlere çek"]),
    T("hr_get_org_unit", "hr", "read",
      "Bir organizasyon biriminin özetini (yönetici, kişi sayısı, alt ekipler) getirir.",
      "birim özetini", "birim bilgisi", ["birim", "departman", "ekip", "bölüm"],
      ["getir", "göster", "özetle", "aç"],
      [P("unit_name", "org_name", True, "Birim adı, örn. 'Satış'", "birim adı")],
      [("manager", "name"), ("headcount", "count"), ("sub_teams", "count")],
      ["burada kaç kişi çalışıyor", "bu bölümün başında kim var", "kaç alt ekibi var"]),
    T("hr_list_unit_members", "hr", "read",
      "Bir birimdeki çalışanların listesini duruma göre filtreleyerek getirir.",
      "birim çalışan listesini", "çalışan listesi", ["liste", "kimler", "çalışanlar", "ekip"],
      ["listele", "çıkar", "getir", "göster"],
      [P("unit_name", "org_name", True, "Birim adı", "birim adı"),
       P("status", "enum", False, "Durum filtresi", "durum", ("active", "on_leave", "left"),
         {"active": ["aktif", "çalışan"], "on_leave": ["izinli", "izinde"], "left": ["ayrılan", "ayrılmış"]})],
      [("count", "count")],
      ["bu ekipte kimler var", "orada çalışanların adlarını ver", "kim kim orada"]),
)

# ---- 2. payroll / Bordro & Ücret ---------------------------------------- #
_add(
    T("payroll_get_salary", "payroll", "read",
      "Bir çalışanın güncel maaş bilgisini (net / brüt) getirir.",
      "maaş bilgisini", "maaş", ["maaş", "ücret", "kazanç"],
      ["göster", "getir", "söyle", "bak"],
      [EMP(), P("basis", "enum", False, "Maaş türü", "maaş türü", ("net", "gross"),
                {"net": ["net", "elime geçen"], "gross": ["brüt", "brut"]})],
      [("amount", "amount"), ("currency", "title")],
      ["eline ne kadar geçiyor", "bu kişi ne kazanıyor", "aylık ödemesi nedir"]),
    T("payroll_get_payslip", "payroll", "read",
      "Bir çalışanın belirli bir aya ait bordro dökümünü getirir.",
      "bordroyu", "bordro", ["bordro", "pusula", "maaş dökümü"],
      ["getir", "aç", "göster", "çıkar"],
      [EMP(), P("period", "period", True, "Bordro dönemi, YYYY-AA")],
      [("gross", "amount"), ("net", "amount"), ("deductions", "amount")],
      ["o ayki ödeme dökümü lazım", "kesintilerle birlikte hesabı görmek istiyorum"]),
    T("payroll_get_bonus", "payroll", "read",
      "Bir çalışanın prim / ikramiye ödemelerini yıla göre getirir.",
      "prim bilgisini", "prim", ["prim", "ikramiye", "bonus"],
      ["listele", "getir", "göster", "dök"],
      [EMP(), P("year", "year", False, "Yıl")],
      [("total", "amount"), ("payments", "count")],
      ["bu yıl ne kadar ek ödeme aldım", "performans ödemem yattı mı"]),
    T("payroll_get_benefits", "payroll", "read",
      "Bir çalışanın yan haklarını (özel sağlık, yemek, ulaşım, BES) listeler.",
      "yan hakları", "yan haklar", ["yan hak", "menfaat", "sağlık", "yemek kartı"],
      ["listele", "göster", "getir", "çıkar"],
      [EMP()],
      [("items", "count")],
      ["bana ne gibi ek imkânlar tanımlı", "sağlık sigortam var mı, neler kapsıyor"]),
    T("payroll_get_tax_summary", "payroll", "read",
      "Bir çalışanın yıl içi kümülatif gelir vergisi matrahı ve kesinti özetini getirir.",
      "vergi özetini", "vergi özeti", ["vergi", "matrah", "kümülatif", "kesinti"],
      ["getir", "göster", "hesapla", "çıkar"],
      [EMP(), P("year", "year", False, "Yıl")],
      [("cumulative_base", "amount"), ("tax_paid", "amount"), ("bracket", "pct")],
      ["neden her ay elime geçen azalıyor", "yıl içi biriken kesintim ne durumda"]),
    T("payroll_create_salary_change_request", "payroll", "write",
      "Bir çalışan için ücret değişikliği talebi oluşturur. Hassas işlem; onay gerektirir.",
      "ücret değişikliği talebini", "ücret değişikliği", ["ücret", "zam", "maaş değişikliği", "revizyon"],
      ["oluştur", "aç", "başlat", "gir"],
      [EMP(), P("new_gross_amount", "amount", True, "Yeni brüt aylık ücret"),
       P("reason", "reason", True, "Gerekçe"),
       P("effective_date", "future_date", False, "Geçerlilik tarihi")],
      [("request_id", "id"), ("state", "enum")],
      ["bu kişinin ödemesini yukarı çekmek istiyorum", "maaşını artıralım, talebini aç"]),
    T("payroll_get_deduction_breakdown", "payroll", "read",
      "Bir bordronun kesinti kalemlerini (SGK, işsizlik, gelir vergisi, damga) ayrıntılı getirir.",
      "kesinti dökümünü", "kesinti dökümü", ["kesinti", "sgk", "damga", "stopaj"],
      ["çıkar", "getir", "ayrıştır", "göster"],
      [EMP(), P("period", "period", True, "Dönem, YYYY-AA")],
      [("sgk", "amount"), ("income_tax", "amount"), ("stamp", "amount")],
      ["brütten ne ne kesilmiş görmek istiyorum", "o ay hangi kalemler düşülmüş"]),
    T("payroll_get_payment_history", "payroll", "read",
      "Bir çalışana yapılan maaş ödemelerinin geçmişini tarih aralığına göre listeler.",
      "ödeme geçmişini", "ödeme geçmişi", ["ödeme", "geçmiş", "yatan", "havale"],
      ["listele", "getir", "dök", "çıkar"],
      [EMP(), P("start_date", "past_date", False, "Başlangıç"),
       P("end_date", "past_date", False, "Bitiş")],
      [("count", "count"), ("total", "amount")],
      ["hesabıma son aylarda neler yattı", "geçmiş ödemelerin listesi lazım"]),
    T("payroll_get_severance_estimate", "payroll", "read",
      "Bir çalışan için tahmini kıdem ve ihbar tutarını hesaplar.",
      "kıdem tahminini", "kıdem tahmini", ["kıdem", "ihbar", "tazminat", "ayrılık"],
      ["hesapla", "getir", "tahmin et", "çıkar"],
      [EMP(), P("termination_date", "future_date", False, "Çıkış tarihi")],
      [("severance", "amount"), ("notice", "amount"), ("years", "count")],
      ["ayrılırsam elime ne geçer", "bu kişinin çıkışında ne kadar ödenir"]),
)

# ---- 3. timesheet / Puantaj -------------------------------------------- #
_add(
    T("timesheet_get_records", "timesheet", "read",
      "Bir çalışanın tarih aralığındaki puantaj kaydını (çalışılan gün, giriş/çıkış, devamsızlık) getirir.",
      "puantaj kaydını", "puantaj", ["puantaj", "devam", "giriş çıkış", "mesai kaydı"],
      ["getir", "çıkar", "göster", "dök"],
      [EMP(), P("start_date", "date", True, "Başlangıç"), P("end_date", "date", True, "Bitiş")],
      [("worked_days", "count"), ("absences", "count")],
      ["o dönem kaç gün işbaşı yapmış", "giriş çıkışlarını görmek istiyorum"]),
    T("timesheet_get_overtime", "timesheet", "read",
      "Bir çalışanın belirli bir aya ait fazla mesai saatlerini ve karşılığını getirir.",
      "fazla mesai bilgisini", "fazla mesai", ["mesai", "fazla çalışma", "ek çalışma"],
      ["getir", "topla", "göster", "hesapla"],
      [EMP(), P("period", "period", True, "Dönem, YYYY-AA")],
      [("hours", "hours"), ("pay", "amount")],
      ["geçen ay normalin üstünde ne kadar çalıştım", "ek çalışma karşılığım ne oldu"]),
    T("timesheet_submit_time_correction", "timesheet", "write",
      "Bir çalışanın puantaj düzeltme talebini oluşturur. Onay gerektirir.",
      "puantaj düzeltme talebini", "puantaj düzeltmesi", ["düzeltme", "puantaj", "hatalı giriş"],
      ["oluştur", "gir", "aç", "başlat"],
      [EMP(), P("date", "past_date", True, "Düzeltilecek gün"),
       P("reason", "reason", True, "Düzeltme gerekçesi")],
      [("request_id", "id"), ("state", "enum")],
      ["o gün yanlış işlenmiş, düzeltilsin", "sistemde eksik görünüyorum, talep aç"]),
    T("timesheet_get_attendance_summary", "timesheet", "read",
      "Bir birim veya çalışan için dönemsel devam-devamsızlık özetini getirir.",
      "devam özetini", "devam özeti", ["devamsızlık", "geç kalma", "devam özeti"],
      ["özetle", "getir", "çıkar", "göster"],
      [P("scope_id", "id", True, "Çalışan veya birim kimliği", "kimlik", prefix="EMP"),
       P("period", "period", True, "Dönem")],
      [("late_count", "count"), ("absent_days", "count")],
      ["o ay kaç kez geç kalınmış", "devamsızlık tablosu lazım"]),
    T("timesheet_get_shift_schedule", "timesheet", "read",
      "Bir çalışanın belirli bir haftadaki vardiya planını getirir.",
      "vardiya planını", "vardiya planı", ["vardiya", "nöbet", "çalışma planı"],
      ["getir", "göster", "çıkar", "aç"],
      [EMP(), P("week_start", "date", True, "Hafta başlangıcı")],
      [("shifts", "count")],
      ["bu hafta hangi saatlerde çalışıyorum", "nöbet listem ne"]),
    T("timesheet_request_leave_of_absence", "timesheet", "write",
      "Uzun süreli ücretsiz devamsızlık (izinsiz ayrılış değil) talebi oluşturur. Onay gerektirir.",
      "uzun devamsızlık talebini", "uzun devamsızlık", ["ücretsiz", "uzun devamsızlık", "ara verme"],
      ["oluştur", "başlat", "aç", "gir"],
      [EMP(), P("start_date", "future_date", True, "Başlangıç"),
       P("duration_days", "duration", True, "Süre (gün)"), P("reason", "reason", True, "Gerekçe")],
      [("request_id", "id"), ("state", "enum")],
      ["bir süre işe ara vermem gerekiyor", "uzun süreli uzaklaşacağım, kaydını aç"]),
)

# ---- 4. finance / Finans --------------------------------------------- #
_EXP_ENUM = ("draft", "submitted", "approved", "rejected", "paid")
_add(
    T("finance_get_invoice", "finance", "read",
      "Bir faturanın ayrıntılarını (tutar, tarih, tedarikçi, durum) getirir.",
      "faturayı", "fatura", ["fatura", "irsaliye"],
      ["getir", "aç", "göster", "bul"],
      [ID("invoice_id", "INV", "fatura", True)],
      [("amount", "amount"), ("due_date", "future_date"), ("status", "enum")],
      ["şu belgede ne kadar yazıyor", "o kağıdın vadesi ne zaman"]),
    T("finance_list_invoices", "finance", "read",
      "Bir tedarikçi veya döneme ait faturaları listeler.",
      "faturaları", "fatura listesi", ["fatura", "liste", "tedarikçi"],
      ["listele", "getir", "çıkar", "dök"],
      [P("vendor_id", "id", False, "Tedarikçi kimliği", "tedarikçi", prefix="VEN"),
       P("period", "period", False, "Dönem")],
      [("count", "count"), ("total", "amount")],
      ["o firmadan gelen belgeleri sırala", "bu ay hangi ödemeler birikti"]),
    T("finance_create_expense_report", "finance", "write",
      "Bir çalışan için masraf beyanı oluşturur. Onay gerektirir.",
      "masraf beyanını", "masraf beyanı", ["masraf", "harcama", "gider", "beyan"],
      ["oluştur", "gir", "aç", "başlat"],
      [EMP(), P("amount", "amount", True, "Toplam tutar"),
       P("category", "enum", True, "Masraf türü", "masraf türü",
         ("travel", "meal", "supplies", "other"),
         {"travel": ["seyahat", "yol"], "meal": ["yemek"], "supplies": ["kırtasiye", "malzeme"], "other": ["diğer"]}),
       P("description", "reason", False, "Açıklama")],
      [("report_id", "id"), ("status", "enum")],
      ["cebimden yaptığım harcamayı geri almak istiyorum", "iş için ödediğim tutarı beyan et"]),
    T("finance_get_expense_status", "finance", "read",
      "Bir masraf beyanının onay ve ödeme durumunu getirir.",
      "masraf beyanının durumunu", "masraf durumu", ["masraf", "durum", "onay", "ödeme"],
      ["kontrol et", "sor", "bak", "getir"],
      [ID("report_id", "EXP", "masraf beyanı", True)],
      [("status", "enum"), ("paid_date", "date")],
      ["harcama iadem ne aşamada", "beyanım onaylandı mı"]),
    T("finance_approve_expense", "finance", "action",
      "Bir masraf beyanını onaylar. Yalnız yetkili yönetici çağırabilir; onay gerektirir.",
      "masraf beyanını", "masraf onayı", ["onayla", "masraf", "kabul"],
      ["onayla", "kabul et", "geçir"],
      [ID("report_id", "EXP", "masraf beyanı", True),
       P("approver_id", "emp_id", True, "Onaylayan yöneticinin personel kimliği", "onaylayan personel numarası")],
      [("status", "enum")],
      ["ekibimden gelen o beyanı geçir", "harcamayı kabul ediyorum, işleme al"]),
    T("finance_get_budget_status", "finance", "read",
      "Bir bütçe kaleminin kullanım ve kalan durumunu getirir.",
      "bütçe durumunu", "bütçe durumu", ["bütçe", "harcama limiti", "kalan"],
      ["getir", "göster", "kontrol et", "özetle"],
      [P("budget_code", "id", True, "Bütçe kodu", "bütçe kodu", prefix="BUD"),
       P("year", "year", False, "Yıl")],
      [("allocated", "amount"), ("spent", "amount"), ("remaining", "amount")],
      ["bu kalemde daha ne kadar param var", "harcayabileceğim limit doldu mu"]),
    T("finance_create_purchase_order", "finance", "write",
      "Bir tedarikçiye satın alma siparişi oluşturur. Onay gerektirir.",
      "satın alma siparişini", "satın alma siparişi", ["sipariş", "satın alma", "po", "tedarik"],
      ["oluştur", "aç", "gir", "başlat"],
      [P("vendor_id", "id", True, "Tedarikçi kimliği", "tedarikçi", prefix="VEN"),
       P("amount", "amount", True, "Sipariş tutarı"),
       P("description", "reason", True, "Kalem açıklaması")],
      [("po_id", "id"), ("status", "enum")],
      ["o firmadan alım yapmam lazım, kaydını aç", "tedarik talebini başlat"]),
    T("finance_get_vendor", "finance", "read",
      "Bir tedarikçinin kayıt bilgilerini (ad, vergi no, ödeme koşulu, iletişim) getirir.",
      "tedarikçi kaydını", "tedarikçi kaydı", ["tedarikçi", "satıcı", "firma"],
      ["getir", "aç", "göster", "bul"],
      [P("vendor_id", "id", True, "Tedarikçi kimliği", "tedarikçi", prefix="VEN")],
      [("name", "title"), ("payment_terms", "title")],
      ["o satıcı hakkında ne kayıtlıyız", "firmanın ödeme vadesi kaç gün"]),
    T("finance_get_reimbursement_policy_limit", "finance", "read",
      "Belirli bir masraf türü için günlük/işlem başına harcama üst sınırını getirir.",
      "harcama limitini", "harcama limiti", ["limit", "üst sınır", "politika"],
      ["getir", "söyle", "göster", "kontrol et"],
      [P("category", "enum", True, "Masraf türü", "masraf türü",
         ("travel", "meal", "supplies", "other"),
         {"travel": ["seyahat", "yol", "konaklama"], "meal": ["yemek", "öğün"], "supplies": ["malzeme"], "other": ["diğer"]})],
      [("daily_cap", "amount"), ("needs_receipt", "count")],
      ["yemekte günlük ne kadara kadar ödenir", "konaklama için tavan ne"]),
)

# ---- 5. crm / Müşteri İlişkileri ----------------------------------- #
_DEAL_STAGE = ("lead", "qualified", "proposal", "negotiation", "won", "lost")
_add(
    T("crm_get_contact", "crm", "read",
      "Bir kişi kaydının ayrıntılarını (ad, unvan, e-posta, telefon, bağlı hesap) getirir.",
      "kişi kaydını", "kişi kaydı", ["kişi", "kontak", "irtibat"],
      ["getir", "aç", "göster", "bul"],
      [ID("contact_id", "CNT", "kişi", True)],
      [("name", "name"), ("email", "email"), ("account", "title")],
      ["o kartta iletişim bilgisi ne", "bu kaydın detayları lazım"]),
    T("crm_search_contacts", "crm", "read",
      "İsim, e-posta veya şirkete göre kişi kayıtlarını arar.",
      "kişileri", "kişi araması", ["ara", "bul", "kim", "kayıt"],
      ["ara", "bul", "getir", "listele"],
      [P("query", "query", True, "Aranacak isim / e-posta / şirket", "aranacak isim")],
      [("matches", "count")],
      ["şu ada uyan kayıt var mı", "bu kişiyi sistemde bul"]),
    T("crm_create_contact", "crm", "write",
      "Yeni bir kişi kaydı oluşturur. Onay gerektirir.",
      "kişi kaydını", "yeni kişi", ["ekle", "yeni", "kişi", "kayıt"],
      ["oluştur", "ekle", "kaydet", "gir"],
      [P("name", "name", True, "Kişinin adı"),
       P("account_id", "id", True, "Bağlı hesap kimliği", "hesap", prefix="ACC"),
       P("email", "email", False, "E-posta"), P("phone", "phone", False, "Telefon")],
      [("contact_id", "id")],
      ["yeni bir irtibat noktası tanımlamam lazım", "bu kişiyi sisteme işle"]),
    T("crm_update_contact", "crm", "write",
      "Bir kişi kaydının iletişim alanlarını günceller. Onay gerektirir.",
      "kişi kaydını", "kişi güncelleme", ["güncelle", "değiştir", "düzelt", "kişi"],
      ["güncelle", "değiştir", "düzelt", "yenile"],
      [ID("contact_id", "CNT", "kişi", True),
       P("email", "email", False, "Yeni e-posta"), P("phone", "phone", False, "Yeni telefon"),
       P("title", "title", False, "Yeni unvan")],
      [("contact_id", "id")],
      ["o kişinin numarası değişti, işle", "kartındaki bilgiyi yenile"]),
    T("crm_get_account", "crm", "read",
      "Bir müşteri hesabının özetini (sektör, büyüklük, sorumlu, açık fırsat) getirir.",
      "hesabı", "hesap", ["hesap", "müşteri", "firma", "cari"],
      ["getir", "aç", "özetle", "göster"],
      [ID("account_id", "ACC", "hesap", True)],
      [("industry", "title"), ("owner", "name"), ("open_deals", "count")],
      ["o firmanın durumu ne", "bu cari hakkında ne biliyoruz"]),
    T("crm_list_account_contacts", "crm", "read",
      "Bir hesaba bağlı tüm kişi kayıtlarını listeler.",
      "hesabın kişilerini", "hesap kişileri", ["kişiler", "irtibatlar", "hesap", "liste"],
      ["listele", "getir", "çıkar", "göster"],
      [ID("account_id", "ACC", "hesap", True)],
      [("count", "count")],
      ["o firmada kimlerle görüşüyoruz", "hesaba bağlı isimleri ver"]),
    T("crm_log_interaction", "crm", "write",
      "Bir kişiyle yapılan görüşmeyi (arama, toplantı, e-posta) kayıt altına alır. Onay gerektirir.",
      "görüşme kaydını", "görüşme kaydı", ["görüşme", "not", "etkileşim", "arama"],
      ["kaydet", "gir", "işle", "ekle"],
      [ID("contact_id", "CNT", "kişi", True),
       P("channel", "enum", True, "Kanal", "kanal", ("call", "meeting", "email"),
         {"call": ["telefon", "arama"], "meeting": ["toplantı", "görüşme"], "email": ["e-posta", "mail"]}),
       P("summary", "reason", True, "Görüşme özeti"),
       P("date", "past_date", False, "Görüşme tarihi")],
      [("interaction_id", "id")],
      ["az önceki telefonu not düş", "yaptığımız toplantıyı sisteme yaz"]),
    T("crm_get_deal", "crm", "read",
      "Bir satış fırsatının ayrıntılarını (tutar, aşama, kapanış tahmini, sorumlu) getirir.",
      "fırsatı", "fırsat", ["fırsat", "deal", "satış", "pipeline"],
      ["getir", "aç", "göster", "özetle"],
      [ID("deal_id", "DEA", "fırsat", True)],
      [("amount", "amount"), ("stage", "enum"), ("close_date", "future_date")],
      ["o satış ne durumda", "anlaşmanın büyüklüğü ne"]),
    T("crm_list_deals", "crm", "read",
      "Bir hesap, sorumlu veya aşamaya göre satış fırsatlarını listeler.",
      "fırsatları", "fırsat listesi", ["fırsatlar", "satışlar", "pipeline", "liste"],
      ["listele", "getir", "çıkar", "dök"],
      [P("account_id", "id", False, "Hesap kimliği", "hesap", prefix="ACC"),
       P("stage", "enum", False, "Aşama filtresi", "aşama", _DEAL_STAGE)],
      [("count", "count"), ("total", "amount")],
      ["o müşteride hangi satışlar açık", "kapanışa yakın işleri sırala"]),
    T("crm_update_deal_stage", "crm", "write",
      "Bir satış fırsatının aşamasını ilerletir/değiştirir. Onay gerektirir.",
      "fırsatın aşamasını", "fırsat aşaması", ["aşama", "ilerlet", "güncelle", "taşı"],
      ["güncelle", "ilerlet", "taşı", "geçir"],
      [ID("deal_id", "DEA", "fırsat", True),
       P("stage", "enum", True, "Yeni aşama", "aşama", _DEAL_STAGE)],
      [("stage", "enum")],
      ["o anlaşmayı bir sonraki adıma al", "müşteri sözleşmeyi imzaladı, kaydı güncelle"]),
    T("crm_get_customer_churn_risk", "crm", "read",
      "Bir hesabın güncel kayıp (churn) risk skorunu ve etkenlerini getirir.",
      "kayıp riskini", "kayıp riski", ["risk", "churn", "kayıp", "memnuniyet"],
      ["getir", "hesapla", "göster", "değerlendir"],
      [ID("account_id", "ACC", "hesap", True)],
      [("score", "pct"), ("top_factor", "title")],
      ["o müşteriyi kaybetme ihtimalimiz ne", "hangi hesaplar tehlikede"]),
)

# ---- 6. it_support / BT Destek -------------------------------------- #
_TICKET_PRIO = ("low", "medium", "high", "urgent")
_add(
    T("it_create_ticket", "it_support", "write",
      "Bir BT destek talebi (arıza / istek) oluşturur.",
      "destek talebini", "destek talebi", ["talep", "arıza", "sorun", "ticket", "kayıt aç"],
      ["oluştur", "aç", "gir", "başlat"],
      [P("reporter_id", "emp_id", True, "Talebi açan personel kimliği", "personel numarası"),
       P("category", "enum", True, "Talep türü", "talep türü",
         ("hardware", "software", "access", "network"),
         {"hardware": ["donanım", "cihaz"], "software": ["yazılım", "program"],
          "access": ["erişim", "yetki"], "network": ["ağ", "internet", "bağlantı"]}),
       P("summary", "reason", True, "Sorun özeti"),
       P("priority", "enum", False, "Öncelik", "öncelik", _TICKET_PRIO)],
      [("ticket_id", "id"), ("status", "enum")],
      ["bilgisayarım açılmıyor, yardım lazım", "programa giremiyorum, kayıt açar mısın"]),
    T("it_get_ticket", "it_support", "read",
      "Bir BT destek talebinin durumunu ve atanan kişiyi getirir.",
      "destek talebini", "talep durumu", ["talep", "ticket", "durum", "atanan"],
      ["kontrol et", "getir", "bak", "sor"],
      [ID("ticket_id", "TIC", "destek talebi", True)],
      [("status", "enum"), ("assignee", "name")],
      ["açtığım kayıt ne aşamada", "sorunla kim ilgileniyor"]),
    T("it_update_ticket", "it_support", "write",
      "Bir BT talebine not ekler veya durumunu günceller. Onay gerektirir.",
      "destek talebini", "talep güncelleme", ["güncelle", "not", "durum", "talep"],
      ["güncelle", "not ekle", "kapat", "ilerlet"],
      [ID("ticket_id", "TIC", "destek talebi", True),
       P("status", "enum", False, "Yeni durum", "durum", ("open", "in_progress", "resolved", "closed")),
       P("note", "reason", False, "Eklenecek not")],
      [("status", "enum")],
      ["o kayda bir açıklama düş", "sorun çözüldü, kapatabilirsin"]),
    T("it_list_my_tickets", "it_support", "read",
      "Bir çalışanın açtığı BT taleplerini duruma göre listeler.",
      "taleplerimi", "talep listesi", ["taleplerim", "kayıtlarım", "açtıklarım", "liste"],
      ["listele", "getir", "göster", "çıkar"],
      [P("reporter_id", "emp_id", True, "Personel kimliği", "personel numarası"),
       P("status", "enum", False, "Durum filtresi", "durum", ("open", "in_progress", "resolved", "closed"))],
      [("count", "count")],
      ["benim açık işlerim neler", "geçmişte ne bildirmişim"]),
    T("it_reset_password_request", "it_support", "action",
      "Bir çalışan için parola sıfırlama talebi başlatır (kendi hesabı için).",
      "parola sıfırlamayı", "parola sıfırlama", ["parola", "şifre", "sıfırla", "giriş"],
      ["başlat", "talep et", "aç", "gönder"],
      [P("employee_id", "emp_id", True, "Kendi personel kimliğiniz", "personel numarası"),
       P("system", "enum", True, "Hangi sistem", "sistem", ("email", "vpn", "portal", "erp"))],
      [("request_id", "id"), ("status", "enum")],
      ["hesabıma giremiyorum, kilit açılsın", "girişimi yenilemem gerekiyor"]),
    T("it_get_asset", "it_support", "read",
      "Bir zimmet varlığının (dizüstü, telefon, monitör) kayıt bilgisini getirir.",
      "zimmet kaydını", "zimmet kaydı", ["zimmet", "varlık", "cihaz", "envanter no"],
      ["getir", "aç", "göster", "bul"],
      [ID("asset_id", "AST", "varlık", True)],
      [("type", "title"), ("assigned_to", "name"), ("status", "enum")],
      ["o cihaz kimin üstünde", "envanterde bu numara ne"]),
    T("it_assign_asset", "it_support", "write",
      "Bir zimmet varlığını bir çalışana atar. Onay gerektirir.",
      "zimmeti", "zimmet atama", ["zimmet", "ata", "teslim", "ver"],
      ["ata", "teslim et", "kaydet", "ver"],
      [ID("asset_id", "AST", "varlık", True),
       P("employee_id", "emp_id", True, "Zimmetlenecek personel kimliği", "personel numarası")],
      [("status", "enum")],
      ["yeni dizüstüyü o kişinin üstüne geç", "cihazı şu personele tanımla"]),
    T("it_check_service_status", "it_support", "read",
      "Bir kurumsal servisin (e-posta, VPN, ERP) güncel çalışma durumunu getirir.",
      "servis durumunu", "servis durumu", ["servis", "sistem", "kesinti", "erişim sorunu"],
      ["kontrol et", "getir", "sor", "bak"],
      [P("service", "enum", True, "Servis", "servis", ("email", "vpn", "erp", "portal", "wifi"))],
      [("state", "enum"), ("since", "date")],
      ["sistem herkeste mi yavaş", "bağlantı sorunu bizde mi kaynaklı"]),
    T("it_grant_access_request", "it_support", "write",
      "Bir çalışan için bir uygulamaya erişim yetkisi talebi oluşturur. Onay gerektirir.",
      "erişim talebini", "erişim talebi", ["erişim", "yetki", "izin", "hesap açma"],
      ["oluştur", "talep et", "başlat", "aç"],
      [P("employee_id", "emp_id", True, "Personel kimliği", "personel numarası"),
       P("application", "app_name", True, "Uygulama adı"),
       P("access_level", "enum", False, "Yetki düzeyi", "yetki düzeyi", ("read", "write", "admin"))],
      [("request_id", "id"), ("status", "enum")],
      ["yeni ekip üyesine şu uygulamayı açtır", "bu sisteme girebilmem lazım, talep et"]),
)

# ---- 7. logistics / Lojistik -------------------------------------- #
_add(
    T("logistics_get_shipment", "logistics", "read",
      "Bir sevkiyatın ayrıntılarını (içerik, çıkış/varış, taşıyıcı, durum) getirir.",
      "sevkiyatı", "sevkiyat", ["sevkiyat", "gönderi", "kargo", "taşıma"],
      ["getir", "aç", "göster", "bul"],
      [ID("shipment_id", "SHP", "sevkiyat", True)],
      [("carrier", "title"), ("status", "enum"), ("eta", "future_date")],
      ["o gönderinin içinde ne var", "yük nereden çıkmış"]),
    T("logistics_track_shipment", "logistics", "read",
      "Bir sevkiyatın güncel konum ve teslim tahminini getirir.",
      "sevkiyatın konumunu", "sevkiyat takibi", ["takip", "nerede", "konum", "teslim"],
      ["takip et", "sorgula", "getir", "göster"],
      [ID("shipment_id", "SHP", "sevkiyat", True)],
      [("location", "title"), ("eta", "future_date")],
      ["paket şu an nerede", "ne zaman elimize ulaşır"]),
    T("logistics_create_shipment", "logistics", "write",
      "Yeni bir sevkiyat kaydı oluşturur. Onay gerektirir.",
      "sevkiyatı", "yeni sevkiyat", ["sevkiyat", "gönderi", "yeni", "sevk"],
      ["oluştur", "aç", "başlat", "planla"],
      [P("origin", "place", True, "Çıkış noktası (şehir/depo)"),
       P("destination", "place", True, "Varış noktası"),
       P("weight_kg", "weight", True, "Ağırlık (kg)"),
       P("ship_date", "future_date", False, "Sevk tarihi")],
      [("shipment_id", "id"), ("status", "enum")],
      ["şu adrese bir yük göndermem lazım", "sevk kaydını hazırla"]),
    T("logistics_list_shipments", "logistics", "read",
      "Bir tarih aralığı veya duruma göre sevkiyatları listeler.",
      "sevkiyatları", "sevkiyat listesi", ["sevkiyatlar", "gönderiler", "liste", "kargolar"],
      ["listele", "getir", "çıkar", "dök"],
      [P("start_date", "date", False, "Başlangıç"), P("end_date", "date", False, "Bitiş"),
       P("status", "enum", False, "Durum", "durum", ("pending", "in_transit", "delivered", "delayed"))],
      [("count", "count")],
      ["bu hafta yola çıkanları göster", "geciken gönderiler hangileri"]),
    T("logistics_get_carrier_rate", "logistics", "read",
      "Bir güzergâh ve ağırlık için taşıyıcı fiyat teklifini getirir.",
      "taşıma fiyatını", "taşıma fiyatı", ["fiyat", "navlun", "ücret", "teklif"],
      ["getir", "hesapla", "sorgula", "göster"],
      [P("origin", "place", True, "Çıkış"), P("destination", "place", True, "Varış"),
       P("weight_kg", "weight", True, "Ağırlık (kg)")],
      [("price", "amount"), ("transit_days", "count")],
      ["şu iki nokta arası göndermek kaça mal olur", "navlun ne tutar"]),
    T("logistics_schedule_pickup", "logistics", "write",
      "Bir sevkiyat için taşıyıcıdan alım (pickup) randevusu oluşturur. Onay gerektirir.",
      "alım randevusunu", "alım randevusu", ["alım", "pickup", "randevu", "toplama"],
      ["oluştur", "planla", "ayarla", "ver"],
      [ID("shipment_id", "SHP", "sevkiyat", True),
       P("pickup_date", "future_date", True, "Alım tarihi")],
      [("confirmation", "id"), ("status", "enum")],
      ["kargonun alınması için gün ayarla", "taşıyıcı gelip alsın, randevu koy"]),
    T("logistics_report_damage", "logistics", "write",
      "Bir sevkiyat için hasar/eksik bildirimi oluşturur. Onay gerektirir.",
      "hasar bildirimini", "hasar bildirimi", ["hasar", "eksik", "kırık", "şikayet"],
      ["oluştur", "bildir", "kaydet", "aç"],
      [ID("shipment_id", "SHP", "sevkiyat", True),
       P("description", "reason", True, "Hasar açıklaması")],
      [("case_id", "id"), ("status", "enum")],
      ["gelen üründe kırık vardı, bildirim gir", "eksik teslim aldık, kayıt aç"]),
)

# ---- 8. inventory / Envanter ------------------------------------- #
_add(
    T("inventory_get_stock_level", "inventory", "read",
      "Bir ürünün belirli bir depodaki güncel stok miktarını getirir.",
      "stok miktarını", "stok", ["stok", "adet", "mevcut", "eldeki"],
      ["getir", "göster", "kontrol et", "say"],
      [ID("item_id", "ITM", "ürün", True),
       P("warehouse_id", "id", False, "Depo kimliği", "depo", prefix="WH")],
      [("on_hand", "count"), ("reserved", "count")],
      ["o üründen kaç tane var", "elimizde ne kadar kaldı"]),
    T("inventory_list_low_stock", "inventory", "read",
      "Yeniden sipariş eşiğinin altına düşen ürünleri listeler.",
      "kritik stokları", "kritik stok listesi", ["kritik", "azalan", "eşik", "tükenen"],
      ["listele", "çıkar", "getir", "göster"],
      [P("warehouse_id", "id", False, "Depo kimliği", "depo", prefix="WH")],
      [("count", "count")],
      ["neler bitmek üzere", "hangi ürünleri acil almalıyız"]),
    T("inventory_create_stock_transfer", "inventory", "write",
      "İki depo arasında stok transferi oluşturur. Onay gerektirir.",
      "stok transferini", "stok transferi", ["transfer", "sevk", "depo değişimi", "aktar"],
      ["oluştur", "başlat", "aç", "planla"],
      [ID("item_id", "ITM", "ürün", True),
       P("from_warehouse", "id", True, "Kaynak depo", "kaynak depo", prefix="WH"),
       P("to_warehouse", "id", True, "Hedef depo", "hedef depo", prefix="WH"),
       P("quantity", "count", True, "Miktar")],
      [("transfer_id", "id"), ("status", "enum")],
      ["o üründen bir kısmını diğer depoya kaydır", "şubeye mal göndermem lazım"]),
    T("inventory_get_item", "inventory", "read",
      "Bir ürünün kayıt bilgilerini (ad, kategori, birim, tedarikçi, fiyat) getirir.",
      "ürün kaydını", "ürün kaydı", ["ürün", "malzeme", "kart", "sku"],
      ["getir", "aç", "göster", "bul"],
      [ID("item_id", "ITM", "ürün", True)],
      [("name", "title"), ("category", "title"), ("unit_price", "amount")],
      ["bu kod hangi ürün", "o malzemenin birim fiyatı ne"]),
    T("inventory_adjust_stock_count", "inventory", "write",
      "Sayım sonucu bir ürünün stok adedini düzeltir. Onay gerektirir.",
      "stok sayımını", "stok düzeltmesi", ["sayım", "düzelt", "fire", "fazla"],
      ["düzelt", "güncelle", "işle", "kaydet"],
      [ID("item_id", "ITM", "ürün", True),
       P("warehouse_id", "id", True, "Depo", "depo", prefix="WH"),
       P("counted_quantity", "count", True, "Sayılan miktar"),
       P("reason", "reason", True, "Fark gerekçesi")],
      [("adjustment", "count"), ("status", "enum")],
      ["saydık, sistemdeki rakam yanlış; düzelt", "depoda eksik çıktı, güncelle"]),
    T("inventory_get_warehouse", "inventory", "read",
      "Bir deponun bilgilerini (konum, kapasite, sorumlu, doluluk) getirir.",
      "depo kaydını", "depo kaydı", ["depo", "ambar", "lokasyon", "kapasite"],
      ["getir", "aç", "göster", "özetle"],
      [P("warehouse_id", "id", True, "Depo kimliği", "depo", prefix="WH")],
      [("location", "title"), ("capacity", "count"), ("utilization", "pct")],
      ["o ambar nerede", "deponun doluluk oranı ne"]),
    T("inventory_reserve_stock", "inventory", "write",
      "Bir sipariş için stok rezervasyonu oluşturur. Onay gerektirir.",
      "stok rezervasyonunu", "stok rezervasyonu", ["rezerv", "ayır", "blokaj", "tut"],
      ["oluştur", "ayır", "tut", "bloke et"],
      [ID("item_id", "ITM", "ürün", True),
       P("quantity", "count", True, "Miktar"),
       P("order_id", "id", True, "Sipariş kimliği", "sipariş", prefix="ORD")],
      [("reservation_id", "id"), ("status", "enum")],
      ["o sipariş için ürünleri kenara ayır", "malı bloke et, satış bekliyor"]),
)

# ---- 9. sales / Satış ------------------------------------------- #
_add(
    T("sales_get_quote", "sales", "read",
      "Bir satış teklifinin ayrıntılarını (kalemler, tutar, geçerlilik, durum) getirir.",
      "teklifi", "teklif", ["teklif", "fiyat teklifi", "proforma"],
      ["getir", "aç", "göster", "bul"],
      [ID("quote_id", "QTE", "teklif", True)],
      [("total", "amount"), ("valid_until", "future_date"), ("status", "enum")],
      ["o teklifte ne kadar yazdık", "verdiğimiz fiyat ne zamana kadar geçerli"]),
    T("sales_create_quote", "sales", "write",
      "Bir müşteri için yeni satış teklifi oluşturur. Onay gerektirir.",
      "teklifi", "yeni teklif", ["teklif", "fiyat ver", "yeni", "hazırla"],
      ["oluştur", "hazırla", "çıkar", "gir"],
      [ID("account_id", "ACC", "hesap", True),
       P("amount", "amount", True, "Teklif tutarı"),
       P("valid_days", "duration", False, "Geçerlilik süresi (gün)")],
      [("quote_id", "id"), ("status", "enum")],
      ["o müşteriye fiyat çıkaralım", "yeni bir teklif hazırla"]),
    T("sales_get_order", "sales", "read",
      "Bir satış siparişinin ayrıntılarını (kalemler, tutar, teslim, durum) getirir.",
      "siparişi", "sipariş", ["sipariş", "order", "satış kaydı"],
      ["getir", "aç", "göster", "bul"],
      [ID("order_id", "ORD", "sipariş", True)],
      [("total", "amount"), ("status", "enum"), ("delivery_date", "future_date")],
      ["o satışın detayı ne", "siparişte hangi ürünler var"]),
    T("sales_list_orders", "sales", "read",
      "Bir müşteri, tarih aralığı veya duruma göre siparişleri listeler.",
      "siparişleri", "sipariş listesi", ["siparişler", "satışlar", "liste", "orderlar"],
      ["listele", "getir", "çıkar", "dök"],
      [P("account_id", "id", False, "Hesap kimliği", "hesap", prefix="ACC"),
       P("start_date", "date", False, "Başlangıç"), P("end_date", "date", False, "Bitiş"),
       P("status", "enum", False, "Durum", "durum", ("open", "shipped", "delivered", "cancelled"))],
      [("count", "count"), ("total", "amount")],
      ["o müşteriden bu ay ne geldi", "açık siparişleri sırala"]),
    T("sales_get_order_status", "sales", "read",
      "Bir siparişin güncel işlem/teslim durumunu getirir.",
      "siparişin durumunu", "sipariş durumu", ["durum", "nerede", "teslim", "sipariş"],
      ["kontrol et", "sor", "getir", "bak"],
      [ID("order_id", "ORD", "sipariş", True)],
      [("status", "enum"), ("delivery_date", "future_date")],
      ["müşteriye ne zaman ulaşır", "o sipariş hazırlandı mı"]),
    T("sales_cancel_order", "sales", "write",
      "Bir satış siparişini iptal eder. Onay gerektirir.",
      "siparişi", "sipariş iptali", ["iptal", "sipariş", "geri", "vazgeç"],
      ["iptal et", "geri al", "kaldır", "durdur"],
      [ID("order_id", "ORD", "sipariş", True),
       P("reason", "reason", True, "İptal gerekçesi")],
      [("status", "enum")],
      ["müşteri vazgeçti, o satışı kapat", "yanlış girilmiş, iptal et"]),
    T("sales_get_pricing", "sales", "read",
      "Bir ürün ve müşteri segmenti için güncel liste/indirimli fiyatı getirir.",
      "fiyatı", "fiyat", ["fiyat", "liste fiyatı", "iskonto", "tarife"],
      ["getir", "söyle", "göster", "hesapla"],
      [ID("item_id", "ITM", "ürün", True),
       P("segment", "enum", False, "Müşteri segmenti", "segment", ("retail", "wholesale", "key_account"))],
      [("list_price", "amount"), ("discount", "pct")],
      ["o ürünü toptancıya kaça veriyoruz", "bu kalemin güncel tarifesi ne"]),
    T("sales_get_target_progress", "sales", "read",
      "Bir satış temsilcisinin dönem hedefine göre gerçekleşme durumunu getirir.",
      "hedef gerçekleşmesini", "hedef gerçekleşme", ["hedef", "kota", "gerçekleşme", "prim"],
      ["getir", "göster", "hesapla", "özetle"],
      [P("rep_id", "emp_id", True, "Temsilci personel kimliği", "personel numarası"),
       P("period", "period", False, "Dönem")],
      [("target", "amount"), ("achieved", "amount"), ("pct", "pct")],
      ["kotamın ne kadarını yaptım", "bu ay hedefe ulaştım mı"]),
    T("sales_apply_discount_approval", "sales", "write",
      "Liste dışı bir indirim için onay talebi oluşturur. Onay gerektirir.",
      "indirim onayını", "indirim onayı", ["indirim", "iskonto onayı", "özel fiyat"],
      ["oluştur", "talep et", "başlat", "gönder"],
      [ID("quote_id", "QTE", "teklif", True),
       P("discount", "pct", True, "İstenen indirim oranı"),
       P("reason", "reason", True, "Gerekçe")],
      [("request_id", "id"), ("status", "enum")],
      ["müşteriye ekstra iskonto vermek istiyorum, onaya sun", "liste dışı fiyat için izin al"]),
)

# ---- 10. calendar / Takvim ------------------------------------ #
_add(
    T("calendar_get_events", "calendar", "read",
      "Bir kişinin belirli bir gün/aralıktaki takvim etkinliklerini listeler.",
      "takvim etkinliklerini", "takvim", ["takvim", "etkinlik", "toplantı", "ajanda", "program"],
      ["listele", "getir", "göster", "çıkar"],
      [P("employee_id", "emp_id", True, "Personel kimliği", "personel numarası"),
       P("date", "date", True, "Gün")],
      [("count", "count")],
      ["o gün neyim var", "programım ne durumda", "hangi toplantılara gireceğim"]),
    T("calendar_create_event", "calendar", "write",
      "Bir takvim etkinliği oluşturur ve katılımcıları davet eder. Onay gerektirir.",
      "etkinliği", "yeni etkinlik", ["etkinlik", "toplantı", "davet", "randevu", "planla"],
      ["oluştur", "planla", "kur", "ayarla"],
      [P("title", "event_title", True, "Etkinlik başlığı"),
       P("start", "future_date", True, "Başlangıç"),
       P("duration_min", "minutes", False, "Süre (dakika)"),
       P("attendee_id", "emp_id", False, "Davet edilecek personel")],
      [("event_id", "id")],
      ["yarın için bir görüşme koy", "şu konuyu konuşmak üzere zaman ayarla"]),
    T("calendar_cancel_event", "calendar", "write",
      "Bir takvim etkinliğini iptal eder ve katılımcıları bilgilendirir. Onay gerektirir.",
      "etkinliği", "etkinlik iptali", ["iptal", "etkinlik", "toplantı", "kaldır"],
      ["iptal et", "kaldır", "sil", "ertele"],
      [ID("event_id", "EVT", "etkinlik", True)],
      [("status", "enum")],
      ["o toplantıyı kaldır", "randevuyu iptal edelim"]),
    T("calendar_find_free_slot", "calendar", "read",
      "İki veya daha çok kişinin ortak boş zamanını belirli bir aralıkta bulur.",
      "ortak boş zamanı", "boş zaman", ["boş zaman", "müsait", "uygun saat", "slot"],
      ["bul", "getir", "öner", "ara"],
      [P("employee_id", "emp_id", True, "Birinci kişi", "personel numarası"),
       P("other_id", "emp_id", True, "İkinci kişi", "diğer personel"),
       P("date", "future_date", True, "Aranacak gün")],
      [("slots", "count")],
      ["ikimizin de uygun olduğu bir zaman var mı", "ne zaman buluşabiliriz"]),
    T("calendar_get_room_availability", "calendar", "read",
      "Bir toplantı odasının belirli bir gün/saatteki doluluk durumunu getirir.",
      "oda doluluğunu", "oda doluluğu", ["oda", "salon", "müsait mi", "rezerv"],
      ["kontrol et", "getir", "bak", "sorgula"],
      [P("room_id", "id", True, "Oda kimliği", "oda", prefix="ROOM"),
       P("date", "future_date", True, "Gün")],
      [("free_slots", "count")],
      ["büyük salon yarın boş mu", "o oda ne zaman kullanılabilir"]),
    T("calendar_book_room", "calendar", "write",
      "Bir toplantı odasını belirli bir zaman aralığı için rezerve eder. Onay gerektirir.",
      "oda rezervasyonunu", "oda rezervasyonu", ["oda ayır", "salon tut", "rezerve et", "kirala"],
      ["rezerve et", "ayır", "tut", "kaydet"],
      [P("room_id", "id", True, "Oda kimliği", "oda", prefix="ROOM"),
       P("start", "future_date", True, "Başlangıç"),
       P("duration_min", "minutes", True, "Süre (dakika)")],
      [("booking_id", "id"), ("status", "enum")],
      ["toplantı için o salonu bize ayır", "odayı adımıza kilitle"]),
    T("calendar_reschedule_event", "calendar", "write",
      "Bir takvim etkinliğini yeni bir zamana taşır. Onay gerektirir.",
      "etkinliği", "etkinlik erteleme", ["ertele", "taşı", "değiştir", "yeni saat"],
      ["ertele", "taşı", "kaydır", "güncelle"],
      [ID("event_id", "EVT", "etkinlik", True),
       P("new_start", "future_date", True, "Yeni başlangıç")],
      [("status", "enum")],
      ["görüşmeyi başka güne alalım", "o randevuyu ileri çek"]),
)

# ---- 11. documents / Belge ----------------------------------- #
_add(
    T("docs_search", "documents", "read",
      "Anahtar kelime, tür veya sahibe göre belgeleri arar.",
      "belgeleri", "belge araması", ["belge", "doküman", "dosya", "ara"],
      ["ara", "bul", "getir", "listele"],
      [P("query", "query", True, "Aranacak ifade", "aranacak ifade"),
       P("doc_type", "enum", False, "Belge türü", "belge türü",
         ("policy", "report", "contract", "form"),
         {"policy": ["politika", "prosedür"], "report": ["rapor"], "contract": ["sözleşme"], "form": ["form"]})],
      [("matches", "count")],
      ["şu konuyla ilgili yazı var mı", "o başlıkta bir şey bul"]),
    T("docs_get_document", "documents", "read",
      "Bir belgenin meta bilgilerini (başlık, sürüm, sahip, tarih, erişim) getirir.",
      "belgeyi", "belge", ["belge", "doküman", "dosya", "kayıt"],
      ["getir", "aç", "göster", "bul"],
      [ID("doc_id", "DOC", "belge", True)],
      [("title", "title"), ("version", "count"), ("owner", "name")],
      ["o dosyanın son sürümü hangisi", "belgeyi kim hazırlamış"]),
    T("docs_create_document_request", "documents", "write",
      "İK/idari bir belge talebi (çalışma belgesi, bordro yazısı) oluşturur. Onay gerektirir.",
      "belge talebini", "belge talebi", ["belge talebi", "yazı", "çalışma belgesi", "evrak"],
      ["oluştur", "aç", "gir", "başlat"],
      [P("employee_id", "emp_id", True, "Personel kimliği", "personel numarası"),
       P("document_type", "enum", True, "Belge türü", "belge türü",
         ("employment_cert", "salary_letter", "experience_cert"),
         {"employment_cert": ["çalışma belgesi", "görev belgesi"],
          "salary_letter": ["maaş yazısı", "gelir belgesi"], "experience_cert": ["hizmet belgesi", "deneyim belgesi"]}),
       P("purpose", "reason", False, "Kullanım amacı / kurum")],
      [("request_id", "id"), ("status", "enum")],
      ["bankaya vermek için çalıştığıma dair kağıt lazım", "gelir yazısı çıkart"]),
    T("docs_get_request_status", "documents", "read",
      "Bir belge talebinin hazırlanma durumunu getirir.",
      "belge talebinin durumunu", "belge talebi durumu", ["belge", "talep", "durum", "hazır mı"],
      ["kontrol et", "sor", "getir", "bak"],
      [ID("request_id", "DRQ", "belge talebi", True)],
      [("status", "enum"), ("ready_date", "date")],
      ["istediğim evrak hazır oldu mu", "yazım ne zaman çıkar"]),
    T("docs_share_document", "documents", "write",
      "Bir belgeyi belirli kişilerle paylaşır / erişim verir. Onay gerektirir.",
      "belgeyi", "belge paylaşımı", ["paylaş", "erişim ver", "gönder", "yetkilendir"],
      ["paylaş", "gönder", "erişim ver", "ilet"],
      [ID("doc_id", "DOC", "belge", True),
       P("recipient_id", "emp_id", True, "Paylaşılacak personel", "personel numarası"),
       P("access_level", "enum", False, "Erişim düzeyi", "erişim düzeyi", ("view", "comment", "edit"))],
      [("status", "enum")],
      ["o dosyayı şu kişiye aç", "belgeyi ekiple paylaş"]),
    T("docs_list_folder", "documents", "read",
      "Bir klasördeki belgeleri listeler.",
      "klasör içeriğini", "klasör içeriği", ["klasör", "dizin", "içerik", "liste"],
      ["listele", "getir", "aç", "göster"],
      [P("folder_id", "id", True, "Klasör kimliği", "klasör", prefix="FLD")],
      [("count", "count")],
      ["o dizinde neler var", "klasörün içindekileri göster"]),
    T("docs_request_signature", "documents", "write",
      "Bir belge için elektronik imza akışı başlatır. Onay gerektirir.",
      "imza talebini", "imza talebi", ["imza", "onaya gönder", "imzala", "e-imza"],
      ["başlat", "gönder", "oluştur", "aç"],
      [ID("doc_id", "DOC", "belge", True),
       P("signer_id", "emp_id", True, "İmzalayacak kişi", "personel numarası")],
      [("workflow_id", "id"), ("status", "enum")],
      ["sözleşmeyi imzaya sun", "o belgeyi onaya yolla"]),
)

# ---- 12. reporting / Raporlama ------------------------------ #
_add(
    T("reporting_run_report", "reporting", "read",
      "Kayıtlı bir raporu verilen parametrelerle çalıştırır ve sonucu döndürür.",
      "raporu", "rapor", ["rapor", "analiz", "döküm", "çalıştır"],
      ["çalıştır", "getir", "üret", "al"],
      [ID("report_id", "RPT", "rapor", True),
       P("period", "period", False, "Dönem parametresi")],
      [("rows", "count"), ("generated_at", "date")],
      ["o analizi bir çalıştır", "raporun bu ayki halini al"]),
    T("reporting_list_reports", "reporting", "read",
      "Bir kategoriye göre kayıtlı raporları listeler.",
      "raporları", "rapor listesi", ["raporlar", "liste", "hangi raporlar", "katalog"],
      ["listele", "getir", "göster", "çıkar"],
      [P("category", "enum", False, "Rapor kategorisi", "kategori",
         ("sales", "finance", "hr", "operations"))],
      [("count", "count")],
      ["ne tür dökümler alabiliyorum", "mevcut analizleri sırala"]),
    T("reporting_schedule_report", "reporting", "write",
      "Bir raporun düzenli aralıklarla e-posta ile gönderilmesini planlar. Onay gerektirir.",
      "rapor gönderimini", "rapor planı", ["planla", "otomatik", "düzenli", "abonelik"],
      ["planla", "kur", "ayarla", "abone et"],
      [ID("report_id", "RPT", "rapor", True),
       P("frequency", "enum", True, "Sıklık", "sıklık", ("daily", "weekly", "monthly"),
         {"daily": ["günlük", "her gün"], "weekly": ["haftalık", "her hafta"], "monthly": ["aylık", "her ay"]}),
       P("recipient", "email", True, "Alıcı e-posta")],
      [("schedule_id", "id"), ("status", "enum")],
      ["o raporu bana her hafta otomatik yolla", "düzenli gönderime bağla"]),
    T("reporting_get_report_result", "reporting", "read",
      "Daha önce çalıştırılmış bir raporun kayıtlı sonucunu getirir.",
      "rapor sonucunu", "rapor sonucu", ["sonuç", "çıktı", "önceki rapor", "kayıtlı"],
      ["getir", "aç", "göster", "indir"],
      [ID("run_id", "RUN", "rapor çalıştırması", True)],
      [("rows", "count"), ("summary", "title")],
      ["dün aldığım dökümü tekrar aç", "o çalıştırmanın çıktısı neydi"]),
    T("reporting_export_dataset", "reporting", "action",
      "Bir veri kümesini seçilen formatta dışa aktarır (kişisel/gizli veri hariç). Onay gerektirir.",
      "veri kümesini", "veri dışa aktarımı", ["dışa aktar", "export", "indir", "csv"],
      ["dışa aktar", "indir", "çıkar", "aktar"],
      [ID("dataset_id", "DS", "veri kümesi", True),
       P("format", "enum", True, "Format", "format", ("csv", "xlsx", "json"))],
      [("file_id", "id"), ("row_count", "count")],
      ["şu tabloyu dosya olarak al", "veriyi excele döksün"]),
    T("reporting_get_kpi_snapshot", "reporting", "read",
      "Bir birim için güncel temel performans göstergesi (KPI) özetini getirir.",
      "KPI özetini", "KPI özeti", ["kpi", "gösterge", "metrik", "performans özeti"],
      ["getir", "göster", "özetle", "çıkar"],
      [P("unit_name", "org_name", True, "Birim adı", "birim adı"),
       P("period", "period", False, "Dönem")],
      [("revenue", "amount"), ("headcount", "count"), ("nps", "count")],
      ["o bölüm bu ay nasıl gidiyor", "temel rakamları bir göster"]),
)

# ---- 13. support / Müşteri Destek ------------------------- #
_add(
    T("support_get_customer", "support", "read",
      "Bir müşterinin destek profilini (plan, açık kayıt sayısı, SLA seviyesi) getirir.",
      "müşteri profilini", "müşteri profili", ["müşteri", "abone", "profil", "plan"],
      ["getir", "aç", "göster", "özetle"],
      [ID("customer_id", "CUS", "müşteri", True)],
      [("plan", "title"), ("open_cases", "count"), ("sla_tier", "title")],
      ["o abonenin paketi ne", "bu müşterinin açık sorunu var mı"]),
    T("support_get_customer_health", "support", "read",
      "Bir müşterinin sağlık skorunu (kullanım, memnuniyet, ödeme) getirir.",
      "müşteri sağlık skorunu", "sağlık skoru", ["sağlık", "skor", "memnuniyet", "kullanım"],
      ["getir", "hesapla", "göster", "değerlendir"],
      [ID("customer_id", "CUS", "müşteri", True)],
      [("score", "count"), ("trend", "title")],
      ["o hesap iyi durumda mı", "müşteri bizden memnun mu"]),
    T("support_list_open_cases", "support", "read",
      "Bir müşteri veya atanan kişiye göre açık destek kayıtlarını listeler.",
      "açık kayıtları", "açık kayıtlar", ["kayıt", "case", "açık", "liste"],
      ["listele", "getir", "çıkar", "göster"],
      [P("customer_id", "id", False, "Müşteri kimliği", "müşteri", prefix="CUS"),
       P("assignee_id", "emp_id", False, "Atanan personel", "atanan personel numarası")],
      [("count", "count")],
      ["o müşteride ne açık işimiz var", "bana atanmış sorunlar neler"]),
    T("support_create_case", "support", "write",
      "Yeni bir müşteri destek kaydı oluşturur.",
      "destek kaydını", "destek kaydı", ["kayıt aç", "case", "sorun", "talep"],
      ["oluştur", "aç", "gir", "başlat"],
      [P("customer_id", "id", True, "Müşteri kimliği", "müşteri", prefix="CUS"),
       P("subject", "reason", True, "Konu"),
       P("severity", "enum", False, "Önem", "önem", ("s1", "s2", "s3", "s4"))],
      [("case_id", "id"), ("status", "enum")],
      ["müşteri arıza bildirdi, kayıt aç", "şu sorunu sisteme gir"]),
    T("support_escalate_case", "support", "action",
      "Bir destek kaydını üst seviye ekibe yükseltir. Onay gerektirir.",
      "destek kaydını", "kayıt yükseltme", ["yükselt", "escalate", "üst ekip", "acil"],
      ["yükselt", "ilet", "aktar", "escalate et"],
      [ID("case_id", "CASE", "destek kaydı", True),
       P("reason", "reason", True, "Yükseltme gerekçesi")],
      [("status", "enum"), ("escalated_to", "title")],
      ["bu işi ikinci kademeye devret", "çözemedik, üste taşı"]),
    T("support_get_sla_status", "support", "read",
      "Bir destek kaydının SLA (yanıt/çözüm süresi) durumunu getirir.",
      "SLA durumunu", "SLA durumu", ["sla", "süre", "gecikme", "hedef süre"],
      ["kontrol et", "getir", "göster", "bak"],
      [ID("case_id", "CASE", "destek kaydı", True)],
      [("response_due", "future_date"), ("breached", "count")],
      ["o kayıtta süremiz doldu mu", "SLA'yı aştık mı"]),
    T("support_send_customer_update", "support", "write",
      "Bir destek kaydında müşteriye durum güncellemesi mesajı gönderir. Onay gerektirir.",
      "müşteri bilgilendirmesini", "müşteri bilgilendirmesi", ["bilgilendir", "mesaj gönder", "haber ver", "yanıtla"],
      ["gönder", "ilet", "yaz", "bildir"],
      [ID("case_id", "CASE", "destek kaydı", True),
       P("message", "reason", True, "Mesaj içeriği")],
      [("status", "enum")],
      ["müşteriye gelişmeyi yazalım", "o kayıtta güncelleme geç"]),
)


# --------------------------------------------------------------------------- #
#  Bütünlük + bölme
# --------------------------------------------------------------------------- #
_NAMES = [t.name for t in TOOLS]
assert len(_NAMES) == len(set(_NAMES)), "yinelenen tool adı"


def _fold_kw(s):
    s = s.replace("İ", "i").replace("I", "ı").lower()
    for a, b in (("ı", "i"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("ç", "c")):
        s = s.replace(a, b)
    return s


def _compute_disc_kw():
    """Her tool için AYIRT EDİCİ yüzey sözcüğü/sözcükleri: bu tool'un kw'lerinden,
    diğer tool'ların nesne+syn metninde EN AZ geçen(ler). K-1 (keyword->tool adı
    kısayolu) ölçümü ve anti-kısayol denetimi bunu kullanır.  `kw` (tam liste) ise
    hard-negative 'aynı kelime farklı tool' gruplaması için kalır."""
    corpora = {}
    for t in TOOLS:
        corpora[t.name] = _fold_kw(" ".join([t.obj_nom, t.obj] + list(t.syn)))
    for t in TOOLS:
        others = " ".join(v for n, v in corpora.items() if n != t.name)
        others += " " + " ".join(_fold_kw(x.obj_nom) for x in TOOLS if x.name != t.name)
        scored = sorted(t.kw, key=lambda k: others.count(_fold_kw(k)))
        t.disc_kw = tuple(scored[:2]) if scored else ()


# --------------------------------------------------------------------------- #
#  SYN OVERRIDE — anti-kısayol (D-4): dolaylı ifadeler; tool'un ayırt edici
#  yüzey sözcüğünü İÇERMEZ. Model niyeti açıklamadan çözmek zorunda kalır.
# --------------------------------------------------------------------------- #
_SYN_OVERRIDE = {
    "hr_get_leave_balance": ["kaç günüm kaldı", "daha ne kadar dinlenebilirim",
                             "bu yıl geriye ne kaldı bende", "kullanabileceğim gün sayısı nedir"],
    "hr_get_leave_history": ["daha önce ne zaman uzaklaştım işten", "geçmişte hangi tarihlerde yoktum",
                             "önceki yıllarda ne kadar kullanmışım"],
    "hr_get_leave_request_status": ["başvurum kabul edildi mi", "yöneticiden dönüş geldi mi",
                                    "hâlâ bekliyor mu yoksa sonuçlandı mı"],
    "hr_create_leave_request": ["önümüzdeki hafta işte olmayacağım, gerekeni yap",
                                "birkaç gün uzaklaşacağım, sisteme geç", "şu tarihlerde yokum, kaydı aç"],
    "hr_cancel_leave_request": ["o planımdan vazgeçtim, geri al", "artık o günleri istemiyorum",
                                "yanlış girmişim, kaldır"],
    "hr_update_leave_request": ["o günleri başka zamana almak istiyorum", "planım kaydı, yeni güne çek",
                               "girdiğim aralığı değiştirelim"],
    "hr_get_employee_profile": ["bu kişi kim, ne iş yapıyor", "ne zamandır bizimle, hangi masada",
                               "kayıtta ne yazıyor bu personel için"],
    "hr_get_employment_status": ["hâlâ bizimle mi çalışıyor", "şu an işbaşında mı yoksa uzakta mı",
                                "bu kişi hâlâ kadroda mı"],
    "hr_get_manager": ["bu kişi kime rapor veriyor", "kimin altında çalışıyor", "üstü kim"],
    "hr_get_org_unit": ["orada kaç kişi var", "başında kim var o tarafın", "kaç alt grup bağlı"],
    "hr_list_unit_members": ["orada kimler var", "o taraftaki isimleri ver", "kim kim orada çalışıyor"],
    "payroll_get_salary": ["eline ne geçiyor aylık", "bu kişi ne kazanıyor", "aylık ödemesi ne kadar"],
    "payroll_get_payslip": ["o ayki hesap dökümü lazım", "kesintilerle birlikte o ayı görmek istiyorum",
                            "geçen ay ne yatmış detaylı"],
    "payroll_get_bonus": ["bu yıl ekstra ne aldım", "performans ödemem çıktı mı", "yıl içinde ek ne geldi"],
    "payroll_get_benefits": ["bana ne gibi ek imkânlar tanımlı", "sağlık ve yemek tarafında ne var bende",
                             "maaş dışında neler sağlanıyor"],
    "payroll_get_tax_summary": ["neden her ay elime geçen azalıyor", "yıl içinde biriken kesintim ne durumda",
                                "dilim değişti mi bu yıl"],
    "payroll_create_salary_change_request": ["bu kişinin ödemesini yukarı çekmek istiyorum",
                                             "kazancını artıralım, talebini başlat"],
    "payroll_get_deduction_breakdown": ["brütten ne ne düşülmüş görmek istiyorum",
                                        "o ay hangi kalemler kesilmiş"],
    "payroll_get_payment_history": ["hesabıma son aylarda neler yattı", "geçmişte yapılan ödemeleri sırala"],
    "payroll_get_severance_estimate": ["ayrılırsa eline ne geçer", "çıkışta ne kadar ödenir bu kişiye"],
    "timesheet_get_records": ["o dönem kaç gün işbaşı yapmış", "giriş çıkışlarını görmek istiyorum",
                              "hangi günler gelmemiş"],
    "timesheet_get_overtime": ["geçen ay normalin üstünde ne kadar çalıştım", "ek çalışma karşılığım ne oldu"],
    "timesheet_submit_time_correction": ["o gün yanlış işlenmiş, düzeltilsin", "sistemde eksik görünüyorum, talep aç"],
    "timesheet_get_attendance_summary": ["o ay kaç kez geç kalınmış", "gelmeyen günlerin tablosu lazım"],
    "timesheet_get_shift_schedule": ["bu hafta hangi saatlerde çalışıyorum", "haftalık listem ne"],
    "timesheet_request_leave_of_absence": ["bir süre işe ara vermem gerekiyor", "uzunca uzaklaşacağım, kaydını aç"],
    "finance_get_invoice": ["o belgede ne kadar yazıyor", "o kâğıdın vadesi ne zaman", "tutar ve tarih ne"],
    "finance_list_invoices": ["o firmadan gelen belgeleri sırala", "bu ay hangi ödemeler birikti"],
    "finance_create_expense_report": ["cebimden yaptığım harcamayı geri almak istiyorum",
                                      "iş için ödediğim tutarı sisteme gir"],
    "finance_get_expense_status": ["harcama iadem ne aşamada", "o kaydım onaylandı mı", "para yattı mı"],
    "finance_approve_expense": ["ekibimden gelen o kaydı geçir", "onu kabul ediyorum, işleme al"],
    "finance_get_budget_status": ["bu kalemde daha ne kadar param var", "harcama limitim doldu mu"],
    "finance_create_purchase_order": ["o firmadan alım yapmam lazım, kaydını aç", "onlardan bir şey ısmarla"],
    "finance_get_vendor": ["o firma hakkında ne biliyoruz", "ödeme vadesi kaç gün onların"],
    "finance_get_reimbursement_policy_limit": ["yemekte günlük ne kadara kadar ödenir", "konaklama için tavan ne"],
    "crm_get_contact": ["o kartta iletişim bilgisi ne", "bu kaydın detayları lazım"],
    "crm_search_contacts": ["şu ada uyan biri var mı sistemde", "bu ismi bul bakalım"],
    "crm_create_contact": ["yeni bir irtibat noktası tanımlamam lazım", "bunu sisteme işle"],
    "crm_update_contact": ["onun numarası değişti, işle", "kartındaki bilgiyi yenile"],
    "crm_get_account": ["o firmanın durumu ne", "bu cari hakkında ne biliyoruz", "orada ne kadar iş açık"],
    "crm_list_account_contacts": ["o firmada kimlerle görüşüyoruz", "oraya bağlı isimleri ver"],
    "crm_log_interaction": ["az önceki telefonu not düş", "yaptığımız görüşmeyi sisteme yaz"],
    "crm_get_deal": ["o iş ne durumda", "anlaşmanın büyüklüğü ne", "hangi aşamada kaldı"],
    "crm_list_deals": ["o müşteride neler açık", "kapanışa yakın olanları sırala"],
    "crm_update_deal_stage": ["o işi bir sonraki adıma al", "müşteri imzaladı, kaydı ilerlet"],
    "crm_get_customer_churn_risk": ["o müşteriyi kaybetme ihtimalimiz ne", "hangi hesaplar tehlikede"],
    "it_create_ticket": ["bilgisayarım açılmıyor, yardım lazım", "programa giremiyorum, bir el atın"],
    "it_get_ticket": ["açtığım iş ne aşamada", "sorunla kim ilgileniyor"],
    "it_update_ticket": ["o işe bir açıklama düş", "çözüldü, kapatabilirsin"],
    "it_list_my_tickets": ["benim açık işlerim neler", "geçmişte ne bildirmişim"],
    "it_reset_password_request": ["hesabıma giremiyorum, kilit açılsın", "girişimi yenilemem gerekiyor"],
    "it_get_asset": ["o cihaz kimin üstünde", "envanterde bu numara ne"],
    "it_assign_asset": ["yeni dizüstüyü o kişinin üstüne geç", "bu cihazı şu personele tanımla"],
    "it_check_service_status": ["sistem herkeste mi yavaş", "sorun bizden mi kaynaklı"],
    "it_grant_access_request": ["yeni ekip üyesine şu uygulamayı açtır", "bu sisteme girebilmem lazım"],
    "logistics_get_shipment": ["o gönderinin içinde ne var", "yük nereden çıkmış"],
    "logistics_track_shipment": ["paket şu an nerede", "ne zaman elimize ulaşır"],
    "logistics_create_shipment": ["şu noktaya bir yük göndermem lazım", "sevk kaydını hazırla"],
    "logistics_list_shipments": ["bu hafta yola çıkanları göster", "gecikenler hangileri"],
    "logistics_get_carrier_rate": ["şu iki nokta arası göndermek kaça mal olur", "bu ne tutar taşıması"],
    "logistics_schedule_pickup": ["gelip alsınlar diye gün ayarla", "toplama için randevu koy"],
    "logistics_report_damage": ["gelen üründe kırık vardı, bildir", "eksik teslim aldık, kaydını aç"],
    "inventory_get_stock_level": ["o üründen kaç tane var", "elimizde ne kadar kaldı"],
    "inventory_list_low_stock": ["neler bitmek üzere", "hangilerini acil almalıyız"],
    "inventory_create_stock_transfer": ["o üründen bir kısmını diğer tarafa kaydır", "şubeye mal göndermem lazım"],
    "inventory_get_item": ["bu kod hangi ürün", "onun birim fiyatı ne"],
    "inventory_adjust_stock_count": ["saydık, sistemdeki rakam yanlış; düzelt", "eksik çıktı, güncelle"],
    "inventory_get_warehouse": ["o yer nerede", "doluluk oranı ne orada"],
    "inventory_reserve_stock": ["o iş için ürünleri kenara ayır", "malı tut, satış bekliyor"],
    "sales_get_quote": ["o müşteriye ne kadar yazmıştık", "verdiğimiz fiyat ne zamana kadar geçerli"],
    "sales_create_quote": ["o müşteriye fiyat çıkaralım", "yeni bir fiyat hazırla"],
    "sales_get_order": ["o satışın detayı ne", "içinde hangi kalemler var"],
    "sales_list_orders": ["o müşteriden bu ay ne geldi", "açık olanları sırala"],
    "sales_get_order_status": ["müşteriye ne zaman ulaşır", "o hazırlandı mı"],
    "sales_cancel_order": ["müşteri vazgeçti, o satışı kapat", "yanlış girilmiş, geri al"],
    "sales_get_pricing": ["o ürünü toptancıya kaça veriyoruz", "bunun güncel değeri ne"],
    "sales_get_target_progress": ["kotamın ne kadarını yaptım", "bu ay istediğime ulaştım mı"],
    "sales_apply_discount_approval": ["müşteriye ekstra indirim vermek istiyorum, onaya sun",
                                     "liste dışı fiyat için izin al"],
    "calendar_get_events": ["o gün neyim var", "programım ne durumda", "hangi görüşmelere gireceğim"],
    "calendar_create_event": ["yarın için bir görüşme koy", "şu konuyu konuşmak üzere zaman ayarla"],
    "calendar_cancel_event": ["o görüşmeyi kaldır", "randevuyu boz"],
    "calendar_find_free_slot": ["ikimizin de uygun olduğu zaman var mı", "ne zaman buluşabiliriz"],
    "calendar_get_room_availability": ["büyük yer yarın boş mu", "orası ne zaman kullanılabilir"],
    "calendar_book_room": ["toplantı için orayı bize ayır", "orayı adımıza kilitle"],
    "calendar_reschedule_event": ["görüşmeyi başka güne alalım", "o randevuyu ileri çek"],
    "docs_search": ["şu konuyla ilgili yazı var mı", "o başlıkta bir şey bul"],
    "docs_get_document": ["onun son sürümü hangisi", "kim hazırlamış onu"],
    "docs_create_document_request": ["bankaya vermek için çalıştığıma dair kâğıt lazım", "gelir yazısı çıkart"],
    "docs_get_request_status": ["istediğim evrak hazır oldu mu", "yazım ne zaman çıkar"],
    "docs_share_document": ["onu şu kişiye aç", "bunu ekiple görüşülür yap"],
    "docs_list_folder": ["o dizinde neler var", "onun içindekileri göster"],
    "docs_request_signature": ["sözleşmeyi imzaya sun", "onu onaya yolla"],
    "reporting_run_report": ["o dökümü bir çalıştır", "bunun bu ayki halini al"],
    "reporting_list_reports": ["ne tür dökümler alabiliyorum", "mevcut olanları sırala"],
    "reporting_schedule_report": ["onu bana her hafta kendiliğinden yolla", "belirli aralıkla gönderime bağla"],
    "reporting_get_report_result": ["dün aldığım dökümü tekrar aç", "o çalıştırmanın çıktısı neydi"],
    "reporting_export_dataset": ["şu tabloyu dosya olarak al", "veriyi excele döksün"],
    "reporting_get_kpi_snapshot": ["o taraf bu ay nasıl gidiyor", "temel rakamları bir göster"],
    "support_get_customer": ["o abonenin paketi ne", "bunun açık sorunu var mı"],
    "support_get_customer_health": ["o hesap iyi durumda mı", "bizden memnun mu onlar"],
    "support_list_open_cases": ["o müşteride ne açık işimiz var", "bana atanmış sorunlar neler"],
    "support_create_case": ["müşteri arıza bildirdi, kaydını aç", "şu sorunu sisteme gir"],
    "support_escalate_case": ["bu işi ikinci kademeye devret", "çözemedik, üste taşı"],
    "support_get_sla_status": ["o işte süremiz doldu mu", "hedefi aştık mı orada"],
    "support_send_customer_update": ["müşteriye gelişmeyi yazalım", "orada bir güncelleme geç"],
}
for _t in TOOLS:
    if _t.name in _SYN_OVERRIDE:
        _t.syn = tuple(_SYN_OVERRIDE[_t.name])

_compute_disc_kw()


def by_name(name: str) -> Tool | None:
    for t in TOOLS:
        if t.name == name:
            return t
    return None


def assign_splits(val_ratio=0.15, test_ratio=0.15, seed="v2-split"):
    """Domain-stratifiye, deterministik hash tabanlı train/val/test bölmesi.
    val/test tool'ları eğitimde HEDEF olarak kullanılmaz (yalnız çeldirici)."""
    from collections import defaultdict
    dom = defaultdict(list)
    for t in TOOLS:
        dom[t.domain].append(t)
    for domain, ts in dom.items():
        ts_sorted = sorted(ts, key=lambda t: hashlib.md5(f"{seed}:{t.name}".encode()).hexdigest())
        n = len(ts_sorted)
        n_val = max(1, round(n * val_ratio))
        n_test = max(1, round(n * test_ratio))
        for i, t in enumerate(ts_sorted):
            if i < n_test:
                t.split = "test"
            elif i < n_test + n_val:
                t.split = "val"
            else:
                t.split = "train"
    return TOOLS


assign_splits()
