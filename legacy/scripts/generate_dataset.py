#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Büyük İK — Qwen LoRA tool-calling / tool-routing dataset üreticisi
==================================================================

NVIDIA When2Call yaklaşımını Büyük İK (sentetik İnsan Kaynakları) alanına
uyarlayan, KENDİNE YETERLİ (harici bağımlılık yok, API yok, deterministik)
bir sentetik veri üreticisidir.

Öğretilen dört karar davranışı:
    direct            -> tool gerekmiyor, doğrudan cevap ver
    tool_call         -> tool gerekli + gerekli parametreler mevcut
    request_for_info  -> tool var ama zorunlu bilgi eksik / onay gerekli
    cannot_answer     -> mevcut araçlarla cevaplanamaz

Çıktı:
    <out-dir>/<prefix>_train.jsonl       -> yalnızca {"tools": [...], "messages": [...]}
    <out-dir>/<prefix>_val.jsonl         -> aynı yapı
    <out-dir>/<prefix>_train.meta.jsonl  -> id + decision/intent/difficulty/... (QC & eval)
    <out-dir>/<prefix>_val.meta.jsonl
    <out-dir>/<prefix>_tools.json        -> bağımsız tool şeması envanteri
    <report>                             -> dağılım + üretim istatistikleri (markdown)

Kullanım:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --n 2000 --seed 20260827 --out-dir ./data
    python scripts/generate_dataset.py --n 500 --dry-run

Varsayılan çıktı dizini: depo kökündeki  data/  (bu dosyanın bir üst dizini / data).
Üretim raporu: çıktı KANONİK data/ ise  docs/generation_report.md ,  aksi halde
<out-dir>/generation_report.md  (repo raporu deneme/test koşularından korunur).

Tüm çalışan, ID, maaş, tarih, departman bilgisi SENTETİKTİR.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

# Windows konsolu cp1254 olabilir; '✓' / Türkçe karakterler stdout'ta çökmesin.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# 0. Genel yapılandırma
# ---------------------------------------------------------------------------

DEFAULT_N = 3000
DEFAULT_SEED = 20260827
DEFAULT_TODAY = "2026-08-27"
DEFAULT_PREFIX = "buyuk_ik_tool_calling"
DEFAULT_VAL_RATIO = 0.10

# Hedef karar dağılımı (When2Call ruhu — bkz. prompt §6)
TARGET_MIX = {
    "tool_call": 0.30,
    "direct": 0.25,
    "request_for_info": 0.25,
    "cannot_answer": 0.20,
}

# Qwen sohbet şablonu tool-call biçimi (bkz. prompt §21).
TOOLCALL_OPEN = "<tool_call>"
TOOLCALL_CLOSE = "</tool_call>"


def tool_call_block(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"{TOOLCALL_OPEN}\n{payload}\n{TOOLCALL_CLOSE}"


def tool_call_blocks(calls: list[tuple[str, dict]]) -> str:
    return "\n".join(tool_call_block(n, a) for n, a in calls)


# ---------------------------------------------------------------------------
# 1. TOOL ENVANTERİ  (flat "function" şeması — bkz. prompt §14, §37)
# ---------------------------------------------------------------------------

def _p(type_, desc, **extra):
    d = {"type": type_, "description": desc}
    d.update(extra)
    return d


EMP_ID_DESC = "Çalışanın benzersiz personel kimliği, 'EMP-1234' biçiminde"

TOOLS: dict[str, dict] = {
    # ---- Çalışan bilgileri --------------------------------------------------
    "get_employee_info": {
        "name": "get_employee_info",
        "description": "Bir çalışanın temel kayıt bilgilerini (ad, departman, pozisyon, işe giriş tarihi, yöneticisi) getirir.",
        "parameters": {
            "type": "object",
            "properties": {"employee_id": _p("string", EMP_ID_DESC)},
            "required": ["employee_id"],
        },
    },
    "get_employee_status": {
        "name": "get_employee_status",
        "description": "Bir çalışanın güncel çalışma durumunu (aktif, izinli, ücretsiz izinde, ayrıldı) döndürür.",
        "parameters": {
            "type": "object",
            "properties": {"employee_id": _p("string", EMP_ID_DESC)},
            "required": ["employee_id"],
        },
    },
    "get_departman_bilgisi": {
        "name": "get_departman_bilgisi",
        "description": "Bir departmanın özet bilgisini (yönetici, çalışan sayısı, alt ekipler) getirir.",
        "parameters": {
            "type": "object",
            "properties": {"departman_adi": _p("string", "Departmanın adı, örn. 'Yazılım Geliştirme'")},
            "required": ["departman_adi"],
        },
    },
    "get_calisan_listesi": {
        "name": "get_calisan_listesi",
        "description": "Bir departmandaki çalışanların listesini getirir; duruma göre filtrelenebilir.",
        "parameters": {
            "type": "object",
            "properties": {
                "departman_adi": _p("string", "Departmanın adı"),
                "durum": _p("string", "Çalışma durumu filtresi", enum=["aktif", "izinli", "ayrildi"]),
            },
            "required": ["departman_adi"],
        },
    },
    "get_yonetici_bilgisi": {
        "name": "get_yonetici_bilgisi",
        "description": "Bir çalışanın bağlı olduğu yöneticinin bilgisini getirir.",
        "parameters": {
            "type": "object",
            "properties": {"employee_id": _p("string", EMP_ID_DESC)},
            "required": ["employee_id"],
        },
    },
    # ---- İzin yönetimi ----------------------------------------------------
    "get_izin_bakiyesi": {
        "name": "get_izin_bakiyesi",
        "description": "Bir çalışanın izin türüne göre KALAN (kullanılabilir) izin bakiyesini getirir. İzin türü verilmezse tüm türlerin özetini döndürür.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": _p("string", EMP_ID_DESC),
                "izin_tipi": _p("string", "Sorgulanacak izin türü", enum=["yillik", "mazeret", "hastalik"]),
            },
            "required": ["employee_id"],
        },
    },
    "get_izin_gecmisi": {
        "name": "get_izin_gecmisi",
        "description": "Bir çalışanın geçmişte KULLANDIĞI izin kayıtlarını (tarih, tür, gün sayısı) listeler. Tarih aralığı verilmezse son 12 ayı döndürür.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": _p("string", EMP_ID_DESC),
                "baslangic_tarihi": _p("string", "Aralık başlangıcı, YYYY-AA-GG"),
                "bitis_tarihi": _p("string", "Aralık bitişi, YYYY-AA-GG"),
            },
            "required": ["employee_id"],
        },
    },
    "get_izin_talebi_durumu": {
        "name": "get_izin_talebi_durumu",
        "description": "Bir çalışanın izin taleplerinin onay durumunu (beklemede, onaylandı, reddedildi) getirir. talep_id verilmezse en son talebi döndürür.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": _p("string", EMP_ID_DESC),
                "talep_id": _p("string", "İzin talebinin kimliği, 'LV-2026-0001' biçiminde"),
            },
            "required": ["employee_id"],
        },
    },
    "create_izin_talebi": {
        "name": "create_izin_talebi",
        "description": "Bir çalışan için yeni izin talebi oluşturur. Değişiklik yaratan bir işlemdir; yürütülmeden önce kullanıcıdan açık onay alınmalıdır.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": _p("string", EMP_ID_DESC),
                "izin_tipi": _p("string", "İzin türü", enum=["yillik", "mazeret", "hastalik"]),
                "baslangic_tarihi": _p("string", "İzin başlangıç tarihi, YYYY-AA-GG"),
                "bitis_tarihi": _p("string", "İzin bitiş tarihi, YYYY-AA-GG"),
                "aciklama": _p("string", "İsteğe bağlı açıklama / gerekçe"),
            },
            "required": ["employee_id", "izin_tipi", "baslangic_tarihi", "bitis_tarihi"],
        },
    },
    "cancel_izin_talebi": {
        "name": "cancel_izin_talebi",
        "description": "Mevcut bir izin talebini iptal eder. Değişiklik yaratan bir işlemdir; yürütülmeden önce kullanıcıdan açık onay alınmalıdır.",
        "parameters": {
            "type": "object",
            "properties": {"talep_id": _p("string", "İptal edilecek izin talebinin kimliği, 'LV-2026-0001' biçiminde")},
            "required": ["talep_id"],
        },
    },
    "update_izin_talebi": {
        "name": "update_izin_talebi",
        "description": "Mevcut bir izin talebinin tarihlerini günceller. Değişiklik yaratan bir işlemdir; yürütülmeden önce kullanıcıdan açık onay alınmalıdır.",
        "parameters": {
            "type": "object",
            "properties": {
                "talep_id": _p("string", "Güncellenecek izin talebinin kimliği"),
                "yeni_baslangic_tarihi": _p("string", "Yeni başlangıç tarihi, YYYY-AA-GG"),
                "yeni_bitis_tarihi": _p("string", "Yeni bitiş tarihi, YYYY-AA-GG"),
            },
            "required": ["talep_id"],
        },
    },
    # ---- Mali / finansal İK ---------------------------------------------
    "get_maas_bilgisi": {
        "name": "get_maas_bilgisi",
        "description": "Bir çalışanın güncel maaş bilgisini getirir. 'tur' verilmezse hem net hem brüt döndürülür.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": _p("string", EMP_ID_DESC),
                "tur": _p("string", "Maaş türü", enum=["net", "brut"]),
            },
            "required": ["employee_id"],
        },
    },
    "get_bordro": {
        "name": "get_bordro",
        "description": "Bir çalışanın belirli bir aya ait bordro (maaş pusulası) dökümünü getirir.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": _p("string", EMP_ID_DESC),
                "donem": _p("string", "Bordro dönemi, YYYY-AA biçiminde"),
            },
            "required": ["employee_id", "donem"],
        },
    },
    "get_prim_bilgisi": {
        "name": "get_prim_bilgisi",
        "description": "Bir çalışanın prim / bonus ödemelerini getirir. Dönem verilmezse yürürlükteki yılı döndürür.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": _p("string", EMP_ID_DESC),
                "donem": _p("string", "Prim dönemi, YYYY veya YYYY-AA biçiminde"),
            },
            "required": ["employee_id"],
        },
    },
    "get_yan_haklar": {
        "name": "get_yan_haklar",
        "description": "Bir çalışanın yan haklarını (özel sağlık sigortası, yemek kartı, ulaşım, BES katkısı vb.) listeler.",
        "parameters": {
            "type": "object",
            "properties": {"employee_id": _p("string", EMP_ID_DESC)},
            "required": ["employee_id"],
        },
    },
    "create_ucret_degisiklik_talebi": {
        "name": "create_ucret_degisiklik_talebi",
        "description": "Bir çalışan için ücret değişikliği talebi oluşturur. Değişiklik yaratan hassas bir işlemdir; yürütülmeden önce kullanıcıdan açık onay alınmalıdır.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": _p("string", EMP_ID_DESC),
                "yeni_brut_ucret": _p("number", "Talep edilen yeni brüt aylık ücret"),
                "gerekce": _p("string", "Ücret değişikliği gerekçesi"),
                "gecerlilik_tarihi": _p("string", "Değişikliğin geçerli olacağı tarih, YYYY-AA-GG"),
            },
            "required": ["employee_id", "yeni_brut_ucret", "gerekce"],
        },
    },
    "create_pozisyon_degisiklik_talebi": {
        "name": "create_pozisyon_degisiklik_talebi",
        "description": "Bir çalışan için pozisyon / unvan değişikliği talebi oluşturur. Değişiklik yaratan bir işlemdir; yürütülmeden önce kullanıcıdan açık onay alınmalıdır.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": _p("string", EMP_ID_DESC),
                "yeni_pozisyon": _p("string", "Talep edilen yeni pozisyon / unvan"),
                "gerekce": _p("string", "İsteğe bağlı gerekçe"),
                "gecerlilik_tarihi": _p("string", "Geçerlilik tarihi, YYYY-AA-GG"),
            },
            "required": ["employee_id", "yeni_pozisyon"],
        },
    },
    # ---- Puantaj / çalışma --------------------------------------------
    "get_puantaj": {
        "name": "get_puantaj",
        "description": "Bir çalışanın belirli bir tarih aralığındaki puantaj kaydını (çalışılan gün, giriş/çıkış, devamsızlık) getirir.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": _p("string", EMP_ID_DESC),
                "baslangic_tarihi": _p("string", "Aralık başlangıcı, YYYY-AA-GG"),
                "bitis_tarihi": _p("string", "Aralık bitişi, YYYY-AA-GG"),
            },
            "required": ["employee_id", "baslangic_tarihi", "bitis_tarihi"],
        },
    },
    "get_mesai_bilgisi": {
        "name": "get_mesai_bilgisi",
        "description": "Bir çalışanın belirli bir aya ait fazla mesai (ek çalışma) saatlerini ve karşılığını getirir.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": _p("string", EMP_ID_DESC),
                "donem": _p("string", "Dönem, YYYY-AA biçiminde"),
            },
            "required": ["employee_id", "donem"],
        },
    },
    # ---- İletişim / kayıt güncelleme --------------------------------
    "update_employee_contact": {
        "name": "update_employee_contact",
        "description": "Bir çalışanın iletişim bilgilerini (telefon, e-posta, adres) günceller. Değişiklik yaratan bir işlemdir; yürütülmeden önce kullanıcıdan açık onay alınmalıdır.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": _p("string", EMP_ID_DESC),
                "telefon": _p("string", "Yeni telefon numarası"),
                "email": _p("string", "Yeni e-posta adresi"),
                "adres": _p("string", "Yeni ikamet adresi"),
            },
            "required": ["employee_id"],
        },
    },
    "update_employee_information": {
        "name": "update_employee_information",
        "description": "Bir çalışanın özlük kayıt bilgilerini (medeni durum, öğrenim durumu, acil durum kişisi) günceller. İletişim bilgileri için update_employee_contact kullanılır. Değişiklik yaratan bir işlemdir; yürütülmeden önce kullanıcıdan açık onay alınmalıdır.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": _p("string", EMP_ID_DESC),
                "medeni_durum": _p("string", "Yeni medeni durum", enum=["bekar", "evli", "bosanmis", "dul"]),
                "ogrenim_durumu": _p("string", "Yeni öğrenim durumu", enum=["lise", "onlisans", "lisans", "yuksek_lisans", "doktora"]),
                "acil_durum_kisisi": _p("string", "Acil durumda aranacak kişinin adı"),
                "acil_durum_telefonu": _p("string", "Acil durumda aranacak telefon numarası"),
            },
            "required": ["employee_id"],
        },
    },
    # ---- Yetki / erişim -------------------------------------------------
    "check_employee_access": {
        "name": "check_employee_access",
        "description": "Bir talep edenin, başka bir çalışanın belirli bir kaynağına (maaş, izin, iletişim, bordro, performans) erişim yetkisi olup olmadığını kontrol eder.",
        "parameters": {
            "type": "object",
            "properties": {
                "requester_id": _p("string", "Talep eden çalışanın personel kimliği"),
                "hedef_employee_id": _p("string", "Erişilmek istenen çalışanın personel kimliği"),
                "kaynak_tipi": _p("string", "Erişilmek istenen kaynak türü", enum=["maas", "izin", "iletisim", "bordro", "performans"]),
            },
            "required": ["requester_id", "hedef_employee_id", "kaynak_tipi"],
        },
    },
}

ALL_TOOL_NAMES = list(TOOLS)

# Değişiklik yaratan ve açık onay gerektiren tool'lar (politika — şemayı kirletmez).
CONFIRMATION_REQUIRED = {
    "create_izin_talebi",
    "cancel_izin_talebi",
    "update_izin_talebi",
    "update_employee_contact",
    "update_employee_information",
    "create_ucret_degisiklik_talebi",
    "create_pozisyon_degisiklik_talebi",
}
WRITE_TOOLS = set(CONFIRMATION_REQUIRED)

# Anlamca yakın (karıştırılabilir) tool'lar — distractor seçimi için (bkz. prompt §16).
CONFUSABLE = {
    "get_izin_bakiyesi": ["get_izin_gecmisi", "get_izin_talebi_durumu", "get_yan_haklar", "create_izin_talebi"],
    "get_izin_gecmisi": ["get_izin_bakiyesi", "get_izin_talebi_durumu", "get_puantaj"],
    "get_izin_talebi_durumu": ["get_izin_gecmisi", "get_izin_bakiyesi", "create_izin_talebi", "cancel_izin_talebi"],
    "create_izin_talebi": ["update_izin_talebi", "cancel_izin_talebi", "get_izin_bakiyesi"],
    "cancel_izin_talebi": ["update_izin_talebi", "create_izin_talebi", "get_izin_talebi_durumu"],
    "update_izin_talebi": ["cancel_izin_talebi", "create_izin_talebi", "get_izin_talebi_durumu"],
    "get_maas_bilgisi": ["get_bordro", "get_prim_bilgisi", "get_yan_haklar"],
    "get_bordro": ["get_maas_bilgisi", "get_prim_bilgisi", "get_mesai_bilgisi"],
    "get_prim_bilgisi": ["get_maas_bilgisi", "get_bordro", "get_yan_haklar"],
    "get_yan_haklar": ["get_maas_bilgisi", "get_prim_bilgisi", "get_bordro"],
    "get_puantaj": ["get_mesai_bilgisi", "get_izin_gecmisi", "get_employee_status"],
    "get_mesai_bilgisi": ["get_puantaj", "get_bordro", "get_prim_bilgisi"],
    "get_employee_info": ["get_employee_status", "get_yonetici_bilgisi", "get_departman_bilgisi"],
    "get_employee_status": ["get_employee_info", "get_puantaj", "get_izin_talebi_durumu"],
    "get_yonetici_bilgisi": ["get_employee_info", "get_departman_bilgisi", "get_calisan_listesi"],
    "get_departman_bilgisi": ["get_calisan_listesi", "get_yonetici_bilgisi", "get_employee_info"],
    "get_calisan_listesi": ["get_departman_bilgisi", "get_yonetici_bilgisi", "get_employee_status"],
    "create_ucret_degisiklik_talebi": ["get_maas_bilgisi", "create_pozisyon_degisiklik_talebi", "get_prim_bilgisi"],
    "create_pozisyon_degisiklik_talebi": ["create_ucret_degisiklik_talebi", "get_employee_info", "get_yonetici_bilgisi"],
    "update_employee_contact": ["get_employee_info", "update_employee_information", "update_izin_talebi"],
    "update_employee_information": ["update_employee_contact", "get_employee_info", "get_employee_status"],
    "check_employee_access": ["get_employee_info", "get_maas_bilgisi", "get_izin_bakiyesi"],
}


# ---------------------------------------------------------------------------
# 2. SLOT DEĞER HAVUZLARI  (hepsi sentetik)
# ---------------------------------------------------------------------------

DEPARTMANLAR = [
    "Yazılım Geliştirme", "İnsan Kaynakları", "Muhasebe", "Satış", "Pazarlama",
    "Operasyon", "Hukuk", "Bilgi Teknolojileri", "Finans", "Müşteri Deneyimi",
    "Ar-Ge", "Lojistik", "Kalite Güvence", "Satın Alma", "İç Denetim",
]
# departman yüzey biçimi -> kanonik ad
DEPARTMAN_YUZEY = {
    "yazılım ekibi": "Yazılım Geliştirme", "yazılım geliştirme": "Yazılım Geliştirme",
    "yazılım departmanı": "Yazılım Geliştirme", "yazılımcılar": "Yazılım Geliştirme",
    "İK": "İnsan Kaynakları", "IK": "İnsan Kaynakları", "insan kaynakları": "İnsan Kaynakları",
    "İK departmanı": "İnsan Kaynakları", "muhasebe": "Muhasebe", "muhasebe birimi": "Muhasebe",
    "satış ekibi": "Satış", "satış departmanı": "Satış", "pazarlama": "Pazarlama",
    "operasyon": "Operasyon", "operasyon ekibi": "Operasyon", "hukuk": "Hukuk",
    "hukuk müşavirliği": "Hukuk", "BT": "Bilgi Teknolojileri", "bilgi teknolojileri": "Bilgi Teknolojileri",
    "finans": "Finans", "finans ekibi": "Finans", "müşteri deneyimi": "Müşteri Deneyimi",
    "CX ekibi": "Müşteri Deneyimi", "Ar-Ge": "Ar-Ge", "arge": "Ar-Ge", "lojistik": "Lojistik",
    "kalite güvence": "Kalite Güvence", "QA ekibi": "Kalite Güvence", "satın alma": "Satın Alma",
    "iç denetim": "İç Denetim", "pazarlama ekibi": "Pazarlama", "pazarlama departmanı": "Pazarlama",
    "muhasebe departmanı": "Muhasebe", "lojistik ekibi": "Lojistik", "satınalma": "Satın Alma",
    "denetim": "İç Denetim", "IT": "Bilgi Teknolojileri", "bt ekibi": "Bilgi Teknolojileri",
    "hukuk ekibi": "Hukuk", "operasyon departmanı": "Operasyon", "finans departmanı": "Finans",
}

POZISYONLAR = [
    "Kıdemli Yazılım Mühendisi", "Takım Lideri", "Uzman", "Kıdemli Uzman",
    "Müdür Yardımcısı", "Proje Yöneticisi", "İş Analisti", "Veri Analisti",
    "Ürün Yöneticisi", "Kıdemli Uzman Yardımcısı", "Ekip Koordinatörü",
    "Teknik Lider", "Baş Uzman", "Departman Müdürü", "Kıdemli İş Analisti",
    "Yazılım Mühendisi",
]

# izin türü yüzey biçimi -> kanonik enum (yüzeyler "izin/izni" eki İÇERMEZ)
IZIN_TIPI_YUZEY = {
    "yıllık": "yillik", "senelik": "yillik", "yillik": "yillik", "yıllik": "yillik",
    "mazeret": "mazeret", "mazerert": "mazeret",
    "hastalık": "hastalik", "hastalik": "hastalik", "sağlık": "hastalik", "rapor": "hastalik",
}
IZIN_TIPI_DISP = {"yillik": "yıllık izin", "mazeret": "mazeret izni", "hastalik": "hastalık izni"}

# (yüzey dönem, kanonik YYYY-AA)
DONEMLER = [
    ("Ağustos 2026", "2026-08"), ("Temmuz 2026", "2026-07"), ("Haziran 2026", "2026-06"),
    ("Mayıs 2026", "2026-05"), ("Nisan 2026", "2026-04"), ("Mart 2026", "2026-03"),
    ("Şubat 2026", "2026-02"), ("Ocak 2026", "2026-01"), ("Aralık 2025", "2025-12"),
    ("Kasım 2025", "2025-11"), ("2026 Eylül", "2026-09"), ("2026 Temmuz", "2026-07"),
    ("09/2026", "2026-09"), ("2026-07", "2026-07"), ("Ekim 2025", "2025-10"),
    ("Eylül 2025", "2025-09"), ("2026 Mayıs", "2026-05"), ("Nisan 2026 ayı", "2026-04"),
    ("07/2026", "2026-07"), ("2026-03", "2026-03"), ("Şubat 2026 dönemi", "2026-02"),
]
DONEM_YIL = [("2026", "2026"), ("2025", "2025"), ("bu yıl", "2026"), ("geçen yıl", "2025")]

# (yüzey aralık, kanonik başlangıç, kanonik bitiş)  — hepsi açık / netleştirilmiş
DATE_RANGES = [
    ("15-20 Eylül 2026", "2026-09-15", "2026-09-20"),
    ("3 Ekim 2026 ile 7 Ekim 2026 arası", "2026-10-03", "2026-10-07"),
    ("1 Eylül - 12 Eylül 2026", "2026-09-01", "2026-09-12"),
    ("22 Eylül 2026 ve 26 Eylül 2026", "2026-09-22", "2026-09-26"),
    ("14/10/2026 - 18/10/2026", "2026-10-14", "2026-10-18"),
    ("5 Kasım 2026 başlangıçlı 3 günlük", "2026-11-05", "2026-11-07"),
    ("28 Ağustos - 1 Eylül 2026", "2026-08-28", "2026-09-01"),
    ("10 Aralık 2026 ile 24 Aralık 2026", "2026-12-10", "2026-12-24"),
    ("2026-09-07 / 2026-09-09", "2026-09-07", "2026-09-09"),
    ("6-8 Ekim 2026", "2026-10-06", "2026-10-08"),
    ("2 Kasım 2026 - 6 Kasım 2026", "2026-11-02", "2026-11-06"),
    ("17 ile 21 Kasım 2026 arası", "2026-11-17", "2026-11-21"),
    ("30 Eylül 2026 başlangıçlı 4 günlük", "2026-09-30", "2026-10-03"),
    ("7-11 Aralık 2026", "2026-12-07", "2026-12-11"),
    ("23/09/2026 ile 25/09/2026", "2026-09-23", "2026-09-25"),
    ("1 Ekim - 3 Ekim 2026", "2026-10-01", "2026-10-03"),
    ("12 Kasım 2026 ve 13 Kasım 2026", "2026-11-12", "2026-11-13"),
    ("19-26 Ekim 2026", "2026-10-19", "2026-10-26"),
]
# puantaj / geçmiş için ay aralıkları
MONTH_RANGES = [
    ("1 Temmuz 2026 - 31 Temmuz 2026", "2026-07-01", "2026-07-31"),
    ("Haziran 2026", "2026-06-01", "2026-06-30"),
    ("Mayıs 2026 ayı", "2026-05-01", "2026-05-31"),
    ("2026 ikinci çeyrek", "2026-04-01", "2026-06-30"),
    ("Ocak 2026 - Mart 2026", "2026-01-01", "2026-03-31"),
    ("Ağustos 2026", "2026-08-01", "2026-08-31"),
    ("Nisan 2026", "2026-04-01", "2026-04-30"),
    ("2026 ilk çeyrek", "2026-01-01", "2026-03-31"),
    ("Temmuz 2026 - Eylül 2026", "2026-07-01", "2026-09-30"),
    ("1 Şubat 2026 - 28 Şubat 2026", "2026-02-01", "2026-02-28"),
    ("Mart 2026 ayı", "2026-03-01", "2026-03-31"),
    ("Ekim 2025 - Aralık 2025", "2025-10-01", "2025-12-31"),
]

# Belirsiz tarih ifadeleri (request_for_info tetikler)
VAGUE_DATES = [
    "önümüzdeki hafta", "gelecek ay", "birkaç gün", "yakında bir ara",
    "bahar aylarında", "yaz döneminde", "bir haftalığına", "yılbaşı civarı",
    "çocukların tatili boyunca", "önümüzdeki dönem",
]

TALEP_IDS = [
    "LV-2026-0148", "LV-2026-0293", "LV-2026-0501", "LV-2026-0677", "LV-2026-0812",
    "LV-2025-1140", "LV-2026-0934", "LV-2026-1057", "LV-2026-0206", "LV-2026-0745",
]

FIRST_NAMES = [
    "Ahmet", "Mehmet", "Ayşe", "Elif", "Can", "Deniz", "Zeynep", "Burak", "Seda",
    "Emre", "Merve", "Kaan", "Selin", "Onur", "Gizem", "Barış", "Ece", "Tolga",
    "Pınar", "Serkan", "Büşra", "Yusuf", "Nazlı", "Cem",
]


def emp_id(rng) -> str:
    return f"EMP-{rng.randint(1000, 6999)}"


def emp_ref_forms(eid: str) -> list[str]:
    """eid metin içinde nasıl geçebilir — hepsi aynı kanonik eid'e çözülür."""
    num = eid.split("-")[1]
    return [
        eid,
        f"{eid} numaralı çalışan",
        f"{num} numaralı çalışan",
        f"{num} numaralı personel",
        f"personel {eid}",
        f"çalışan {eid}",
        f"{eid} kodlu personel",
        f"sicil no {num}",
    ]


# ---------------------------------------------------------------------------
# 3. YARDIMCILAR
# ---------------------------------------------------------------------------

def tr_fold(s: str) -> str:
    s = s.replace("İ", "i").replace("I", "ı")
    s = s.lower()
    for a, b in (("ı", "i"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("ç", "c")):
        s = s.replace(a, b)
    return s


def norm_sig(text: str) -> str:
    """Yakın-kopya tespiti için imza: harf-dışı, rakam ve ID'ler silinir."""
    t = tr_fold(text)
    t = re.sub(r"emp-?\d+", " ", t)
    t = re.sub(r"lv-?\d[\d-]*", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def resolve_relative_period(term: str, today: date) -> str | None:
    first_this = today.replace(day=1)
    if term in ("bu ay", "içinde bulunduğumuz ay"):
        return f"{today:%Y-%m}"
    if term in ("geçen ay", "geçen ayki", "önceki ay"):
        prev = first_this - timedelta(days=1)
        return f"{prev:%Y-%m}"
    if term in ("bu yıl",):
        return f"{today:%Y}"
    if term in ("geçen yıl", "geçen sene"):
        return f"{today.year - 1}"
    return None


def build_tools_list(rng, targets: list[str], k_min=4, k_max=8) -> list[dict]:
    """Doğru tool(lar) + aynı alandan distractor'lar, karıştırılmış."""
    k = rng.randint(k_min, k_max)
    chosen: list[str] = []
    for t in targets:
        if t not in chosen:
            chosen.append(t)
    # önce karıştırılabilir komşular
    neigh: list[str] = []
    for t in targets:
        neigh.extend(CONFUSABLE.get(t, []))
    rng.shuffle(neigh)
    for n in neigh:
        if len(chosen) >= k:
            break
        if n not in chosen:
            chosen.append(n)
    # sonra rastgele
    rest = [n for n in ALL_TOOL_NAMES if n not in chosen]
    rng.shuffle(rest)
    for n in rest:
        if len(chosen) >= k:
            break
        chosen.append(n)
    rng.shuffle(chosen)
    return [TOOLS[n] for n in chosen]


def _lower_first(q: str) -> str:
    """İlk kelimeyi küçült — ama ID / kısaltma (EMP-1042, İK) bozma."""
    if not q:
        return q
    head = q.split(" ", 1)[0]
    if re.search(r"\d", head) or (head.isupper() and len(head) > 1):
        return q
    return q[0].lower() + q[1:]


def _strip_end(q: str) -> str:
    return q.rstrip(" ?.!")


UZUN_ONEK = [
    "Önümüzdeki dönem için planlama yapıyorum.",
    "Yöneticimle görüşmeden önce bir şeyi netleştirmem gerekiyor.",
    "Muhasebeyle bir konuşma yapacağım, ondan önce kontrol etmek istedim.",
    "Kafam biraz karıştı, yardımcı olabilir misin.",
    "Bu ay birkaç işi toparlamaya çalışıyorum.",
    "Aylık kapanış öncesi son bir kontrol yapıyorum.",
    "Sabahtan beri bununla uğraşıyorum, bir türlü emin olamadım.",
    "Bir toplantıya gireceğim, öncesinde şunu teyit etmem lazım.",
    "İK portalında bulamadım, o yüzden buradan soruyorum.",
    "Eşimle tatil planı yapıyoruz, ona göre karar vereceğiz.",
]
UZUN_SON = [
    "Buna göre ilerleyeceğim, teşekkürler.",
    "Detayları paylaşabilirsen sevinirim.",
    "Acele etmiyorum ama bugün içinde lazım.",
    "Yanlış bir şey yapmak istemiyorum, o yüzden soruyorum.",
    "Mümkünse bugün kapatmak istiyorum bu işi.",
    "Teyit alınca rahatlayacağım.",
]
KONUSMA_ONEK = ["Ya ", "Abi ", "Şuna bi baksana, ", "Bir saniye, ", "Pardon ya, ",
                "Hocam ", "Bak şimdi, ", "Şey, "]
KONUSMA_SON = [" ya", " bi bakar mısın", " acaba", ", ne dersin", " hemen lazım",
               " bu arada", ", olur mu"]
RESMI_ONEK = [
    "Sayın yetkili, ",
    "İlgili birime iletilmek üzere: ",
    "Bilgi talebi — ",
    "Merhaba, aşağıdaki hususta bilgilendirilmek istiyorum: ",
    "Konu: bilgi talebi. ",
    "İK birimine, ",
]
RESMI_SON = [
    " Gereğini rica ederim.",
    " Yardımlarınız için şimdiden teşekkür ederim.",
    " Bilgilerinize arz ederim.",
    " İyi çalışmalar dilerim.",
    " Saygılarımla.",
    " Konuyla ilgilenmenizi rica ederim.",
]

REGISTER_WEIGHTS = [
    (None, 42), ("resmi", 16), ("konusma_dili", 14), ("uzun", 14), ("yazim_hatali", 14),
]


def _typo(rng, q: str) -> str:
    # En yaygın Türkçe "yazım hatası": diyakritikleri düşürüp küçük harfle yazmak,
    # noktalama atlamak, "mısın" gibi ekleri bitişik yazmak. ID / tarih / tutar
    # token'ları BOZULMAZ (halüsinasyon riskini sıfırda tutmak için).
    q = tr_fold(q)
    q = q.replace("?", "")
    q = re.sub(r"\.(\s|$)", r"\1", q)  # yalnızca cümle sonu noktası — "example.com" bozulmaz
    if rng.random() < 0.55:
        q = re.sub(r"\bm[iı]s[iı]n\b", "misin", q)
        q = q.replace(" mi ", " mi ").replace("bakar misin", "bakarmisin")
    if rng.random() < 0.25:
        q = q.replace(" de ", " de").replace(" ki ", " ki ")
    return q.strip()


def style_user_text(rng, text: str, intent: str):
    """Aynı niyeti farklı dil kaydında ifade et. (styled_text, register|None) döner."""
    if intent in NO_FRAME_INTENTS:
        return text, None
    reg = rng.choices([r for r, _ in REGISTER_WEIGHTS], weights=[w for _, w in REGISTER_WEIGHTS])[0]
    if reg is None:
        return text, None
    if reg == "resmi":
        return rng.choice(RESMI_ONEK) + _lower_first(_strip_end(text)) + "." + rng.choice(RESMI_SON), "resmi"
    if reg == "konusma_dili":
        if rng.random() < 0.5:
            return rng.choice(KONUSMA_ONEK) + _lower_first(text), "konusma_dili"
        return _strip_end(text) + rng.choice(KONUSMA_SON) + "?", "konusma_dili"
    if reg == "uzun":
        core = text if text[-1:] in ".?!" else text + "."
        core = core[0].upper() + core[1:]
        return f"{rng.choice(UZUN_ONEK)} {core} {rng.choice(UZUN_SON)}", "uzun"
    if reg == "yazim_hatali":
        return _typo(rng, text), "yazim_hatali"
    return text, None


# Paylaşılan cevap havuzları -------------------------------------------------

REFUSAL_OPEN = [
    "", "", "Ne yazık ki ", "Maalesef ", "Üzgünüm, ", "Kusura bakmayın ama ",
    "Bu konuda yardımcı olamıyorum. ", "Açık olmak gerekirse, ",
]

REFUSAL_CORE = {
    "plain": [
        "bu bilgi mevcut Büyük İK araçlarının kapsamına girmiyor",
        "mevcut sistem yetenekleriyle bu soruya güvenilir bir yanıt üretemiyorum",
        "Büyük İK üzerinden bu veriye erişemiyorum; bunu karşılayan bir aracım yok",
        "bu işlem için tanımlı bir Büyük İK aracı bulunmuyor",
        "mevcut İK servisleri bu bilgiyi sağlamıyor",
        "bu, Büyük İK asistanının yapabilecekleri arasında değil",
        "bu konuda doğrulayabileceğim bir veri kaynağı yok, o yüzden tahminde bulunmuyorum",
        "bu talep sistemde desteklenen işlemlerin dışında",
        "elimdeki araçlar yalnızca İK kayıtlarını sorgulayabiliyor, bu bunlardan biri değil",
        "bu isteği yerine getirebilecek bir yeteneğim yok",
    ],
    "future": [
        "bu geleceğe dönük ve doğrulanamaz bir bilgi; Büyük İK araçları böyle bir tahmin üretmiyor",
        "ileri tarihli kesin bir öngörü sunamam; sistemde bunu destekleyen bir veri yok",
        "gelecekte ne olacağını kestiremem; araçlarım yalnızca mevcut ve geçmiş kayıtları getiriyor",
        "bu tür bir tahmin spekülasyon olur ve yanlış yönlendirmemek için bunu yapmıyorum",
        "henüz gerçekleşmemiş bir durumu kesin olarak bildiremem",
    ],
    "privacy": [
        "başka bir çalışanın bu bilgisini paylaşamam; kişisel ve mali veriler yalnızca ilgili kişiye ve yetkili İK ekibine açıktır",
        "bu, başkasına ait gizli bir bilgi ve yetkinizi doğrulayan bir süreç olmadan bunu getiremem",
        "çalışanların birbirinin maaş, izin veya iletişim bilgilerine erişimi yoktur",
        "bu veriye erişim özel yetki gerektiriyor; bu asistan üzerinden başkasının özel bilgisini açamam",
        "kişisel verilerin korunması gereği bu bilgiyi üçüncü bir kişiyle paylaşamam",
        "bu bilgi yalnızca sahibinin görebileceği bir kayıt",
    ],
    "career": [
        "bu kişisel bir karar ve Büyük İK asistanı olarak bu yönde tavsiye vermem doğru olmaz",
        "kariyer ve istihdam kararları için yönlendirme yapmam uygun değil",
        "hangi yolun sizin için doğru olduğunu ben söyleyemem",
        "bu konuda sizin yerinize karar veremem",
    ],
    "financial": [
        "yatırım veya finansal tavsiye veremem",
        "paranızı nasıl değerlendireceğiniz konusunda yönlendirme yapmam uygun değil",
        "bu bir yatırım kararı ve bu konuda tavsiye vermem doğru olmaz",
    ],
    "unsupported": [
        "bu işlemi bu asistan üzerinden yapamam; onay/işlem yetkisi bende değil",
        "bu tür bir işlem Büyük İK araçlarında tanımlı değil, dolayısıyla gerçekleştiremem",
        "izin taleplerini onaylamak, kayıt silmek veya toplu işlem yapmak yetki alanımın dışında",
        "bu değişikliği yapacak bir aracım yok; ilgili adım İK / yönetici tarafında yürür",
        "böyle bir toplu işlemi yürütemem; her talep kendi onay akışından geçer",
        "bu isteği yerine getirebilecek bir yeteneğim yok",
    ],
}

REFUSAL_REDIRECT = {
    "plain": ["", "", " Yardımcı olabileceğim bir İK konusu varsa memnuniyetle bakarım.",
              " İzin, maaş, bordro veya puantaj konularında destek verebilirim."],
    "future": ["", " Dilerseniz mevcut kayıtlara bakabilirim.", " Bugünkü verilerle ilgili sorularınıza yanıt verebilirim."],
    "privacy": ["", " Kendi bilgilerinizi sorarsanız yardımcı olabilirim.",
                " Yetki durumunuzu birlikte kontrol etmemi isterseniz söyleyin."],
    "career": [" Bu konuyu İK iş ortağınızla veya yöneticinizle görüşmeniz daha doğru olur.",
               " Kariyer hedeflerinizi İK ekibiyle konuşabilirsiniz.", ""],
    "financial": [" Yetkili bir finansal danışmana başvurmanızı öneririm.",
                  " Bu tür kararlar için bir uzmana danışmanız daha doğru olur.", ""],
    "unsupported": ["", " İzin, maaş, bordro veya puantaj sorgularında yardımcı olabilirim.",
                    " Talebinizi doğru kanaldan (yöneticiniz / İK ekibi) iletmeniz gerekir.",
                    " Bilgi sorgusu yapabilir ya da onayınızla izin talebi oluşturabilirim."],
}


def refusal_text(rng, pool: str) -> str:
    core = rng.choice(REFUSAL_CORE[pool])
    op = rng.choice(REFUSAL_OPEN)
    red = rng.choice(REFUSAL_REDIRECT[pool])
    cap = (not op) or op.rstrip().endswith((".", "!", "?"))
    if cap:
        core = core[0].upper() + core[1:]
    s = op + core
    if not s.rstrip().endswith((".", "!", "?")):
        s = s.rstrip() + "."
    return (s + red).strip()

ACK_WORDS = [
    "Evet.", "Onaylıyorum.", "Evet, onaylıyorum.", "Tamam, devam et.", "Olur, yap.",
    "Evet lütfen oluştur.", "Devam edebilirsin.", "Evet bunu uygula.", "Onaylıyorum, ilerle.",
    "Tabii, işleme al.", "Evet, doğru. Devam et.", "Uygundur, yapabilirsin.",
]

CONFIRM_ASKS = [
    "Bu işlemi belirttiğiniz bilgilerle yapmamı onaylıyor musunuz?",
    "Devam etmemi ister misiniz?",
    "Bu değişikliği şimdi uygulamam uygun mu?",
    "İşlemi tamamlamam için onayınızı paylaşır mısınız?",
    "Aşağıdaki bilgilerle devam etmemi onaylıyor musunuz?",
    "Bunu bu şekilde kaydetmemi ister misiniz?",
]


# ---------------------------------------------------------------------------
# 4. DIRECT — tool gerekmeyen sorular (tanım / politika / süreç / meta)
# ---------------------------------------------------------------------------

DIRECT_INTENTS: list[dict] = [
    {
        "intent": "define_annual_leave", "domain": "izin_yonetimi", "difficulty": "kolay",
        "q": [
            "Yıllık izin nedir?",
            "Yıllık ücretli izin ne demek, biraz açıklar mısın?",
            "yillik izin tam olarak neyi kapsiyor",
            "İK terimi olarak 'yıllık izin' neyi ifade eder?",
            "Yıllık izin kavramını kısaca anlatır mısın?",
            "Şirkette yıllık izin denince ne kastediliyor?",
            "Yeni başladım, yıllık izin olayını tam anlamadım; nedir bu?",
            "Yıllık izin hakkı dediğimiz şey nedir?",
        ],
        "a": [
            "Yıllık izin, çalışanın dinlenmesi için her yıl hak ettiği ücretli izindir. Süresi kıdeme göre değişir; genellikle önceden planlanıp yöneticinin onayıyla kullanılır ve kullanıldıkça izin bakiyenizden düşülür.",
            "Yıllık ücretli izin, iş yerinde belirli bir süreyi (genelde bir yılı) doldurmuş çalışanın dinlenme amacıyla kullanabildiği, ücretinin kesilmediği izindir. Hak edilen gün sayısı çalışma süresine göre artar.",
            "Kısaca: çalıştığınız her yıl karşılığında kazandığınız, maaşınız ödenmeye devam ederken kullanabileceğiniz dinlenme iznidir. Ne kadarının kaldığını izin bakiyenizden takip edebilirsiniz.",
        ],
    },
    {
        "intent": "define_excuse_leave", "domain": "izin_yonetimi", "difficulty": "kolay",
        "q": [
            "Mazeret izni nedir?",
            "Mazeret izni hangi durumlarda kullanılır?",
            "mazeret izni ne ise yariyor",
            "Mazeret izni derken tam olarak neyi kastediyoruz?",
            "Hangi hallerde mazeret izni verilir?",
            "Mazeret izni kavramını açıklar mısın?",
            "Mazeret izni ile ilgili genel bilgi verir misin?",
        ],
        "a": [
            "Mazeret izni, evlilik, çocuğun doğumu, birinci derece yakının vefatı, taşınma gibi belirli hayat olayları için verilen kısa süreli izindir. Yıllık izinden ayrı tutulur ve olayına özgü olarak tanımlanır.",
            "Mazeret izni; doğum, evlilik, ölüm, tabii afet gibi öngörülemeyen ya da özel durumlar için kullanılan izindir. Genellikle birkaç günle sınırlıdır ve belgelenmesi istenebilir.",
            "Belirli mazeret hallerinde (evlilik, doğum, yakın kaybı vb.) çalışana tanınan, yıllık izin bakiyesini etkilemeyen özel izindir. Şirket politikanıza göre süresi değişebilir.",
        ],
    },
    {
        "intent": "define_sick_leave", "domain": "izin_yonetimi", "difficulty": "kolay",
        "q": [
            "Hastalık izni nedir?",
            "Raporlu izin ne anlama geliyor?",
            "hastalik izni nasil isliyor genel olarak",
            "Sağlık iznini kısaca açıklar mısın?",
            "Hastalık izni ile rapor aynı şey mi, nedir bunlar?",
            "İşe gelemeyecek kadar hastayken kullanılan izne ne deniyor, nasıl bir şey?",
        ],
        "a": [
            "Hastalık izni, çalışanın sağlık sorunu nedeniyle çalışamadığı ve genellikle bir sağlık raporuna dayanan izindir. Rapor süresi boyunca devamsızlık, izin olarak kaydedilir ve ilgili mevzuata göre ödeme yapılır.",
            "Sağlık/hastalık izni, doktor raporuyla belgelenen ve çalışanın iyileşene kadar işe gelmemesini sağlayan izindir. Raporun İK'ya iletilmesi gerekir.",
            "Halk arasında 'rapor' denen şey budur: hekim tarafından verilen istirahat süresince çalışanın işe gelmemesi izin olarak işlenir.",
        ],
    },
    {
        "intent": "difference_leave_types", "domain": "izin_yonetimi", "difficulty": "orta",
        "q": [
            "Yıllık izin ile mazeret izni arasındaki fark nedir?",
            "Mazeret izni ile hastalık izni arasında ne fark var?",
            "yillik izin ve mazeret izni farki nedir",
            "Yıllık izinle raporlu izni birbirinden ayıran şey ne?",
            "Mazeret izni mi yıllık izin mi; ikisi arasındaki temel ayrım nedir?",
            "Hastalık izni ile mazeret izni karıştırıyorum, farkları neler?",
            "İzin türleri arasındaki farkı özetler misin?",
        ],
        "a": [
            "Yıllık izin planlı dinlenme için kullanılır ve izin bakiyenizden düşer. Mazeret izni evlilik/doğum/vefat gibi belirli olaylara bağlıdır, olayına özgüdür ve yıllık bakiyeyi etkilemez. Hastalık izni ise sağlık raporuna dayanır. Üçü de ücretli olabilir ama gerekçe ve kayıt biçimleri farklıdır.",
            "Temel ayrım gerekçede: yıllık izin bir haktır ve istediğiniz zaman (onayla) kullanırsınız; mazeret izni yalnızca tanımlı durumlarda açılır; hastalık izni raporla belgelenir. Yalnızca yıllık izin bakiyeden düşer.",
            "Yıllık izin = birikmiş dinlenme hakkı. Mazeret izni = özel hayat olaylarına bağlı kısa izin. Hastalık izni = doktor raporlu istirahat. Kaynağı ve bakiyeye etkisi bu yüzden farklıdır.",
        ],
    },
    {
        "intent": "leave_request_process", "domain": "ik_islemleri", "difficulty": "orta",
        "q": [
            "İzin talebini nasıl oluştururum?",
            "İzin başvurusu adımları nedir?",
            "izin almak icin ne yapmam lazim",
            "Yıllık izne çıkmak istiyorum, süreç nasıl işliyor?",
            "İzin talebi verirken hangi bilgileri girmem gerekiyor?",
            "İzin nasıl talep edilir, kısaca anlatır mısın?",
            "İzin isteyeceğim ama prosedürü bilmiyorum; nasıl bir yol izlemeliyim?",
        ],
        "a": [
            "Genel akış şöyle: izin türünü seçersiniz, başlangıç ve bitiş tarihlerini girersiniz, gerekiyorsa kısa bir açıklama eklersiniz ve talebi gönderirsiniz. Talep yöneticinize onaya düşer; onaylandığında izniniz kesinleşir. İsterseniz gerekli bilgileri verin, talebi sizin için oluşturabilirim.",
            "İzin türü, tarih aralığı ve (varsa) açıklama ile talep açılır; ardından yönetici onayına gider. Onaydan sonra izin bakiyeniz güncellenir. Tarihleri ve türü söylerseniz talebi başlatabilirim.",
            "Önce hangi izni (yıllık, mazeret, hastalık) kullanacağınıza karar verin, sonra net tarihleri belirleyin ve talebi iletin. Onay zinciri tamamlanınca izin geçerli olur.",
        ],
    },
    {
        "intent": "leave_accrual", "domain": "izin_yonetimi", "difficulty": "orta",
        "q": [
            "Yıllık izin nasıl hak ediliyor?",
            "İzin günleri neye göre birikiyor?",
            "yillik izin hakedisi nasil hesaplaniyor",
            "Kıdem arttıkça izin günü artıyor mu, nasıl bir mantığı var?",
            "Yeni işe girenin ne zaman izin hakkı oluşur?",
            "İzin hakkı kaç ayda bir işliyor?",
        ],
        "a": [
            "Yıllık izne hak kazanmak için genellikle iş yerinde bir yılı doldurmak gerekir. Sonrasında her yıl için mevzuatta ve şirket politikanızda belirlenen gün sayısı bakiyenize eklenir; kıdem yükseldikçe bu sayı artar.",
            "İzin, kıdeme bağlı olarak yıllık dilimler halinde tanımlanır. İlk yıl dolduğunda ilk hak doğar; ilerleyen yıllarda çalışma süresine göre gün sayısı kademeli artar.",
            "Kabaca: bir yıl çalışınca ilk izin hakkınız açılır, sonraki her tam yıl için yeni gün eklenir. Kesin sayılar şirketinizin İK politikasına göre değişir.",
        ],
    },
    {
        "intent": "leave_carryover", "domain": "izin_yonetimi", "difficulty": "orta",
        "q": [
            "Kullanmadığım yıllık izin bir sonraki yıla devrediyor mu?",
            "İzin devri nasıl oluyor?",
            "artan izinler yaniyor mu",
            "Bu yılki iznimin bir kısmını kullanamazsam ne olur?",
            "Yıllık izin sonraki seneye aktarılır mı?",
        ],
        "a": [
            "Genel kural olarak kullanılmayan yıllık izin yanmaz; sonraki döneme devredilebilir. Ancak birçok şirket, birikmeyi sınırlamak için devredilebilecek gün sayısına veya kullanım süresine bir tavan koyar. Kendi durumunuz için İK politikanıza bakmanız gerekir.",
            "Kullanılmayan izin çoğunlukla devreder, fakat şirketler 'şu tarihe kadar kullanılmalı' gibi kurallar getirebilir. Politika detayı şirkete göre değişir.",
        ],
    },
    {
        "intent": "gross_net_salary", "domain": "maas_finans", "difficulty": "kolay",
        "q": [
            "Brüt maaş ile net maaş arasındaki fark nedir?",
            "brut net maas farki ne demek",
            "Bordroda brüt ve net neden farklı çıkıyor?",
            "Net maaş nasıl hesaplanıyor, brütten ne kesiliyor?",
            "Brüt maaş nedir, net maaş nedir?",
            "Maaşın brütü ile eline geçeni arasında niye fark oluyor?",
        ],
        "a": [
            "Brüt maaş, işverenin sizin için tanımladığı yasal kesintiler öncesi tutardır. Net maaş ise brütten SGK primi, işsizlik sigortası, gelir vergisi ve damga vergisi düşüldükten sonra elinize geçen tutardır.",
            "Brüt = kesinti öncesi ücret. Net = brütten sigorta ve vergi kesintileri çıktıktan sonra ödenen tutar. Aradaki fark bu yasal kesintilerdir.",
            "Brüt sözleşmedeki ana rakam, net ise banka hesabınıza yatan rakamdır; ikisi arasındaki fark SGK ve vergi kalemlerinden gelir.",
        ],
    },
    {
        "intent": "define_bordro", "domain": "maas_finans", "difficulty": "kolay",
        "q": [
            "Bordro nedir?",
            "Maaş pusulası neyi gösterir?",
            "bordro ne ise yarar",
            "Bordroda hangi bilgiler yer alır?",
            "Bordro dediğimiz belge tam olarak nedir?",
            "Bordromu açtım ama ne anlama geldiğini bilmiyorum; genel olarak nedir bu?",
        ],
        "a": [
            "Bordro, bir aya ait ücret hesabınızın resmi dökümüdür. Brüt ücret, SGK ve işsizlik primleri, gelir ve damga vergisi, varsa prim/kesinti kalemleri ile net ödenen tutarı satır satır gösterir.",
            "Maaş pusulası da denir; o ayki kazançlarınızı, yasal kesintileri ve elinize geçen net tutarı gösteren belgedir. Çalışılan gün ve varsa fazla mesai de burada görünür.",
            "Bordro, maaşınızın nasıl hesaplandığını gösteren aylık belgedir: kazançlar, kesintiler ve net ödeme bir arada listelenir.",
        ],
    },
    {
        "intent": "define_sgk_prim", "domain": "maas_finans", "difficulty": "orta",
        "q": [
            "SGK primi nedir, maaştan neden kesiliyor?",
            "sgk kesintisi ne demek",
            "Bordrodaki SGK işçi payı nedir?",
            "Sosyal güvenlik primi ne işe yarıyor?",
            "Maaşımdan SGK adı altında kesilen tutar ne anlama geliyor?",
        ],
        "a": [
            "SGK primi, emeklilik, sağlık ve iş kazası gibi sosyal güvenlik haklarını finanse eden zorunlu kesintidir. Bir kısmı çalışandan (işçi payı) maaştan kesilir, bir kısmını işveren ayrıca öder.",
            "Sosyal Güvenlik Kurumu'na ödenen bu prim, sağlık hizmetleri ve emeklilik birikiminizin karşılığıdır. Çalışan payı brüt ücret üzerinden hesaplanıp bordroda gösterilir.",
        ],
    },
    {
        "intent": "define_income_tax_cut", "domain": "maas_finans", "difficulty": "orta",
        "q": [
            "Maaştan kesilen gelir vergisi nasıl belirleniyor?",
            "gelir vergisi dilimi ne demek maasta",
            "Yıl içinde net maaşım neden düşüyor?",
            "Gelir vergisi kesintisi neye göre artıyor?",
            "Bordrodaki gelir vergisi kalemi nedir?",
        ],
        "a": [
            "Gelir vergisi, kümülatif (yıl içinde toplanan) vergi matrahınıza göre artan oranlı dilimlerden hesaplanır. Yıl ilerledikçe toplam kazancınız üst dilime geçebildiği için kesinti oranı artar ve net maaşınız bir miktar düşebilir.",
            "Vergi, yıl başından itibaren biriken kazanç üzerinden hesaplanır. Biriken tutar bir üst vergi dilimine girince oran yükselir; bu yüzden aynı brütte bile yılın ikinci yarısında net azalabilir.",
        ],
    },
    {
        "intent": "define_bes", "domain": "maas_finans", "difficulty": "orta",
        "q": [
            "BES katkısı nedir?",
            "Otomatik bireysel emeklilik nasıl çalışıyor?",
            "bes kesintisi maastan neden var",
            "İşveren BES katkısı ne anlama geliyor?",
            "Bireysel emeklilik sistemi kesintisi zorunlu mu, nedir?",
        ],
        "a": [
            "BES (Bireysel Emeklilik Sistemi), devlet ve/veya işveren katkısıyla desteklenen tamamlayıcı bir emeklilik birikimidir. Maaşınızdan belirli bir oran BES hesabınıza aktarılır; otomatik katılımda cayma hakkınız vardır.",
            "BES, zorunlu SGK emekliliğinin üzerine ek birikim yapmanızı sağlayan sistemdir. Katkı payı bordrodan kesilip bireysel emeklilik hesabınıza yatırılır.",
        ],
    },
    {
        "intent": "define_severance", "domain": "maas_finans", "difficulty": "orta",
        "q": [
            "Kıdem tazminatı nedir?",
            "kidem tazminati neye gore hesaplanir genel olarak",
            "Kıdem tazminatına hak kazanmak için ne gerekiyor?",
            "İşten ayrılınca kıdem tazminatı her durumda ödenir mi?",
            "Kıdem tazminatı kavramını açıklar mısın?",
        ],
        "a": [
            "Kıdem tazminatı, belirli koşullarla iş sözleşmesi sona eren ve genellikle en az bir yıl çalışmış çalışana, her tam yıl için son brüt ücreti tutarında ödenen tazminattır. İstifa gibi bazı durumlarda hak doğmayabilir.",
            "En az bir yıllık kıdemi olan çalışana, kanunda sayılan fesih hallerinde her yıl için bir brüt maaş esas alınarak ödenen tutardır. Kendi durumunuzda hak edip etmediğiniz fesih nedenine bağlıdır.",
        ],
    },
    {
        "intent": "define_notice_pay", "domain": "maas_finans", "difficulty": "orta",
        "q": [
            "İhbar tazminatı nedir?",
            "ihbar suresi ve ihbar tazminati ne demek",
            "İhbar öneli kullandırılmazsa ne olur?",
            "İhbar tazminatı hangi durumda ödenir?",
        ],
        "a": [
            "İhbar tazminatı, iş sözleşmesi feshedilirken kıdeme göre belirlenen ihbar süresine uyulmadığında ödenen tazminattır. Taraflardan biri bu süreye uymadan feshederse, o sürenin ücreti kadar tutarı karşı tarafa öder.",
            "Fesihten önce karşı tarafa belirli bir süre önceden haber verilmesi gerekir (ihbar süresi). Bu süre kullandırılmazsa, süre karşılığı ücret ihbar tazminatı olarak ödenir.",
            "İhbar süreleri kıdeme göre 2 ile 8 hafta arasında değişir. Bu süre kullandırılmadan çıkış yapılırsa, karşılığı olan ücret ihbar tazminatı olarak ödenir; deneme süresi içindeki fesihlerde ihbar şartı aranmaz.",
        ],
    },
    {
        "intent": "difference_severance_notice", "domain": "maas_finans", "difficulty": "orta",
        "q": [
            "Kıdem tazminatı ile ihbar tazminatı arasındaki fark nedir?",
            "kidem ve ihbar tazminati ayni sey mi",
            "İşten çıkışta kıdem mi ihbar mı ödenir, ikisi de mi?",
            "Kıdem ve ihbar tazminatını birbirinden ayıran şey ne?",
        ],
        "a": [
            "Kıdem tazminatı geçmiş hizmetin karşılığıdır ve her tam çalışma yılı için hesaplanır. İhbar tazminatı ise feshin süresine uyulmamasının karşılığıdır; ihbar süresi kadar ücrete karşılık gelir. Biri kıdeme, diğeri bildirim süresine bağlıdır ve koşulları farklıdır.",
            "Kıdem = çalıştığınız yıllara bağlı. İhbar = fesih öncesi haber verme süresine bağlı. Duruma göre biri, diğeri veya her ikisi ödenebilir.",
        ],
    },
    {
        "intent": "overtime_calculation", "domain": "puantaj", "difficulty": "orta",
        "q": [
            "Fazla mesai nasıl hesaplanıyor?",
            "fazla mesai ucreti nasil belirlenir",
            "Mesai saatinin karşılığı normal saatten fazla mı?",
            "Fazla çalışma yaptığımda ücreti nasıl yansıyor?",
            "Fazla mesai zamlı mı ödeniyor, oranı nedir?",
        ],
        "a": [
            "Haftalık yasal çalışma süresini aşan çalışmalar fazla mesai sayılır. Her fazla mesai saati, normal saat ücretinin genellikle 1,5 katı olarak hesaplanır; hafta tatili veya resmi tatilde oran daha yüksek olabilir. Ücret yerine izin (serbest zaman) de tercih edilebilir.",
            "Fazla mesai, normal saatlik ücrete zamlı uygulanır (çoğunlukla %50 zamlı). Kaç saat yaptığınız puantajdan gelir, karşılığı bordroya işlenir.",
            "Önce saatlik ücretiniz bulunur (aylık brüt / aylık çalışma saati), sonra fazla çalışılan her saat bunun 1,5 katından ödenir. Dilerseniz bu saatler ücret yerine serbest zaman olarak da kullanılabilir.",
        ],
    },
    {
        "intent": "weekly_rest_work", "domain": "puantaj", "difficulty": "orta",
        "q": [
            "Hafta tatilinde çalışırsam ne olur?",
            "haftalik izin gununde calisinca ekstra ucret var mi",
            "Cumartesi-pazar çalışması nasıl karşılanıyor?",
            "Hafta tatilinde çalışmanın karşılığı nedir?",
        ],
        "a": [
            "Hafta tatili, kesintisiz olarak verilmesi gereken bir dinlenme günüdür. Bu günde çalışılırsa, çalışma fazla mesai gibi zamlı ücretlendirilir; ayrıca çoğu iş yeri telafi (serbest zaman) imkanı sunar.",
            "Hafta tatilinde çalışma, normal günden farklı olarak zamlı ödenir ve/veya başka bir gün izinle telafi edilir. Detay şirket politikanıza bağlıdır.",
            "Hafta tatili ücreti çalışılmasa da ödenir. O gün çalışırsanız bunun üzerine bir de çalıştığınız süre zamlı (en az %50) eklenir; şirketiniz ayrıca serbest gün de verebilir.",
        ],
    },
    {
        "intent": "public_holiday_work", "domain": "puantaj", "difficulty": "orta",
        "q": [
            "Resmi tatilde çalışmak zorunda kalırsam ücreti nasıl olur?",
            "bayramda calisinca ne kadar ucret aliyoruz",
            "Resmi tatil çalışması normal günden farklı mı ödeniyor?",
            "Resmi tatilde işe çağrılırsam hakkım ne olur?",
        ],
        "a": [
            "Resmi ve dini bayram günleri çalışılmasa da ücret ödenir. Bu günlerde çalışılırsa, çalışılan her gün için bir günlük ek ücret ödenir; uygulamada çoğu iş yeri bunu zamlı yansıtır.",
            "Resmi tatilde çalışırsanız, o günün ücretine ek olarak çalıştığınız süre kadar ayrıca ödeme yapılır. Şirketiniz bunun üstüne ek zam veya izin de tanımlayabilir.",
            "Tatil ücreti zaten ödenir; çalışmanız halinde çalışılan gün başına bir yevmiye daha eklenir. Puantajda bu günler ayrı işaretlenir ve karşılığı bordroya yansır.",
        ],
    },
    {
        "intent": "probation_period", "domain": "calisan_bilgileri", "difficulty": "kolay",
        "q": [
            "Deneme süresi nedir, ne kadar sürer?",
            "deneme suresinde haklarim farkli mi",
            "Deneme süresi içinde çıkış nasıl oluyor?",
            "Deneme süresi kavramını açıklar mısın?",
        ],
        "a": [
            "Deneme süresi, iş ilişkisinin başında tarafların birbirini değerlendirdiği dönemdir; iş sözleşmesiyle kararlaştırılır ve genellikle iki ayı geçmez (toplu sözleşmeyle uzatılabilir). Bu sürede taraflar bildirim süresi olmadan sözleşmeyi feshedebilir; ancak çalışılan günlerin ücreti ve sigortası eksiksiz ödenir.",
            "Sözleşmede kararlaştırılan, tarafların uyumu değerlendirdiği başlangıç dönemidir. Süre içinde fesih ihbarsız yapılabilir, fakat çalışılan süreye ait tüm haklar (ücret, SGK) ödenir.",
        ],
    },
    {
        "intent": "define_fringe_benefit", "domain": "maas_finans", "difficulty": "kolay",
        "q": [
            "Yan hak nedir?",
            "yan haklar denince ne anlasiliyor",
            "Maaş dışı haklara örnek verir misin?",
            "Yan haklar kavramı neyi kapsar?",
            "Özlük hakları ile yan haklar aynı şey mi?",
        ],
        "a": [
            "Yan haklar, temel ücretin dışında çalışana sağlanan ek imkanlardır: özel sağlık sigortası, yemek kartı, yol/ulaşım desteği, telefon hattı, BES işveren katkısı, şirket aracı, eğitim bütçesi gibi. Paket şirkete ve pozisyona göre değişir.",
            "Maaşa ek olarak sunulan parasal veya ayni imkanlardır; sağlık sigortası, yemek ve ulaşım yardımı, ikramiye dışı destekler bu kapsamdadır.",
        ],
    },
    {
        "intent": "difference_prim_bonus", "domain": "maas_finans", "difficulty": "orta",
        "q": [
            "Prim ile bonus arasındaki fark nedir?",
            "prim ve bonus ayni mi",
            "Performans primi ile yıl sonu bonusu farklı şeyler mi?",
            "Prim mi bonus mu; ikisini ayıran ne?",
        ],
        "a": [
            "Prim genellikle belirli bir hedefe veya performansa bağlı, düzenli olabilen ödemedir (satış primi, üretim primi gibi). Bonus ise çoğunlukla dönemsel ve şirket sonuçlarına bağlı, takdiri bir ek ödemedir (yıl sonu bonusu gibi). İkisi de brüt üzerinden vergilendirilir.",
            "Prim daha çok bireysel/ekip hedefine bağlıdır ve önceden kriterleri bellidir. Bonus daha çok şirketin genel başarısına ve yönetim takdirine bağlıdır.",
        ],
    },
    {
        "intent": "maternity_leave_general", "domain": "izin_yonetimi", "difficulty": "orta",
        "q": [
            "Doğum izni genel olarak ne kadar sürüyor?",
            "analik izni sureleri nasil",
            "Doğumdan önce ve sonra ne kadar izin kullanılıyor?",
            "Doğum izni hakkında genel bilgi verir misin?",
        ],
        "a": [
            "Genel uygulamada analık izni doğumdan önce ve sonra olmak üzere toplam on altı hafta civarındadır (çoğul gebelikte önceki kısım uzar). Sonrasında talebe bağlı ücretsiz izin ve süt izni hakları da vardır. Kesin süre ve ödeme için İK'nızla ve SGK düzenlemeleriyle teyit etmelisiniz.",
            "Kabaca doğumdan önce sekiz, sonra sekiz hafta olmak üzere ücretli analık izni kullanılır; ardından yarı zamanlı çalışma ve ücretsiz izin seçenekleri gündeme gelir. Ayrıntı mevzuata ve şirket politikanıza göre değişir.",
        ],
    },
    {
        "intent": "paternity_leave_general", "domain": "izin_yonetimi", "difficulty": "kolay",
        "q": [
            "Babalık izni kaç gün?",
            "esim dogum yapinca kac gun izin alabilirim",
            "Babalık izni genel olarak ne kadar?",
            "Yeni baba olan çalışanın izin hakkı nedir?",
        ],
        "a": [
            "Babalık (mazeret) izni genel olarak eşin doğumunda beş gün civarındadır; kamu ve bazı şirketlerde on güne kadar çıkabilir. Kesin süre şirketinizin İK politikasına bağlıdır. İsterseniz tarihleri verin, mazeret izni talebinizi oluşturayım.",
            "Eşin doğum yapması halinde çalışana genelde beş iş günü mazeret izni tanınır; şirketiniz bunu artırmış olabilir.",
            "İş Kanunu'na göre eşi doğum yapan çalışana beş gün mazeret izni verilir. Şirket politikanız daha uzun tanımlamış olabilir; net süreyi İK'nızdan teyit edebilirsiniz.",
        ],
    },
    {
        "intent": "marriage_leave_general", "domain": "izin_yonetimi", "difficulty": "kolay",
        "q": [
            "Evlilik izni kaç gün?",
            "evlenince ne kadar izin veriliyor",
            "Evlilik mazeret izni genel olarak ne kadar sürüyor?",
            "Düğün için izin hakkım ne kadar?",
        ],
        "a": [
            "Evlilik izni genel olarak üç gün mazeret iznidir. Bazı şirketler bunu beş güne çıkarır. Nikah tarihinizi netleştirdiğinizde mazeret izni talebinizi oluşturabilirim.",
            "Çalışanın kendi evliliğinde genelde üç iş günü mazeret izni verilir; şirket politikanız daha uzun tanımlamış olabilir.",
        ],
    },
    {
        "intent": "bereavement_leave_general", "domain": "izin_yonetimi", "difficulty": "kolay",
        "q": [
            "Vefat durumunda izin hakkı ne kadar?",
            "yakinim vefat etti kac gun iznim var",
            "Ölüm izni genel olarak kaç gün?",
            "Birinci derece yakının kaybında mazeret izni ne kadar?",
        ],
        "a": [
            "Ana, baba, eş, kardeş veya çocuğun vefatında genel olarak üç gün mazeret izni verilir. Başınız sağ olsun. Tarihleri iletirseniz izin talebinizi oluşturabilirim.",
            "Birinci derece yakınların vefatında genelde üç iş günü mazeret izni tanınır; şirketiniz kapsamı genişletmiş olabilir.",
        ],
    },
    {
        "intent": "unpaid_leave_general", "domain": "izin_yonetimi", "difficulty": "orta",
        "q": [
            "Ücretsiz izin nasıl bir şey?",
            "ucretsiz izin alirsam maasim ve sigortam ne olur",
            "Ücretsiz izin döneminde haklarım ne oluyor?",
            "Uzun süreli ücretsiz izin mümkün mü, sonuçları neler?",
        ],
        "a": [
            "Ücretsiz izinde iş sözleşmeniz devam eder ama o dönem için ücret ödenmez ve genellikle SGK primi yatmaz (bazı özel hallerde devam eder). Yıllık izin hak edişi ve kıdem açısından bu süre çoğunlukla sayılmaz. İşveren onayı gerekir.",
            "Ücretsiz izin, karşılıklı anlaşmayla ücret ödenmeksizin işe ara verilmesidir. Bu sürede prim ve ücret durur; dönüşte aynı pozisyona dönülmesi esastır.",
        ],
    },
    {
        "intent": "lactation_leave_general", "domain": "izin_yonetimi", "difficulty": "kolay",
        "q": [
            "Süt izni nedir, ne kadar?",
            "sut izni gunde kac saat",
            "Emziren çalışanın izin hakkı nasıl işliyor?",
        ],
        "a": [
            "Süt izni, doğum sonrası belirli bir dönem boyunca (genel uygulamada çocuk bir yaşına gelene kadar) günde toplam bir buçuk saat olarak kullanılır. Bu süre iş saatinden sayılır ve ücretten kesilmez; kullanım saatini çalışan belirler.",
            "Emziren çalışana günde 1,5 saat süt izni tanınır; bu süre çalışılmış kabul edilir. Toplu kullanım için İK'nızla görüşebilirsiniz.",
        ],
    },
    {
        "intent": "define_puantaj", "domain": "puantaj", "difficulty": "kolay",
        "q": [
            "Puantaj nedir?",
            "puantaj kaydi ne ise yarar",
            "Puantaj hangi bilgileri tutar?",
            "İK'da puantaj denince ne anlaşılır?",
        ],
        "a": [
            "Puantaj, çalışanın gün gün çalışma kaydıdır: hangi gün çalıştı, izinli/raporlu muydu, giriş-çıkış saatleri, devamsızlık ve fazla mesai bilgileri burada tutulur. Bordro hesabı büyük ölçüde puantajdan beslenir.",
            "Aylık devam-devamsızlık çizelgesidir; çalışılan günler, izinler ve fazla mesai bu kayıttan takip edilir ve maaş hesabına girer.",
        ],
    },
    {
        "intent": "expense_reimbursement_process", "domain": "ik_islemleri", "difficulty": "orta",
        "q": [
            "Masraf iademi nasıl talep ederim?",
            "is seyahati harcamalarini nasil geri alirim",
            "Harcama iadesi süreci nasıl işliyor?",
            "Fişleri nereye iletmem gerekiyor, masraf iadesi için ne yapmalıyım?",
        ],
        "a": [
            "Masraf iadesi için harcama belgelerinizi (fatura/fiş) toplayıp bir masraf formuyla birlikte gider onay sürecine iletmeniz gerekir. Yöneticiniz onayladıktan sonra tutar genellikle bir sonraki maaşla veya ayrı bir ödemeyle hesabınıza yatar. Bu asistan üzerinden masraf iadesi başlatılamıyor; ilgili gider sistemini kullanmalısınız.",
            "Genel akış: belgeleri topla, masraf formunu doldur, yönetici onayına gönder, finans onayından sonra ödeme yapılır. Süreç şirketin gider yönetimi aracından yürür.",
        ],
    },
    {
        "intent": "how_to_read_payslip", "domain": "maas_finans", "difficulty": "orta",
        "q": [
            "Bordromu nasıl okumalıyım?",
            "bordrodaki kalemleri anlamiyorum genel olarak neye bakmaliyim",
            "Maaş pusulasındaki satırlar ne anlama geliyor?",
            "Bordroyu yorumlamak için nelere dikkat etmeliyim?",
        ],
        "a": [
            "Bordroyu üç blokta düşünün: kazançlar (brüt ücret, fazla mesai, prim), yasal kesintiler (SGK işçi payı, işsizlik, gelir vergisi, damga vergisi) ve sonuçta net ödenen tutar. Çalışılan gün sayısı ve kümülatif vergi matrahı da genelde üstte yer alır ve net tutarın neden değiştiğini açıklar.",
            "Önce brüt ücrete, sonra kesinti kalemlerine, en sonda net ödemeye bakın. Kümülatif gelir vergisi matrahı, yıl içinde netin neden düştüğünü gösterir.",
            "Üst kısımda kimlik/dönem/çalışılan gün bilgisi olur. Ortada kazançlar toplanır, altında SGK ve vergi kesintileri düşülür. En alttaki 'net ödenen' banka hesabınıza yatan tutardır; 'kümülatif matrah' arttıkça netin neden azaldığını buradan görebilirsiniz.",
        ],
    },
    {
        "intent": "leave_approval_flow", "domain": "ik_islemleri", "difficulty": "kolay",
        "q": [
            "İzin talebimi kim onaylıyor?",
            "izin onayi kimden geciyor",
            "İzin başvurusu hangi onay adımlarından geçiyor?",
            "Talebi gönderdikten sonra onay süreci nasıl ilerliyor?",
        ],
        "a": [
            "İzin talebi önce doğrudan yöneticinize düşer. Yöneticiniz onayladığında talep kesinleşir; bazı şirketlerde uzun izinler için ikinci bir onay (departman müdürü veya İK) devreye girer. Onaylandığında izin bakiyeniz otomatik güncellenir.",
            "Genelde tek onay yeterlidir ve bu onay yöneticinizdedir. Onay sonrası İK kaydı ve bakiye güncellemesi sistemce yapılır.",
        ],
    },
    {
        "intent": "annual_leave_days_general", "domain": "izin_yonetimi", "difficulty": "orta",
        "q": [
            "Kıdeme göre yıllık izin kaç gün?",
            "1 yildan az calisana yillik izin var mi",
            "5 yıllık çalışan kaç gün yıllık izne hak kazanır (genel kural)?",
            "Yıllık izin gün sayısı çalışma süresine göre nasıl değişir?",
        ],
        "a": [
            "Genel yasal taban şöyledir: 1-5 yıl arası kıdemde yılda 14 gün, 5-15 yıl arası 20 gün, 15 yıl ve üzeri 26 gün. 18 yaşından küçük ve 50 yaşından büyük çalışanlar için en az 20 gündür. Şirketiniz bu tabanın üzerinde gün tanımlamış olabilir; kendi bakiyeniz için İK kaydınıza bakmalısınız.",
            "Yasal asgari: ilk 5 yıl 14 gün, sonra 20 gün, 15 yıldan sonra 26 gün. Bir yılını doldurmayan çalışanın henüz yıllık izin hakkı doğmaz.",
        ],
    },
    {
        "intent": "advance_request_process", "domain": "ik_islemleri", "difficulty": "orta",
        "q": [
            "Maaş avansı nasıl talep edilir?",
            "avans istemek icin ne yapmam gerekiyor",
            "Avans süreci nasıl işliyor?",
        ],
        "a": [
            "Avans talebi genellikle İK/finans portalı üzerinden açılır: talep tutarını ve gerekçenizi girersiniz, yöneticiniz ve finans onaylar, onaylanan tutar bir sonraki maaştan mahsup edilmek üzere ödenir. Bu asistan avans işlemini başlatamıyor; ilgili portalı kullanmanız gerekir.",
            "Tutarı ve nedeni belirtip onaya gönderirsiniz; onaydan sonra avans ödenir ve gelecek bordronuzdan kesilir. Süreç ayrı bir finans aracından yürür.",
        ],
    },
    {
        "intent": "assistant_identity", "domain": "meta", "difficulty": "kolay",
        "q": [
            "Sen kimsin?",
            "sen ne yapan bir asistansin",
            "Kiminle konuşuyorum, neyi temsil ediyorsun?",
            "Bu sistemin adı ne, sen ne işe yarıyorsun?",
        ],
        "a": [
            "Büyük İK asistanıyım. İzin, maaş, bordro, puantaj ve çalışan bilgileri gibi İK konularında bilgi verebilir; izin talebi oluşturma, iptal etme gibi işlemleri onayınızla yürütebilirim.",
            "Büyük İK sisteminin sanal asistanıyım. İK süreçleriyle ilgili sorularınızı yanıtlar ve yetkim dahilindeki işlemleri sizin için yaparım.",
        ],
    },
    {
        "intent": "assistant_capabilities", "domain": "meta", "difficulty": "kolay",
        "q": [
            "Neler yapabilirsin?",
            "hangi konularda yardimci olabiliyorsun",
            "Bana nasıl yardımcı olabilirsin?",
            "Yeteneklerin neler, ne tür isteklerde bulunabilirim?",
            "Seninle neleri halledebilirim?",
        ],
        "a": [
            "Şunlarda yardımcı olabilirim: izin bakiyesi ve geçmişi, izin talebi oluşturma/iptal, maaş ve bordro bilgisi, prim ve yan haklar, puantaj ve fazla mesai, çalışan ve departman bilgileri. Ayrıca izin ve iletişim bilgisi güncelleme gibi işlemleri onayınızla yapabilirim.",
            "İzinler, maaş/bordro, puantaj ve organizasyon bilgileri hakkında sorularınızı yanıtlarım; izin oluşturma, iptal ve bilgi güncelleme işlemlerini de yürütebilirim. Genel İK kavramlarını da açıklayabilirim.",
            "Başlıca iki şey: İK kayıtlarınızı sorgulamak (izin, maaş, bordro, prim, yan haklar, puantaj, mesai, çalışan/departman bilgisi) ve onayınızla işlem yapmak (izin talebi açma/iptal, iletişim güncelleme). Bir de yıllık izin, bordro, tazminat gibi kavramları açıklayabilirim.",
        ],
    },
    {
        "intent": "assistant_scope_limits", "domain": "meta", "difficulty": "orta",
        "q": [
            "Neleri yapamazsın?",
            "hangi konularda yardimci olamiyorsun",
            "Sınırların neler?",
            "Hangi taleplerde bana yardımcı olamazsın?",
        ],
        "a": [
            "Geleceğe dönük tahminler (kim zam/terfi alacak, işten çıkışlar), başka çalışanların özel/mali bilgileri, İK dışı konular (hava durumu, yatırım tavsiyesi) ve sistemde karşılığı olmayan işlemler kapsamım dışında. Bu tür isteklerde yardımcı olamam.",
            "Kişisel kariyer tavsiyesi veremem, başkasının gizli bilgisini paylaşamam ve mevcut araçların dışındaki işlemleri yapamam. Ayrıca doğrulayamadığım hiçbir bilgiyi uydurmam.",
        ],
    },
    {
        "intent": "privacy_policy_general", "domain": "meta", "difficulty": "orta",
        "q": [
            "Başka bir çalışanın maaşını öğrenebilir miyim?",
            "arkadasimin izin bilgisine bakabilir miyim",
            "Ekip arkadaşımın bordrosunu görmek mümkün mü?",
            "Yönetici değilim; başkasının bilgilerine erişebilir miyim?",
        ],
        "a": [
            "Hayır. Maaş, bordro, izin ve iletişim bilgileri kişiye özeldir; yalnızca kişinin kendisi ve yetkilendirilmiş İK/yönetici rolleri görebilir. Yöneticiyseniz, ekibinizle sınırlı olmak üzere yetkiniz olabilir. Yetki kontrolü için ilgili aracı kullanabilirim.",
            "Çalışanlar birbirinin özlük ve mali bilgilerini göremez. Bu bilgiler yalnızca ilgili kişiye ve yetkili İK ekibine açıktır.",
        ],
    },
    {
        "intent": "greeting", "domain": "meta", "difficulty": "kolay",
        "q": [
            "Merhaba", "Selam", "Günaydın", "İyi günler", "merhabalar", "Selam, orada mısın?",
            "Merhaba, bir şey soracaktım", "Kolay gelsin", "İyi akşamlar", "Selamlar, müsait misin?",
            "Merhaba İK", "hey selam",
        ],
        "a": [
            "Merhaba! Büyük İK asistanınıyım. İzin, maaş, bordro veya puantaj konularında nasıl yardımcı olabilirim?",
            "Merhaba, hoş geldiniz. İK ile ilgili ne öğrenmek veya hangi işlemi yapmak istersiniz?",
            "Selam! Size nasıl yardımcı olabilirim?",
            "İyi günler! Buyurun, dinliyorum.",
        ],
    },
    {
        "intent": "thanks", "domain": "meta", "difficulty": "kolay",
        "q": [
            "Teşekkürler", "Çok teşekkür ederim", "Sağ ol", "Eline sağlık", "teşekkürler yardımcı oldun",
            "Süper, teşekkürler", "Harika, sağ olasın", "Çok yardımcı oldun, teşekkürler",
            "Tamamdır, teşekkür ederim", "eyvallah",
        ],
        "a": [
            "Rica ederim! Başka bir konuda yardım gerekirse buradayım.",
            "Ne demek, her zaman. İyi çalışmalar!",
            "Rica ederim. Başka bir sorunuz olursa çekinmeyin.",
            "Sevindim, kolay gelsin!",
        ],
    },
    {
        "intent": "farewell", "domain": "meta", "difficulty": "kolay",
        "q": [
            "Görüşürüz", "İyi günler, hoşça kal", "Kapatıyorum, teşekkürler", "Bay bay", "şimdilik bu kadar",
            "Sonra devam ederiz", "Şimdilik yeterli, teşekkürler", "Hoşça kal", "Kolay gelsin, görüşürüz",
        ],
        "a": [
            "İyi günler! İhtiyaç olursa yine buradayım.",
            "Görüşmek üzere, kolay gelsin.",
            "Hoşça kalın, iyi çalışmalar.",
            "Tamamdır, iyi günler!",
        ],
    },
    {
        "intent": "resignation_process", "domain": "ik_islemleri", "difficulty": "orta",
        "q": [
            "İstifa süreci nasıl işliyor?",
            "isten ayrilmak istersem ne yapmam gerekiyor",
            "İstifa dilekçesini kime veriyorum, sonrası nasıl ilerliyor?",
            "Ayrılma sürecinin adımları neler?",
            "İstifa edince ihbar süresi nasıl uygulanıyor?",
        ],
        "a": [
            "Genel akış: yazılı istifa bildiriminizi yöneticinize ve İK'ya iletirsiniz, kıdeminize göre ihbar süresi başlar, bu süre boyunca çalışır ya da ihbar tazminatı üzerinden anlaşırsınız. Son gün zimmet iadesi, çıkış görüşmesi ve ardından ihbar/kıdem hak edişlerinizin bordrosu yapılır.",
            "İstifa yazılı olarak verilir; ihbar süresi kıdeminize göre 2-8 hafta arasıdır. Süre sonunda çıkış işlemleri, zimmet teslimi ve son ödeme (varsa kıdem, kullanılmamış izin ücreti) tamamlanır.",
        ],
    },
    {
        "intent": "work_certificate_process", "domain": "ik_islemleri", "difficulty": "kolay",
        "q": [
            "Çalışma belgesi nasıl alırım?",
            "calisan belgesi talebi nasil yapiliyor",
            "Bankaya vermek için çalıştığıma dair belge lazım, nereden alırım?",
            "Hizmet belgesi / görev belgesi nasıl talep edilir?",
        ],
        "a": [
            "Çalışma (görev) belgesi talebinizi İK'ya iletirsiniz; belgede unvanınız, işe giriş tarihiniz ve talep ederseniz ücret bilgisi yer alır. Genellikle 1-2 iş günü içinde hazırlanıp e-posta veya ıslak imzalı olarak verilir. Bu asistan üzerinden belge düzenlenemiyor.",
            "İK'ya kısa bir talep yeterli: hangi kuruma verileceğini ve ücret bilgisi istenip istenmediğini belirtin. Belge birkaç iş günü içinde hazırlanır.",
        ],
    },
    {
        "intent": "remote_work_policy", "domain": "ik_islemleri", "difficulty": "orta",
        "q": [
            "Uzaktan çalışma politikası nedir?",
            "haftada kac gun evden calisabiliyoruz",
            "Hibrit çalışma kuralları neler?",
            "Evden çalışmak için onay almam gerekiyor mu?",
        ],
        "a": [
            "Uzaktan/hibrit çalışma hakkı ve gün sayısı şirket politikanıza ve pozisyonunuza göre değişir; genellikle haftada belirli gün ofiste bulunma şartı olur ve düzenli uzaktan çalışma için yöneticinin onayı ve bir protokol gerekir. Kesin kural için İK politikanıza bakmalısınız.",
            "Genel çerçeve: belirli günler ofis, kalan günler uzaktan olacak şekilde yönetici onayıyla planlanır. Ekipten ekibe değişebilir.",
        ],
    },
    {
        "intent": "notice_period_duration", "domain": "maas_finans", "difficulty": "orta",
        "q": [
            "İhbar süresi kıdeme göre kaç hafta?",
            "ihbar oneli ne kadar oluyor",
            "6 aylık çalışanın ihbar süresi ne kadardır (genel kural)?",
            "İhbar süreleri nasıl belirleniyor?",
        ],
        "a": [
            "Yasal ihbar süreleri kıdeme göre şöyledir: 6 aya kadar 2 hafta, 6 ay - 1,5 yıl arası 4 hafta, 1,5 - 3 yıl arası 6 hafta, 3 yıldan fazla 8 hafta. Sözleşmeniz bu sürelerin üzerinde bir süre öngörmüş olabilir.",
            "Kıdem arttıkça ihbar süresi uzar: 2, 4, 6, 8 hafta şeklinde dört kademe vardır. Süreye uyulmazsa karşılığı ihbar tazminatı olarak ödenir.",
        ],
    },
    {
        "intent": "kumulatif_matrah_explained", "domain": "maas_finans", "difficulty": "zor",
        "q": [
            "Kümülatif gelir vergisi matrahı tam olarak ne demek?",
            "kumulatif matrah neden her ay artiyor",
            "Bordrodaki kümülatif matrah satırı neyi gösteriyor?",
            "Kümülatif matrah net maaşı nasıl etkiliyor?",
        ],
        "a": [
            "Kümülatif matrah, yıl başından o aya kadar gelir vergisine tabi toplam kazancınızdır. Vergi bu biriken tutar üzerinden artan oranlı dilimlere göre hesaplanır; biriken tutar bir üst dilime geçtiğinde vergi oranınız yükselir ve aynı brüt ücrette bile net maaşınız yıl ilerledikçe azalabilir. Ocak'ta sıfırlanır.",
            "Her ay o ayki brüt, önceki ayların matrahına eklenir. Toplam bir vergi dilimi eşiğini aştığında sonraki gelirinize daha yüksek oran uygulanır; bu yüzden yılın ikinci yarısında net düşebilir.",
        ],
    },
    {
        "intent": "overtime_to_leave_conversion", "domain": "puantaj", "difficulty": "orta",
        "q": [
            "Fazla mesaimi ücret yerine izne çevirebilir miyim?",
            "mesai karsiligi izin nasil oluyor",
            "Fazla çalışma karşılığında serbest zaman kullanımı mümkün mü?",
            "Fazla mesai izni ne kadar sürede kullanılmalı?",
        ],
        "a": [
            "Evet. Fazla çalışma karşılığı ücret yerine serbest zaman (izin) tercih edebilirsiniz; her fazla mesai saati için 1,5 saat izin hak edilir ve bu izin genellikle altı ay içinde, iş yoğunluğuna göre kullandırılır. Tercihi yazılı olarak belirtmeniz beklenir.",
            "Fazla mesai saatinin karşılığı 1,5 katı serbest zamandır. Ücret mi izin mi istediğinizi bildirirsiniz; izin, belirli bir süre içinde planlanarak kullanılır.",
        ],
    },
    {
        "intent": "excuse_leave_documentation", "domain": "izin_yonetimi", "difficulty": "kolay",
        "q": [
            "Mazeret izni için belge gerekiyor mu?",
            "evlilik izninde ne belgesi isteniyor",
            "Mazeret iznini neyle belgelendiriyorum?",
            "Doğum/ölüm izni için hangi evrak lazım?",
        ],
        "a": [
            "Genellikle evet: evlilik için evlilik cüzdanı örneği, doğum için doğum belgesi, vefat için ilgili belge gibi olaya özgü bir kanıt istenir. Belgeyi izin dönüşü İK'ya iletmeniz yeterlidir; ön onay için talep sırasında belirtmeniz beklenir.",
            "Mazeret türüne göre bir belge (evlilik cüzdanı, doğum raporu, vefat belgesi vb.) sunmanız istenir. Belge sonradan da tamamlanabilir ama kayıt için gereklidir.",
        ],
    },
    {
        "intent": "sick_leave_pay", "domain": "izin_yonetimi", "difficulty": "orta",
        "q": [
            "Raporlu olduğum günlerde maaşım tam ödenir mi?",
            "istirahat raporunda ucret nasil oluyor",
            "Hastalık izninde SGK geçici iş göremezlik ödeneği nedir?",
            "Rapor parası ile maaş farkını şirket tamamlıyor mu?",
        ],
        "a": [
            "Raporlu günlerde ücret SGK'nın geçici iş göremezlik ödeneği üzerinden ödenir; bu ödenek ilk 2 gün için yapılmaz ve genellikle günlük kazancın bir kısmını karşılar. Birçok şirket, ödenek ile normal ücret arasındaki farkı politika gereği tamamlar, ancak bu zorunlu değildir.",
            "SGK, rapor süresince günlük bir ödenek verir (ilk iki gün hariç). İşveren farkı tamamlayabilir; bu tamamen şirket politikasına bağlıdır.",
        ],
    },
    {
        "intent": "public_holiday_pay_rate", "domain": "puantaj", "difficulty": "orta",
        "q": [
            "Resmi tatilde çalışırsam ücret nasıl hesaplanıyor?",
            "bayram gunu calisinca kac yevmiye aliyorum",
            "Genel tatil günü mesai ücreti kaç katı?",
            "Hafta tatilinde çalışmanın karşılığı nedir?",
        ],
        "a": [
            "Genel (resmi) tatil günlerinde çalışmazsanız o günün ücreti zaten ödenir; çalışırsanız çalıştığınız her gün için bir yevmiye daha eklenir. Hafta tatilinde çalışma ise fazla çalışma sayılır ve %50 zamlı ödenir; bu günler puantajda ayrı işaretlenir.",
            "Resmi tatilde çalışma: normal ücrete ek olarak çalışılan gün başına bir günlük ücret daha. Hafta tatili çalışması fazla mesai kapsamında %50 zamlıdır.",
        ],
    },
    {
        "intent": "leave_min_block_rule", "domain": "izin_yonetimi", "difficulty": "orta",
        "q": [
            "Yıllık iznin bir kısmını kesintisiz kullanmak zorunda mıyım?",
            "yillik izni parca parca kullanabilir miyim",
            "İznin en az kaç günü birlikte kullanılmalı?",
            "Yıllık izni bölmenin bir kuralı var mı?",
        ],
        "a": [
            "Yıllık iznin bir bölümü kesintisiz kullanılmalıdır: yasa gereği yılda en az bir kez 10 günlük kısım bir arada verilir. Kalan günler tarafların anlaşmasıyla bölünerek kullanılabilir.",
            "En az 10 günü bir arada kullanmanız beklenir; geri kalanı yöneticinizle anlaşarak parçalı kullanabilirsiniz.",
        ],
    },
    {
        "intent": "expense_advance_difference", "domain": "maas_finans", "difficulty": "orta",
        "q": [
            "Maaş avansı ile masraf iadesi arasındaki fark ne?",
            "avans mi harcama iadesi mi hangisini istemeliyim",
            "İş için yaptığım harcamayı avans olarak mı almalıyım?",
        ],
        "a": [
            "Maaş avansı, ileride hak edeceğiniz ücretin bir kısmının erken ödenmesidir ve sonraki bordrolarınızdan mahsup edilir. Masraf iadesi ise iş için cebinizden yaptığınız (fatura/fişle belgelenen) harcamanın size geri ödenmesidir ve maaştan kesilmez. İş harcamaları için doğru yol masraf iadesidir.",
            "Avans borçlanma gibidir, geri kesilir. Masraf iadesi belgeli iş giderinizin telafisidir, kesinti olmaz.",
        ],
    },
    {
        "intent": "handover_on_exit", "domain": "ik_islemleri", "difficulty": "kolay",
        "q": [
            "İşten ayrılırken zimmet iadesi nasıl yapılıyor?",
            "cikista neleri teslim etmem gerekiyor",
            "Ayrılış gününde hangi işlemler yapılıyor?",
        ],
        "a": [
            "Son iş gününüzde bilgisayar, telefon, erişim kartı gibi zimmetli eşyaları İK/BT'ye teslim eder, devam eden işlerinizi bir devir notuyla aktarır ve çıkış görüşmesine katılırsınız. Bunlar tamamlanınca çıkış evrakları ve son ödemeniz hazırlanır.",
            "Zimmetli eşyalar teslim edilir, işler devredilir, çıkış görüşmesi yapılır. Ardından ilişik kesme belgesi ve son bordro düzenlenir.",
        ],
    },
    {
        "intent": "probation_leave_rights", "domain": "izin_yonetimi", "difficulty": "orta",
        "q": [
            "Deneme süresindeyken izin kullanabilir miyim?",
            "deneme suresinde yillik izin hakki var mi",
            "İlk aylarda izne çıkmak mümkün mü?",
        ],
        "a": [
            "Yıllık ücretli izne hak kazanmak için işyerinde bir yılı doldurmak gerekir; dolayısıyla deneme süresinde henüz yıllık izin hakkınız oluşmamıştır. Ancak mazeret izinleri (evlilik, doğum, vefat vb.) ve raporlu (hastalık) izin ilk günden itibaren geçerlidir. Ücretsiz izin ise yöneticinizin onayına bağlıdır.",
            "Deneme süresinde yıllık izin doğmaz (bir yıl şartı). Mazeret ve hastalık izni ise süreye bakılmaksızın kullanılabilir.",
        ],
    },
]


# ---------------------------------------------------------------------------
# 5. CANNOT_ANSWER — mevcut araçlarla cevaplanamayan istekler
# ---------------------------------------------------------------------------

CANNOT_INTENTS: list[dict] = [
    # --- gelecek / doğrulanamaz ---
    {"intent": "predict_company_inflation", "domain": "kapsanmayan", "pool": "future", "q": [
        "Şirketimizin gelecek yıl enflasyon tahmini nedir?",
        "onumuzdeki sene sirket enflasyonu kac olur",
        "2027'de şirket giderlerimiz enflasyona göre nasıl artar?",
        "Gelecek yıl için şirketin enflasyon beklentisini söyler misin?",
    ]},
    {"intent": "predict_economic_growth", "domain": "kapsanmayan", "pool": "future", "q": [
        "Önümüzdeki yıl ekonomik büyüme ne olur?",
        "gelecek yil turkiye ekonomisi nasil olacak",
        "2027 büyüme rakamlarını tahmin eder misin?",
    ]},
    {"intent": "predict_stock_market", "domain": "kapsanmayan", "pool": "future", "q": [
        "Borsa yarın yükselir mi?",
        "bu hafta hisse senetleri ne yapar",
        "Piyasa önümüzdeki ay nasıl bir seyir izler?",
    ]},
    {"intent": "predict_future_raises", "domain": "maas_finans", "pool": "future", "q": [
        "Gelecek ay kimler zam alacak?",
        "onumuzdeki donem kime ne kadar zam yapilacak listesini ver",
        "Seneye kesin zam alacak çalışanları söyler misin?",
        "Bir sonraki maaş döneminde zam alacakların listesini çıkar.",
    ]},
    {"intent": "predict_future_promotions", "domain": "organizasyon", "pool": "future", "q": [
        "Önümüzdeki yıl kim terfi edecek?",
        "gelecek donem terfi edecekleri tahmin et",
        "Seneye hangi çalışanlar yükselecek?",
    ]},
    {"intent": "predict_layoffs", "domain": "organizasyon", "pool": "future", "q": [
        "Önümüzdeki dönemde kim işten çıkarılacak?",
        "isten cikarilacak calisanlari soyle",
        "Küçülme olursa hangi departmandan kimler gider?",
    ]},
    {"intent": "predict_exact_future_leave", "domain": "izin_yonetimi", "pool": "future", "q": [
        "Üç yıl sonra tam olarak kaç günüm birikmiş olur, kesin söyle.",
        "2030'da kac gun yillik iznim olacak net rakam ver",
        "Önümüzdeki 5 yıl boyunca her yıl kaç izin biriktireceğimi kesin olarak hesapla.",
    ]},
    {"intent": "guarantee_leave_approval", "domain": "izin_yonetimi", "pool": "future", "q": [
        "Eylülde talep edersem iznim kesin onaylanır mı?",
        "yonetici iznimi onaylayacak mi garanti veriyor musun",
        "Bu izin talebi %100 geçer mi?",
    ]},
    # --- gizlilik / yetki ---
    {"intent": "compare_others_salaries", "domain": "maas_finans", "pool": "privacy", "q": [
        "Diğer çalışanların maaşlarını karşılaştırır mısın?",
        "ekipteki herkesin maasini kiyasla",
        "Departmandaki maaşları büyükten küçüğe sırala.",
        "Benimle aynı pozisyondakilerin maaşlarını yan yana göster.",
    ]},
    {"intent": "coworker_salary", "domain": "maas_finans", "pool": "privacy", "q": [
        "Ahmet'in maaşı ne kadar?",
        "yan masadaki arkadasimin maasini soyle",
        "Takım liderimizin brüt maaşını öğrenebilir miyim?",
        "Mehmet Bey ne kazanıyor, bir bakar mısın?",
    ]},
    {"intent": "ceo_bank_details", "domain": "maas_finans", "pool": "privacy", "q": [
        "CEO'nun banka hesap bilgilerini ver.",
        "genel mudurun iban numarasi ne",
        "Üst yönetimin maaş hesap bilgilerini görebilir miyim?",
    ]},
    {"intent": "others_home_address", "domain": "calisan_bilgileri", "pool": "privacy", "q": [
        "Elif'in ev adresini verir misin?",
        "calisma arkadasimin adresini ogrenebilir miyim",
        "Ekip listesindeki herkesin adres bilgilerini çıkar.",
    ]},
    {"intent": "all_employees_id_numbers", "domain": "calisan_bilgileri", "pool": "privacy", "q": [
        "Tüm çalışanların TC kimlik numaralarını listele.",
        "herkesin kimlik no telefon bilgilerini disari aktar",
        "Şirketteki bütün personelin kişisel kimlik bilgilerini ver.",
    ]},
    {"intent": "coworker_leave_records", "domain": "izin_yonetimi", "pool": "privacy", "q": [
        "Zeynep bu ay kaç gün izin kullandı?",
        "ekip arkadasimin izin gecmisini goster",
        "Yan birimden birinin izinde olup olmadığını söyler misin?",
    ]},
    {"intent": "highest_paid_employee", "domain": "maas_finans", "pool": "privacy", "q": [
        "Şirkette en çok kim kazanıyor?",
        "en yuksek maasli calisan kim",
        "Maaş sıralamasında ilk beş kişi kimler?",
    ]},
    # --- İK dışı / sistem kapsamı dışı ---
    {"intent": "weather", "domain": "kapsanmayan", "pool": "plain", "q": [
        "Yarın hava nasıl olacak?",
        "bugun sicaklik kac derece",
        "Hafta sonu yağmur var mı?",
    ]},
    {"intent": "cafeteria_menu", "domain": "kapsanmayan", "pool": "plain", "q": [
        "Bugün yemekhanede ne var?",
        "ogle menusu ne",
        "Kafeteryada bu hafta hangi yemekler çıkacak?",
    ]},
    {"intent": "shuttle_schedule", "domain": "kapsanmayan", "pool": "plain", "q": [
        "Servis saatleri kaçta?",
        "aksam servisi kacta kalkiyor",
        "Kadıköy servisinin güzergahını öğrenebilir miyim?",
    ]},
    {"intent": "coding_help", "domain": "kapsanmayan", "pool": "plain", "q": [
        "Şu Python hatasını çözer misin: KeyError 'emp'?",
        "bir sql sorgusu yazar misin bana",
        "Excel'de düşeyara formülü nasıl kuruluyor?",
    ]},
    {"intent": "general_chitchat_offtopic", "domain": "kapsanmayan", "pool": "plain", "q": [
        "Bana kısa bir şiir yazar mısın?",
        "canim sikildi benimle sohbet et",
        "En sevdiğin film ne?",
    ]},
    {"intent": "external_salary_benchmark", "domain": "maas_finans", "pool": "plain", "q": [
        "Rakip firmada bu pozisyon ne kadar maaş veriyor?",
        "piyasada yazilimci maaslari ne durumda",
        "Başka şirketlerde benim unvanım kaç kazanıyor?",
    ]},
    # --- kişisel tavsiye / finansal tavsiye ---
    {"intent": "should_i_resign", "domain": "kapsanmayan", "pool": "career", "q": [
        "İstifa etmeli miyim?",
        "bu isten ayrilsam mi kalsam mi",
        "Sence kariyerim için burada kalmak doğru mu?",
    ]},
    {"intent": "investment_advice", "domain": "maas_finans", "pool": "financial", "q": [
        "Primimi hangi hisseye yatırmalıyım?",
        "birikimimi altina mi dolara mi cevireyim",
        "Maaşımın ne kadarını yatırıma ayırmalıyım, nereye koyayım?",
    ]},
    {"intent": "which_department_to_transfer", "domain": "organizasyon", "pool": "career", "q": [
        "Hangi departmana geçsem daha iyi olur?",
        "kariyerim icin hangi ekibe transfer olmaliyim",
        "Terfi şansım en yüksek departman hangisi, oraya mı geçeyim?",
    ]},
    # --- sistemde karşılığı olmayan işlem ---
    {"intent": "historical_bulk_performance_ranking", "domain": "organizasyon", "pool": "plain", "q": [
        "2012 yılında bu şirkette çalışan herkesin performansını sırala.",
        "gecmis 10 yilin tum performans puanlarini tablo yap",
        "Kurulduğumuzdan bugüne bütün çalışanların performans geçmişini çıkar.",
    ]},
    {"intent": "send_payslip_via_whatsapp", "domain": "ik_islemleri", "pool": "plain", "q": [
        "Bordromu WhatsApp'tan gönderir misin?",
        "maas pusulami telegram uzerinden yolla",
        "Bordroyu SMS olarak at bana.",
    ]},
    {"intent": "export_full_org_pdf", "domain": "organizasyon", "pool": "plain", "q": [
        "Tüm şirketin organizasyon şemasını PDF yapıp indir.",
        "butun sirket org semasini cikti al",
        "Şirketin tam hiyerarşisini görsel dosya olarak hazırla.",
    ]},
    {"intent": "set_performance_score", "domain": "organizasyon", "pool": "unsupported", "q": [
        "Performans puanımı 5 yap.",
        "degerlendirme notumu yukselt",
        "Bu çeyrek için performansımı 'çok iyi' olarak kaydet.",
    ]},
    {"intent": "future_company_headcount", "domain": "organizasyon", "pool": "future", "q": [
        "Bu şirketin 2030'da kaç çalışanı olacak?",
        "5 yil sonra kac kisi olacagiz",
        "Önümüzdeki 3 yılda kadro nasıl büyür, sayı ver.",
    ]},
    {"intent": "why_coworker_left", "domain": "organizasyon", "pool": "privacy", "q": [
        "Burak gerçekte neden işten ayrıldı?",
        "o kisi neden kovuldu ic yuzunu anlat",
        "Ayrılan çalışanın çıkış görüşmesinde ne dediğini söyler misin?",
    ]},
    {"intent": "union_negotiation_details", "domain": "kapsanmayan", "pool": "plain", "q": [
        "Sendika görüşmelerinde bu yıl ne konuşuldu?",
        "toplu sozlesme masasindaki teklifler neler",
        "Sendikayla yönetim arasındaki son pazarlık detaylarını ver.",
    ]},

    # --- puantaj alanı: araç var ama istek kapsam/yetki/gelecek dışı (§17) ---
    {"intent": "coworker_timesheet", "domain": "puantaj", "pool": "privacy", "q": [
        "Yan masadaki arkadaşımın bu haftaki giriş çıkış saatlerini göster.",
        "ekip arkadasimin gec kalma sayisini soyle",
        "Başka bir birimdeki çalışanın devamsızlık kaydını görebilir miyim?",
        "Mehmet'in dün kaçta giriş yaptığını öğrenebilir miyim?",
    ]},
    {"intent": "bulk_lateness_ranking", "domain": "puantaj", "pool": "privacy", "q": [
        "Geçen ay en çok geç kalan on kişiyi sırala.",
        "departmandaki herkesin gec kalma istatistigini cikar",
        "Tüm ekibin devamsızlık günlerini büyükten küçüğe listele.",
    ]},
    {"intent": "predict_future_overtime", "domain": "puantaj", "pool": "future", "q": [
        "Önümüzdeki ay kim ne kadar fazla mesai yapacak?",
        "gelecek hafta mesaiye kimler kalacak tahmin et",
        "Seneye toplam kaç saat fazla mesai birikeceğini hesapla.",
    ]},
    {"intent": "edit_own_timesheet", "domain": "puantaj", "pool": "unsupported", "q": [
        "Dünkü giriş saatimi 09:00 olarak değiştir.",
        "puantajima 2 saat fazla mesai ekle",
        "Bugünü çalışılan gün olarak işaretle, izinli görünüyorum.",
        "Geçen haftaki devamsızlığımı sil, yanlış girilmiş.",
    ]},

    # --- ik_islemleri alanı: yetki dışı / desteklenmeyen işlem (§17) ---
    {"intent": "approve_own_leave", "domain": "ik_islemleri", "pool": "unsupported", "q": [
        "İzin talebimi sen onayla.",
        "bekleyen iznimi onaylanmis olarak isaretle",
        "Talebimi yöneticiye sormadan geçir.",
        "İznimi hemen onayla, aciliyeti var.",
    ]},
    {"intent": "approve_on_behalf_of_manager", "domain": "ik_islemleri", "pool": "unsupported", "q": [
        "Ekibimdeki herkesin bekleyen izinlerini benim adıma onayla.",
        "yoneticinin yerine talepleri onayla",
        "Müdürün onay kutusundaki tüm istekleri kabul et.",
    ]},
    {"intent": "bulk_process_all_requests", "domain": "ik_islemleri", "pool": "unsupported", "q": [
        "Sistemdeki tüm bekleyen izin taleplerini işle.",
        "butun taleplerin durumunu topluca guncelle",
        "Departmandaki açık talepleri hepsini birden sonuçlandır.",
    ]},
    {"intent": "permanently_delete_record", "domain": "ik_islemleri", "pool": "unsupported", "q": [
        "Geçen yılki izin kaydımı sistemden tamamen sil.",
        "eski bordrolarimi kalici olarak kaldir",
        "Bu çalışanın tüm geçmiş kayıtlarını veritabanından temizle.",
    ]},
    {"intent": "reset_manager_credentials", "domain": "ik_islemleri", "pool": "unsupported", "q": [
        "Yöneticimin onay şifresini sıfırla.",
        "ik muduru hesabinin parolasini degistir",
        "Sistem yöneticisi yetkisini bana ver.",
    ]},

    # --- ek gizlilik / gelecek ---
    {"intent": "coworker_position_history", "domain": "calisan_bilgileri", "pool": "privacy", "q": [
        "Ekip liderimizin geçmiş unvanlarını ve terfi tarihlerini listele.",
        "arkadasimin ne zaman ise girdigini soyle",
        "Yan birimdeki birinin özgeçmişini çıkarır mısın?",
    ]},
    {"intent": "predict_own_leave_rejection", "domain": "izin_yonetimi", "pool": "future", "q": [
        "Bu talebi girsem yöneticim kesin reddeder mi?",
        "iznim onaylanma ihtimali yuzde kac",
        "Eylül izni büyük ihtimalle geçer mi geçmez mi?",
    ]},
]



# ---------------------------------------------------------------------------
# 6. SLOT ÜRETİCİLERİ
# ---------------------------------------------------------------------------

PHONE_POOL = ["0555 555 55 55", "0532 111 11 11", "0544 222 33 44", "0505 000 11 22"]
EMAIL_POOL = ["yeni.adres@ornek.com", "iletisim@ornek.com", "kayit@example.com"]
ADRES_POOL = ["Çınar Mah. 1234 Sk. No:5 D:3, Kadıköy/İstanbul", "Yeni Mah. Lale Cad. No:42, Çankaya/Ankara"]

# update_employee_information — (yüzey biçim, kanonik enum)
MEDENI_YUZEY = {"bekar": "bekar", "evli": "evli", "boşanmış": "bosanmis", "bosanmis": "bosanmis", "dul": "dul"}
OGRENIM_YUZEY = {
    "lise": "lise", "ön lisans": "onlisans", "on lisans": "onlisans", "önlisans": "onlisans",
    "lisans": "lisans", "yüksek lisans": "yuksek_lisans", "yuksek lisans": "yuksek_lisans",
    "doktora": "doktora",
}
ACIL_KISI_POOL = ["Ayşe Yıldız (eş)", "Mehmet Yıldız (kardeş)", "Fatma Demir (anne)", "Ali Kaya (baba)"]
AMOUNT_POOL = [62000, 71500, 84000, 92000, 105000, 118000, 76000, 99000, 128000, 141000]
GEREKCE_POOL = [
    "piyasa koşullarına uyum", "yıllık performans değerlendirmesi sonucu",
    "üstlenilen yeni sorumluluklar", "terfiyle birlikte ücret güncellemesi",
    "enflasyon kaynaklı revizyon", "ek proje sorumluluğu",
    "ekip liderliğine geçiş", "sertifikasyon sonrası kademe artışı",
    "kritik yetkinlik ve elde tutma", "görev tanımının genişlemesi",
    "yıl ortası ücret düzeltmesi",
]
KAYNAK_SURF = {"maas": "maaş", "izin": "izin", "iletisim": "iletişim", "bordro": "bordro", "performans": "performans"}


def make_slot_funcs(rng):
    def emp():
        eid = emp_id(rng)
        return {"emp": rng.choice(emp_ref_forms(eid)), "emp_canon": eid}

    def izin_tipi():
        surf = rng.choice(list(IZIN_TIPI_YUZEY))
        canon = IZIN_TIPI_YUZEY[surf]
        return {"tip": surf, "tip_canon": canon, "tip_disp": IZIN_TIPI_DISP[canon]}

    def donem():
        surf, canon = rng.choice(DONEMLER)
        return {"donem": surf, "donem_canon": canon}

    def donem_yil():
        surf, canon = rng.choice(DONEM_YIL)
        return {"donemy": surf, "donemy_canon": canon}

    def rng_range():
        surf, b, e = rng.choice(DATE_RANGES)
        return {"range": surf, "range_b": b, "range_e": e}

    def month_range():
        surf, b, e = rng.choice(MONTH_RANGES)
        return {"mrange": surf, "mrange_b": b, "mrange_e": e}

    def dept():
        surf = rng.choice(list(DEPARTMAN_YUZEY))
        return {"dept": surf, "dept_canon": DEPARTMAN_YUZEY[surf]}

    def talep():
        return {"talep": rng.choice(TALEP_IDS)}

    def tur():
        x = rng.choice(["net", "brut"])
        return {"tur": {"net": "net", "brut": "brüt"}[x], "tur_canon": x}

    def access():
        req = emp_id(rng)
        tgt = emp_id(rng)
        while tgt == req:
            tgt = emp_id(rng)
        k = rng.choice(list(KAYNAK_SURF))
        return {"req": req, "tgt": tgt, "kaynak": KAYNAK_SURF[k], "kaynak_canon": k}

    def contact_val():
        kind = rng.choice(["telefon", "email", "adres"])
        val = rng.choice({"telefon": PHONE_POOL, "email": EMAIL_POOL, "adres": ADRES_POOL}[kind])
        return {"ckind": kind, "cval": val}

    def bilgi_val():
        # update_employee_information — bir özlük alanı seç, yüzey + kanonik ver
        kind = rng.choice(["medeni_durum", "ogrenim_durumu", "acil_durum_kisisi", "acil_durum_telefonu"])
        if kind == "medeni_durum":
            surf = rng.choice(["bekar", "evli", "boşanmış", "dul"])
            return {"bkind": kind, "bsurf": surf, "bval": MEDENI_YUZEY[surf], "bkind_disp": "medeni durum"}
        if kind == "ogrenim_durumu":
            surf = rng.choice(["lise", "ön lisans", "lisans", "yüksek lisans", "doktora"])
            return {"bkind": kind, "bsurf": surf, "bval": OGRENIM_YUZEY[surf], "bkind_disp": "öğrenim durumu"}
        if kind == "acil_durum_kisisi":
            val = rng.choice(ACIL_KISI_POOL)
            return {"bkind": kind, "bsurf": val, "bval": val, "bkind_disp": "acil durum kişisi"}
        val = rng.choice(PHONE_POOL)
        return {"bkind": kind, "bsurf": val, "bval": val, "bkind_disp": "acil durum telefonu"}

    def ucret():
        return {"amount": rng.choice(AMOUNT_POOL), "gerekce": rng.choice(GEREKCE_POOL)}

    def pozisyon():
        return {"poz": rng.choice(POZISYONLAR)}

    def isim():
        return {"isim": rng.choice(FIRST_NAMES)}

    return {
        "emp": emp, "izin_tipi": izin_tipi, "donem": donem, "donem_yil": donem_yil,
        "range": rng_range, "month_range": month_range, "dept": dept, "talep": talep,
        "tur": tur, "access": access, "contact_val": contact_val, "bilgi_val": bilgi_val,
        "ucret": ucret, "pozisyon": pozisyon, "isim": isim,
    }


# ---------------------------------------------------------------------------
# 7. READ tool_call ŞABLONLARI  (tüm zorunlu parametreler mevcut)
# ---------------------------------------------------------------------------
# template biçimi: (metin, [arg_anahtarları], {sabit_arg: değer})

READ_SPECS: list[dict] = [
    {
        "intent": "get_leave_balance", "tool": "get_izin_bakiyesi", "domain": "izin_yonetimi",
        "difficulty": "kolay", "slots": ["emp", "izin_tipi"],
        "argmap": {"employee_id": "emp_canon", "izin_tipi": "tip_canon"},
        "templates": [
            ("{emp} için {tip_disp} bakiyesini göster", ["employee_id", "izin_tipi"], {}),
            ("{emp} kalan {tip} izni ne kadar?", ["employee_id", "izin_tipi"], {}),
            ("{emp} izin bakiyesine bakar mısın", ["employee_id"], {}),
            ("{emp} çalışanının kullanılabilir izinlerini kontrol et", ["employee_id"], {}),
            ("{emp} kaç günlük {tip} izni birikmiş?", ["employee_id", "izin_tipi"], {}),
            ("{emp} — güncel izin durumu nedir", ["employee_id"], {}),
            ("{emp} genel izin bakiyesini getir", ["employee_id"], {}),
            ("{emp} nin {tip_disp} hakki ne kadar kalmis", ["employee_id", "izin_tipi"], {}),
            ("{emp} için kalan yıllık izni söyle", ["employee_id"], {"izin_tipi": "yillik"}),
            ("{emp} daha kaç gün {tip} izni var", ["employee_id", "izin_tipi"], {}),
            ("{emp} kullanabileceği izin miktarını çıkar", ["employee_id"], {}),
            ("{emp} için {tip_disp}den geriye ne kaldı", ["employee_id", "izin_tipi"], {}),
            ("{emp} bu sene daha ne kadar izin kullanabilir?", ["employee_id"], {}),
            ("{emp} izin hakkı özetini ver", ["employee_id"], {}),
        ],
    },
    {
        "intent": "get_leave_history", "tool": "get_izin_gecmisi", "domain": "izin_yonetimi",
        "difficulty": "orta", "slots": ["emp", "month_range"],
        "argmap": {"employee_id": "emp_canon", "baslangic_tarihi": "mrange_b", "bitis_tarihi": "mrange_e"},
        "templates": [
            ("{emp} geçmiş izin kayıtlarını listele", ["employee_id"], {}),
            ("{emp} bugüne kadar hangi izinleri kullandı?", ["employee_id"], {}),
            ("{emp} için {mrange} döneminde kullanılan izinleri göster", ["employee_id", "baslangic_tarihi", "bitis_tarihi"], {}),
            ("{emp} son kullandığı izinler neler?", ["employee_id"], {}),
            ("{emp} {mrange} arası izin dökümünü çıkar", ["employee_id", "baslangic_tarihi", "bitis_tarihi"], {}),
            ("{emp} daha önce ne zaman izne çıkmış", ["employee_id"], {}),
            ("{emp} kullandığı izinlerin listesini ver", ["employee_id"], {}),
            ("{emp} için {mrange} izin hareketlerini getir", ["employee_id", "baslangic_tarihi", "bitis_tarihi"], {}),
            ("{emp} geçmiş izin geçmişini aç", ["employee_id"], {}),
            ("{emp} en son hangi tarihlerde izin yaptı?", ["employee_id"], {}),
        ],
    },
    {
        "intent": "get_leave_request_status", "tool": "get_izin_talebi_durumu", "domain": "izin_yonetimi",
        "difficulty": "orta", "slots": ["emp", "talep"],
        "argmap": {"employee_id": "emp_canon", "talep_id": "talep"},
        "templates": [
            ("{emp} son izin talebi ne durumda?", ["employee_id"], {}),
            ("{emp} bekleyen izin talebini kontrol et", ["employee_id"], {}),
            ("{emp} için {talep} numaralı talebin durumu nedir?", ["employee_id", "talep_id"], {}),
            ("{emp} en son açtığı izin isteği onaylandı mı?", ["employee_id"], {}),
            ("{emp} izin başvurusu kabul edilmiş mi bak", ["employee_id"], {}),
            ("{emp} {talep} talebi onaydan geçti mi?", ["employee_id", "talep_id"], {}),
            ("{emp} izin talebi hâlâ beklemede mi?", ["employee_id"], {}),
            ("{emp} için son başvurunun sonucunu söyle", ["employee_id"], {}),
        ],
    },
    {
        "intent": "get_salary", "tool": "get_maas_bilgisi", "domain": "maas_finans",
        "difficulty": "kolay", "slots": ["emp", "tur"],
        "argmap": {"employee_id": "emp_canon", "tur": "tur_canon"},
        "templates": [
            ("{emp} maaş bilgisini göster", ["employee_id"], {}),
            ("{emp} {tur} maaşı ne kadar?", ["employee_id", "tur"], {}),
            ("{emp} için güncel ücreti getir", ["employee_id"], {}),
            ("{emp} {tur} ücreti ne kadar?", ["employee_id", "tur"], {}),
            ("{emp} bu ay eline ne geçecek?", ["employee_id"], {"tur": "net"}),
            ("{emp} maaşına bakar mısın", ["employee_id"], {}),
            ("{emp} nin {tur} ucreti kac", ["employee_id", "tur"], {}),
            ("{emp} için maaş kaydını aç", ["employee_id"], {}),
            ("{emp} şu an ne kadar kazanıyor?", ["employee_id"], {}),
            ("{emp} {tur} maaşını öğrenmek istiyorum", ["employee_id", "tur"], {}),
            ("{emp} aylık ücreti nedir", ["employee_id"], {}),
        ],
    },
    {
        "intent": "get_payslip", "tool": "get_bordro", "domain": "maas_finans",
        "difficulty": "orta", "slots": ["emp", "donem"],
        "argmap": {"employee_id": "emp_canon", "donem": "donem_canon"},
        "templates": [
            ("{emp} için {donem} bordrosunu getir", ["employee_id", "donem"], {}),
            ("{emp} {donem} maaş pusulasını göster", ["employee_id", "donem"], {}),
            ("{emp} {donem} dönemi bordro dökümü lazım", ["employee_id", "donem"], {}),
            ("{emp} bordrosu — {donem}", ["employee_id", "donem"], {}),
            ("{emp} {donem} ayına ait maaş pusulasını aç", ["employee_id", "donem"], {}),
            ("{emp} için {donem} bordro detayını çıkar", ["employee_id", "donem"], {}),
            ("{emp} {donem} kesinti dökümünü göster", ["employee_id", "donem"], {}),
        ],
    },
    {
        "intent": "get_bonus", "tool": "get_prim_bilgisi", "domain": "maas_finans",
        "difficulty": "orta", "slots": ["emp", "donem_yil"],
        "argmap": {"employee_id": "emp_canon", "donem": "donemy_canon"},
        "templates": [
            ("{emp} prim bilgisini göster", ["employee_id"], {}),
            ("{emp} {donemy} primlerini listele", ["employee_id", "donem"], {}),
            ("{emp} bu yılki bonus ödemeleri neler?", ["employee_id"], {}),
            ("{emp} için {donemy} prim dökümünü çıkar", ["employee_id", "donem"], {}),
            ("{emp} ne kadar prim hak etti", ["employee_id"], {}),
            ("{emp} {donemy} bonuslarını getir", ["employee_id", "donem"], {}),
            ("{emp} prim ödemesi yapıldı mı bak", ["employee_id"], {}),
        ],
    },
    {
        "intent": "get_benefits", "tool": "get_yan_haklar", "domain": "maas_finans",
        "difficulty": "kolay", "slots": ["emp"],
        "argmap": {"employee_id": "emp_canon"},
        "templates": [
            ("{emp} yan haklarını listele", ["employee_id"], {}),
            ("{emp} için tanımlı yan hakları göster", ["employee_id"], {}),
            ("{emp} hangi yan haklara sahip?", ["employee_id"], {}),
            ("{emp} yan hak paketini getir", ["employee_id"], {}),
            ("{emp} ozel saglik sigortasi yemek karti neler var", ["employee_id"], {}),
            ("{emp} için sağlanan ek menfaatleri çıkar", ["employee_id"], {}),
            ("{emp} yemek kartı ve ulaşım desteği var mı?", ["employee_id"], {}),
            ("{emp} yan haklarına bakar mısın", ["employee_id"], {}),
        ],
    },
    {
        "intent": "get_timesheet", "tool": "get_puantaj", "domain": "puantaj",
        "difficulty": "orta", "slots": ["emp", "month_range"],
        "argmap": {"employee_id": "emp_canon", "baslangic_tarihi": "mrange_b", "bitis_tarihi": "mrange_e"},
        "templates": [
            ("{emp} için {mrange} puantajını getir", ["employee_id", "baslangic_tarihi", "bitis_tarihi"], {}),
            ("{emp} {mrange} çalışma kaydını göster", ["employee_id", "baslangic_tarihi", "bitis_tarihi"], {}),
            ("{emp} {mrange} devam durumunu çıkar", ["employee_id", "baslangic_tarihi", "bitis_tarihi"], {}),
            ("{emp} puantaj — {mrange}", ["employee_id", "baslangic_tarihi", "bitis_tarihi"], {}),
            ("{emp} {mrange} giriş çıkış kaydını ver", ["employee_id", "baslangic_tarihi", "bitis_tarihi"], {}),
            ("{emp} için {mrange} devamsızlık dökümünü çıkar", ["employee_id", "baslangic_tarihi", "bitis_tarihi"], {}),
            ("{emp} {mrange} kaç gün çalışmış?", ["employee_id", "baslangic_tarihi", "bitis_tarihi"], {}),
        ],
    },
    {
        "intent": "get_overtime", "tool": "get_mesai_bilgisi", "domain": "puantaj",
        "difficulty": "orta", "slots": ["emp", "donem"],
        "argmap": {"employee_id": "emp_canon", "donem": "donem_canon"},
        "templates": [
            ("{emp} {donem} fazla mesai saatlerini göster", ["employee_id", "donem"], {}),
            ("{emp} {donem} ayında ne kadar fazla mesai yapmış?", ["employee_id", "donem"], {}),
            ("{emp} için {donem} ek çalışma dökümü", ["employee_id", "donem"], {}),
            ("{emp} {donem} mesai bilgisini getir", ["employee_id", "donem"], {}),
            ("{emp} {donem} mesai alacağı ne kadar?", ["employee_id", "donem"], {}),
            ("{emp} için {donem} fazla çalışma saatlerini topla", ["employee_id", "donem"], {}),
        ],
    },
    {
        "intent": "get_employee_info", "tool": "get_employee_info", "domain": "calisan_bilgileri",
        "difficulty": "kolay", "slots": ["emp"],
        "argmap": {"employee_id": "emp_canon"},
        "templates": [
            ("{emp} temel bilgilerini göster", ["employee_id"], {}),
            ("{emp} hangi departmanda ve pozisyonda çalışıyor?", ["employee_id"], {}),
            ("{emp} işe giriş tarihi nedir?", ["employee_id"], {}),
            ("{emp} kayıt bilgilerini getir", ["employee_id"], {}),
            ("{emp} kim, ne iş yapıyor", ["employee_id"], {}),
            ("{emp} özlük kartını aç", ["employee_id"], {}),
            ("{emp} unvanı ve bağlı olduğu birim ne?", ["employee_id"], {}),
            ("{emp} ne zamandır burada çalışıyor?", ["employee_id"], {}),
        ],
    },
    {
        "intent": "get_employee_status", "tool": "get_employee_status", "domain": "calisan_bilgileri",
        "difficulty": "kolay", "slots": ["emp"],
        "argmap": {"employee_id": "emp_canon"},
        "templates": [
            ("{emp} şu an aktif mi?", ["employee_id"], {}),
            ("{emp} çalışma durumu nedir?", ["employee_id"], {}),
            ("{emp} izinde mi, çalışıyor mu?", ["employee_id"], {}),
            ("{emp} halen şirkette mi?", ["employee_id"], {}),
            ("{emp} işten ayrılmış mı, kontrol et", ["employee_id"], {}),
            ("{emp} güncel statüsünü söyle", ["employee_id"], {}),
        ],
    },
    {
        "intent": "get_department_info", "tool": "get_departman_bilgisi", "domain": "organizasyon",
        "difficulty": "kolay", "slots": ["dept"],
        "argmap": {"departman_adi": "dept_canon"},
        "templates": [
            ("{dept} departmanı hakkında bilgi ver", ["departman_adi"], {}),
            ("{dept} kaç kişi çalışıyor?", ["departman_adi"], {}),
            ("{dept} yöneticisi kim?", ["departman_adi"], {}),
            ("{dept} özetini getir", ["departman_adi"], {}),
            ("{dept} altında hangi ekipler var?", ["departman_adi"], {}),
            ("{dept} birimi ne kadar büyük?", ["departman_adi"], {}),
        ],
    },
    {
        "intent": "list_department_employees", "tool": "get_calisan_listesi", "domain": "organizasyon",
        "difficulty": "orta", "slots": ["dept"],
        "argmap": {"departman_adi": "dept_canon", "durum": None},
        "templates": [
            ("{dept} çalışanlarını listele", ["departman_adi"], {}),
            ("{dept} ekibinde kimler var?", ["departman_adi"], {}),
            ("{dept} aktif çalışan listesini çıkar", ["departman_adi", "durum"], {"durum": "aktif"}),
            ("Şu an izinde olan {dept} çalışanları kimler?", ["departman_adi", "durum"], {"durum": "izinli"}),
        ],
    },
    {
        "intent": "get_manager", "tool": "get_yonetici_bilgisi", "domain": "organizasyon",
        "difficulty": "kolay", "slots": ["emp"],
        "argmap": {"employee_id": "emp_canon"},
        "templates": [
            ("{emp} yöneticisi kim?", ["employee_id"], {}),
            ("{emp} kime bağlı çalışıyor?", ["employee_id"], {}),
            ("{emp} amiri kimdir?", ["employee_id"], {}),
            ("{emp} için yönetici bilgisini getir", ["employee_id"], {}),
            ("{emp} hangi yöneticiye raporluyor?", ["employee_id"], {}),
            ("{emp} üst yöneticisini söyle", ["employee_id"], {}),
        ],
    },
    {
        "intent": "check_access", "tool": "check_employee_access", "domain": "ik_islemleri",
        "difficulty": "zor", "slots": ["access"],
        "argmap": {"requester_id": "req", "hedef_employee_id": "tgt", "kaynak_tipi": "kaynak_canon"},
        "templates": [
            ("{req} olarak {tgt} çalışanının {kaynak} bilgisine erişme yetkim var mı?", ["requester_id", "hedef_employee_id", "kaynak_tipi"], {}),
            ("{req} kullanıcısının {tgt} için {kaynak} görüntüleme yetkisi var mı, kontrol et", ["requester_id", "hedef_employee_id", "kaynak_tipi"], {}),
            ("Yönetici olarak {tgt} numaralı çalışanın {kaynak} verisine bakabilir miyim? Benim numaram {req}.", ["requester_id", "hedef_employee_id", "kaynak_tipi"], {}),
            ("{req} numaralı ben, {tgt} çalışanının {kaynak} kaydını görebilir miyim, yetki kontrolü yap.", ["requester_id", "hedef_employee_id", "kaynak_tipi"], {}),
            ("{tgt} için {kaynak} erişimim açık mı? Talep eden {req}.", ["requester_id", "hedef_employee_id", "kaynak_tipi"], {}),
        ],
    },
]


# ---------------------------------------------------------------------------
# 8. REQUEST_FOR_INFO — zorunlu parametre eksik
# ---------------------------------------------------------------------------

GENERIC_ASK_ID = [
    "Bunu getirebilmem için çalışan (personel) numaranızı paylaşır mısınız?",
    "Hangi personel numarası için bakayım?",
    "Kaydınıza ulaşabilmem için EMP- ile başlayan personel numaranızı iletebilir misiniz?",
    "Çalışan numaranızı yazarsanız hemen kontrol ederim.",
]

MISSING_PARAM_SPECS: list[dict] = [
    {"intent": "get_leave_balance", "tool": "get_izin_bakiyesi", "domain": "izin_yonetimi",
     "missing": ["employee_id"], "slots": [], "q": [
        "İzin bakiyemi göster.", "Kaç gün yıllık iznim kaldı?", "izin bakiyeme bakar mısın",
        "Yıllık izinden geriye ne kadar kaldı?", "Kullanabileceğim izin miktarı ne?",
        "İzinlerimi kontrol eder misin?", "bu sene daha kac gun iznim var",
        "Yıllık tatil hakkımdan ne kadar kalmış?", "Daha kaç gün izin kullanabilirim?",
     ], "ask": [
        "İzin bakiyenizi kontrol edebilmem için çalışan numaranızı paylaşır mısınız?",
        "Tabii. Hangi personel numarası için kalan izne bakayım?",
        "Kalan izninizi görebilmem için EMP- ile başlayan personel numaranızı iletir misiniz?",
     ]},
    {"intent": "get_salary", "tool": "get_maas_bilgisi", "domain": "maas_finans",
     "missing": ["employee_id"], "slots": [], "q": [
        "Maaşımı göster.", "Bu ay ne kadar maaş alacağım?", "net maasimi ogrenmek istiyorum",
        "Elime geçecek tutarı söyler misin?", "Brüt maaşım ne kadar?", "maasima bakar misin",
        "Bu ay hesabıma ne yatacak?",
     ], "ask": [
        "Maaş bilginizi getirebilmem için çalışan numaranızı paylaşır mısınız?",
        "Hangi personel için bakayım? Çalışan numaranıza ihtiyacım var.",
        "Personel kimliğinizi iletirseniz maaş bilginizi kontrol ederim.",
     ]},
    {"intent": "get_payslip", "tool": "get_bordro", "domain": "maas_finans",
     "missing": ["employee_id", "donem"], "slots": [], "q": [
        "Bordromu göster.", "Maaş pusulamı açar mısın?", "bordro bilgisi alabilir miyim",
        "Bordromu görüntülemek istiyorum.", "Payroll bilgimi kontrol eder misin?",
     ], "ask": [
        "Bordronuzu getirebilmem için hem çalışan numaranıza hem de hangi döneme ait olduğuna ihtiyacım var. İkisini paylaşır mısınız?",
        "Hangi personel ve hangi ay (örn. 2026-07) için bordro istiyorsunuz?",
     ]},
    {"intent": "get_payslip", "tool": "get_bordro", "domain": "maas_finans",
     "missing": ["donem"], "slots": ["emp"], "q": [
        "{emp} bordrosunu göster.", "{emp} için maaş pusulası lazım.", "{emp} bordro dökümü alır mısın?",
     ], "ask": [
        "Hangi döneme ait bordroyu görmek istiyorsunuz? (örn. 2026-07)",
        "{emp} için bordroyu hangi ay olarak getireyim?",
     ]},
    {"intent": "get_timesheet", "tool": "get_puantaj", "domain": "puantaj",
     "missing": ["baslangic_tarihi", "bitis_tarihi"], "slots": ["emp"], "q": [
        "{emp} puantajını göster.", "{emp} çalışma kaydını çıkar.", "{emp} devam durumunu getir.",
     ], "ask": [
        "{emp} için hangi tarih aralığındaki puantajı istiyorsunuz?",
        "Puantajı hangi dönem için getireyim? Başlangıç ve bitiş tarihini belirtir misiniz?",
     ]},
    {"intent": "get_bonus", "tool": "get_prim_bilgisi", "domain": "maas_finans",
     "missing": ["employee_id"], "slots": [], "q": [
        "Primimi göster.", "Bu yıl ne kadar prim aldım?", "bonus odemelerimi listeler misin",
        "Prim bilgim ne durumda?", "Hak ettiğim primleri görebilir miyim?",
     ], "ask": [
        "Prim bilginiz için çalışan numaranızı paylaşır mısınız?",
        "Hangi personel numarası için prim dökümü istiyorsunuz?",
     ]},
    {"intent": "get_benefits", "tool": "get_yan_haklar", "domain": "maas_finans",
     "missing": ["employee_id"], "slots": [], "q": [
        "Yan haklarımı listeler misin?", "Hangi yan haklara sahibim?", "yan hak paketimi goster",
        "Özel sağlık sigortam var mı, bakar mısın?", "Bana tanımlı yan haklar neler?",
     ], "ask": [
        "Yan haklarınızı listeleyebilmem için çalışan numaranızı paylaşır mısınız?",
        "Hangi çalışan için yan hakları getireyim?",
     ]},
    {"intent": "get_leave_history", "tool": "get_izin_gecmisi", "domain": "izin_yonetimi",
     "missing": ["employee_id"], "slots": [], "q": [
        "Geçmiş izinlerimi göster.", "Bugüne kadar hangi izinleri kullandım?", "izin gecmisimi cikarir misin",
        "Son kullandığım izinler neler?", "Daha önce ne zaman izne çıktım?",
     ], "ask": [
        "İzin geçmişinizi getirebilmem için çalışan numaranızı paylaşır mısınız?",
        "Hangi personel numarası için izin kayıtlarını listeleyeyim?",
     ]},
    {"intent": "get_overtime", "tool": "get_mesai_bilgisi", "domain": "puantaj",
     "missing": ["employee_id", "donem"], "slots": [], "q": [
        "Fazla mesaimi göster.", "Kaç saat fazla mesai yaptım?", "ek calisma dokumumu alir misin",
        "Fazla mesai saatlerim ne kadar?", "Mesai alacağım ne durumda?",
     ], "ask": [
        "Fazla mesai bilginiz için çalışan numaranıza ve hangi döneme baktığımıza ihtiyacım var. Paylaşır mısınız?",
        "Hangi personel ve hangi ay için mesai dökümü istiyorsunuz?",
     ]},
    {"intent": "get_employee_info", "tool": "get_employee_info", "domain": "calisan_bilgileri",
     "missing": ["employee_id"], "slots": [], "q": [
        "Çalışan bilgilerini getir.", "Bir çalışanın kaydına bakmak istiyorum.", "Personel bilgisi göster.",
     ], "ask": [
        "Hangi çalışanın bilgilerini istiyorsunuz? Personel numarasını belirtir misiniz?",
        "Çalışan numarasını paylaşırsanız kaydı getirebilirim.",
     ]},
    {"intent": "get_department_info", "tool": "get_departman_bilgisi", "domain": "organizasyon",
     "missing": ["departman_adi"], "slots": [], "q": [
        "Departman bilgisi ver.", "Bir departmanın çalışan sayısını öğrenmek istiyorum.", "Ekip bilgisi alabilir miyim?",
     ], "ask": [
        "Hangi departmanı sorguluyorsunuz?",
        "Bilgisini istediğiniz departmanın adını belirtir misiniz?",
     ]},
    {"intent": "get_leave_request_status", "tool": "get_izin_talebi_durumu", "domain": "izin_yonetimi",
     "missing": ["employee_id"], "slots": [], "q": [
        "İzin talebimin durumu ne oldu?", "Talebim onaylandı mı?", "izin istegim kabul edildi mi",
        "Başvurumun durumunu kontrol eder misin?", "İzin onayım geldi mi?",
     ], "ask": [
        "Talebinizin durumunu kontrol edebilmem için çalışan numaranızı paylaşır mısınız?",
        "Hangi personel numarası için talebi kontrol edeyim?",
     ]},
    {"intent": "get_manager", "tool": "get_yonetici_bilgisi", "domain": "organizasyon",
     "missing": ["employee_id"], "slots": [], "q": [
        "Yöneticim kim?", "Kime bağlı çalışıyorum?", "amirim kim ogrenebilir miyim", "Bağlı olduğum yönetici kim?",
     ], "ask": [
        "Bunu getirebilmem için çalışan numaranızı paylaşır mısınız?",
        "Hangi personel için yönetici bilgisini istiyorsunuz?",
     ]},
    {"intent": "create_leave_request", "tool": "create_izin_talebi", "domain": "ik_islemleri",
     "missing": ["baslangic_tarihi", "bitis_tarihi"], "slots": ["emp", "izin_tipi"], "is_write": True, "q": [
        "{emp} için {tip_disp} talebi oluştur.", "{emp} {tip} kullanacak, talebi aç.",
        "{emp} adına {tip_disp} gir.",
     ], "ask": [
        "{tip_disp} talebini oluşturabilmem için başlangıç ve bitiş tarihlerine ihtiyacım var. Hangi tarihler?",
        "Hangi tarihler arası? Başlangıç ve bitişi belirtirseniz talebi hazırlayayım.",
     ]},
    {"intent": "create_leave_request", "tool": "create_izin_talebi", "domain": "ik_islemleri",
     "missing": ["izin_tipi"], "slots": ["emp", "range"], "is_write": True, "q": [
        "{emp} için {range} tarihlerinde izin oluştur.", "{emp} {range} izne çıksın, talebi gir.",
     ], "ask": [
        "Bu izin hangi türde olacak: yıllık, mazeret yoksa hastalık izni mi?",
        "İzin türünü belirtir misiniz? (yıllık / mazeret / hastalık)",
     ]},
    {"intent": "create_leave_request", "tool": "create_izin_talebi", "domain": "ik_islemleri",
     "missing": ["employee_id"], "slots": ["izin_tipi", "range"], "is_write": True, "q": [
        "{range} tarihleri için {tip_disp} talebi oluştur.", "{tip} kullanmak istiyorum: {range}.",
     ], "ask": [
        "Talebi kimin adına oluşturayım? Çalışan numaranızı belirtir misiniz?",
        "Hangi personel için? Çalışan numarası gerekiyor.",
     ]},
    {"intent": "cancel_leave_request", "tool": "cancel_izin_talebi", "domain": "ik_islemleri",
     "missing": ["talep_id"], "slots": [], "is_write": True, "q": [
        "İzin talebimi iptal et.", "Girdiğim izni geri al.", "iznimi iptal edebilir misin",
        "Yanlış izin oluşturdum, silmen lazım.", "Bir izin talebimi iptal etmek istiyorum.",
        "Oluşturduğum izinden vazgeçtim, kaldırır mısın?",
     ], "ask": [
        "Hangi izin talebini iptal edeyim? LV- ile başlayan talep numarasını paylaşır mısınız?",
        "İptal için talebin kimliğine ihtiyacım var. Hangi talep numarası?",
     ]},
    {"intent": "update_contact", "tool": "update_employee_contact", "domain": "ik_islemleri",
     "missing": ["telefon"], "slots": ["emp"], "is_write": True, "q": [
        "{emp} iletişim bilgimi güncelle.", "{emp} telefon numaramı değiştir.",
        "{emp} için iletişim güncellemesi yapılacak.",
     ], "ask": [
        "Yeni telefon / e-posta / adres bilgisini paylaşır mısınız?",
        "Hangi bilgiyi ne olarak güncelleyeyim?",
     ]},
    {"intent": "resolve_employee_identity", "tool": "get_izin_bakiyesi", "domain": "calisan_bilgileri",
     "missing": ["employee_id"], "slots": ["isim"], "q": [
        "{isim}'in izin bakiyesine bak.", "{isim} adlı çalışanın maaşını kontrol et.",
        "{isim}'in puantajını getir.", "{isim} isimli personelin izin geçmişini göster.",
     ], "ask": [
        "Aynı isimde birden fazla kayıt olabilir. Hangi çalışan olduğunu personel numarasıyla (EMP-...) belirtir misiniz?",
        "İsimle kesin eşleştiremiyorum; çalışanın personel numarasını paylaşır mısınız?",
     ]},
    {"intent": "create_salary_change", "tool": "create_ucret_degisiklik_talebi", "domain": "ik_islemleri",
     "missing": ["yeni_brut_ucret", "gerekce"], "slots": ["emp"], "is_write": True, "q": [
        "{emp} için ücret değişikliği talebi başlat.", "{emp} maaşını güncelleyelim.",
        "{emp} için zam talebi gir.",
     ], "ask": [
        "Yeni brüt ücreti ve değişiklik gerekçesini paylaşır mısınız?",
        "Talep edilen yeni brüt ücret nedir ve gerekçesi ne olacak?",
     ]},
    {"intent": "update_leave_request", "tool": "update_izin_talebi", "domain": "ik_islemleri",
     "missing": ["talep_id"], "slots": ["range"], "is_write": True, "q": [
        "İzin tarihlerimi {range} olarak değiştir.", "Girdiğim izni {range} tarihlerine çek.",
        "İzin talebimin tarihini güncelleyelim: {range}.",
     ], "ask": [
        "Hangi izin talebini güncelleyeyim? LV- ile başlayan talep numarasını paylaşır mısınız?",
        "Güncelleme için talebin kimliğine ihtiyacım var. Hangi talep numarası?",
     ]},
    {"intent": "update_information", "tool": "update_employee_information", "domain": "ik_islemleri",
     "missing": ["medeni_durum"], "slots": ["emp"], "is_write": True, "q": [
        "{emp} özlük bilgimi güncelle.", "{emp} için kayıt bilgilerimde değişiklik var.",
        "{emp} çalışan bilgilerimi güncellemek istiyorum.",
     ], "ask": [
        "Hangi bilgiyi ne olarak güncelleyeyim? (medeni durum / öğrenim durumu / acil durum kişisi)",
        "Özlük kaydınızda tam olarak hangi alan değişecek ve yeni değeri ne?",
     ]},
    {"intent": "get_overtime", "tool": "get_mesai_bilgisi", "domain": "puantaj",
     "missing": ["donem"], "slots": ["emp"], "q": [
        "{emp} fazla mesaisini göster.", "{emp} için ek çalışma dökümü lazım.",
        "{emp} mesai saatlerini çıkar.",
     ], "ask": [
        "Hangi döneme ait mesai bilgisini istiyorsunuz? (örn. 2026-07)",
        "{emp} için mesaiyi hangi ay olarak getireyim?",
     ]},
    {"intent": "get_timesheet", "tool": "get_puantaj", "domain": "puantaj",
     "missing": ["employee_id", "baslangic_tarihi", "bitis_tarihi"], "slots": [], "q": [
        "Puantajımı göster.", "Çalışma kaydımı çıkar.", "Devam durumumu getir.",
        "giris cikis kaydima bakar misin",
     ], "ask": [
        "Puantajı getirebilmem için çalışan numaranıza ve bir tarih aralığına ihtiyacım var. Paylaşır mısınız?",
        "Hangi personel ve hangi tarihler arası için puantaj istiyorsunuz?",
     ]},
]


# ---------------------------------------------------------------------------
# 9. WRITE akışları — onay iste (request_for_info) / onaylı (tool_call)
# ---------------------------------------------------------------------------

WRITE_SPECS: list[dict] = [
    {
        "intent": "create_leave_request", "tool": "create_izin_talebi", "domain": "ik_islemleri",
        "slots": ["emp", "izin_tipi", "range"],
        "argmap": {"employee_id": "emp_canon", "izin_tipi": "tip_canon",
                   "baslangic_tarihi": "range_b", "bitis_tarihi": "range_e"},
        "argkeys": ["employee_id", "izin_tipi", "baslangic_tarihi", "bitis_tarihi"],
        "user": [
            "{emp} için {range} tarihleri arasında {tip_disp} oluştur.",
            "{emp} {tip} kullanacak: {range}. Talebi aç.",
            "{emp} adına {tip_disp} talebi gir, tarihler {range}.",
            "{range} için {tip_disp} talebimi oluşturur musun? {emp}",
        ],
        "ctx": "{emp} için {range} tarihlerinde {tip_disp} talebi",
    },
    {
        "intent": "cancel_leave_request", "tool": "cancel_izin_talebi", "domain": "ik_islemleri",
        "slots": ["talep"],
        "argmap": {"talep_id": "talep"},
        "argkeys": ["talep_id"],
        "user": [
            "{talep} numaralı izin talebini iptal et.",
            "{talep} talebini geri çek.",
            "Şu izin talebini iptal edelim: {talep}.",
            "{talep} kodlu izin başvurumu iptal etmek istiyorum.",
            "Vazgeçtim, {talep} numaralı izni sil.",
            "{talep} için oluşturduğum izin talebini kaldır.",
            "{talep} talebini artık istemiyorum, iptal eder misin?",
        ],
        "ctx": "{talep} numaralı izin talebinin iptali",
    },
    {
        "intent": "update_leave_request", "tool": "update_izin_talebi", "domain": "ik_islemleri",
        "slots": ["talep", "range"],
        "argmap": {"talep_id": "talep", "yeni_baslangic_tarihi": "range_b", "yeni_bitis_tarihi": "range_e"},
        "argkeys": ["talep_id", "yeni_baslangic_tarihi", "yeni_bitis_tarihi"],
        "user": [
            "{talep} numaralı iznin tarihlerini {range} olarak güncelle.",
            "{talep} talebini {range} tarihlerine çek.",
            "{talep} için izin tarihlerini değiştirelim: {range}.",
            "{talep} numaralı izni {range} olacak şekilde revize et.",
            "Tarih değişikliği: {talep} talebi {range} olsun.",
        ],
        "ctx": "{talep} numaralı iznin {range} olarak güncellenmesi",
    },
    {
        "intent": "update_contact", "tool": "update_employee_contact", "domain": "ik_islemleri",
        "slots": ["emp", "contact_val"],
        "argmap": {"employee_id": "emp_canon", "telefon": "cval", "email": "cval", "adres": "cval"},
        "argkeys": ["employee_id", "@ckind"],  # @ckind -> dinamik alan
        "user": [
            "{emp} {ckind} bilgimi {cval} olarak güncelle.",
            "{emp} için yeni {ckind}: {cval}. Kaydet.",
            "{emp} {ckind} bilgisini {cval} yap.",
            "{emp} kaydındaki {ckind} alanını {cval} ile değiştir.",
            "{ckind} bilgim değişti, {cval} olarak güncelle. Personel: {emp}.",
        ],
        "ctx": "{emp} için {ckind} bilgisinin '{cval}' olarak güncellenmesi",
    },
    {
        "intent": "create_salary_change", "tool": "create_ucret_degisiklik_talebi", "domain": "ik_islemleri",
        "slots": ["emp", "ucret"],
        "argmap": {"employee_id": "emp_canon", "yeni_brut_ucret": "@amount", "gerekce": "gerekce"},
        "argkeys": ["employee_id", "yeni_brut_ucret", "gerekce"],
        "user": [
            "{emp} için brüt ücreti {amount} TL'ye çıkaran bir ücret değişikliği talebi oluştur. Gerekçe: {gerekce}.",
            "{emp} maaşını {amount} brüt yap; gerekçe: {gerekce}.",
            "{emp} için {gerekce} nedeniyle brüt ücret {amount} TL olsun, talebi aç.",
            "{emp} ücret revizyonu: yeni brüt {amount} TL, gerekçe {gerekce}.",
        ],
        "ctx": "{emp} için brüt ücretin {amount} TL olması ({gerekce})",
    },
    {
        "intent": "create_position_change", "tool": "create_pozisyon_degisiklik_talebi", "domain": "ik_islemleri",
        "slots": ["emp", "pozisyon"],
        "argmap": {"employee_id": "emp_canon", "yeni_pozisyon": "poz"},
        "argkeys": ["employee_id", "yeni_pozisyon"],
        "user": [
            "{emp} için pozisyonu {poz} olarak değiştiren talep oluştur.",
            "{emp} unvanını {poz} yap.",
            "{emp} için {poz} pozisyonuna geçiş talebi aç.",
            "{emp} artık {poz} olarak görünsün, gerekli talebi başlat.",
            "{emp} — yeni unvan {poz}. Pozisyon değişikliği talebini başlat.",
            "{emp} için {poz} unvanına terfi talebini oluşturur musun?",
        ],
        "ctx": "{emp} için pozisyonun '{poz}' olarak değiştirilmesi",
    },
    {
        "intent": "update_information", "tool": "update_employee_information", "domain": "ik_islemleri",
        "slots": ["emp", "bilgi_val"],
        "argmap": {"employee_id": "emp_canon"},
        "argkeys": ["employee_id", "@bkind"],  # @bkind -> dinamik özlük alanı
        "user": [
            "{emp} {bkind_disp} bilgimi {bsurf} olarak güncelle.",
            "{emp} için özlük kaydında {bkind_disp} alanını {bsurf} yap.",
            "{bkind_disp} bilgim değişti: {bsurf}. Personel {emp}, güncelle.",
            "{emp} kaydındaki {bkind_disp} bilgisini {bsurf} ile değiştir.",
            "{emp} — {bkind_disp}: {bsurf}. Kaydı güncelle.",
        ],
        "ctx": "{emp} için {bkind_disp} bilgisinin '{bsurf}' olarak güncellenmesi",
    },
]


# ---------------------------------------------------------------------------
# 10. MULTI-TURN (eksik bilgi sonradan verilir) ve MULTI-INTENT
# ---------------------------------------------------------------------------

MT_INFO_SPECS: list[dict] = [
    {"intent": "get_leave_balance", "tool": "get_izin_bakiyesi", "domain": "izin_yonetimi",
     "slots": ["emp"], "argmap": {"employee_id": "emp_canon"},
     "u1": ["İzin bakiyemi göster.", "Kaç gün yıllık iznim kaldı?", "Kalan iznimi öğrenmek istiyorum."],
     "ask": ["Çalışan numaranızı paylaşır mısınız?", "Hangi personel numarası için bakayım?"]},
    {"intent": "get_salary", "tool": "get_maas_bilgisi", "domain": "maas_finans",
     "slots": ["emp"], "argmap": {"employee_id": "emp_canon"},
     "u1": ["Maaşımı göster.", "Bu ayki maaşımı öğrenmek istiyorum.", "Net ücretime bakar mısın?"],
     "ask": ["Çalışan numaranızı belirtir misiniz?", "Hangi personel için maaş bilgisini getireyim?"]},
    {"intent": "get_bonus", "tool": "get_prim_bilgisi", "domain": "maas_finans",
     "slots": ["emp"], "argmap": {"employee_id": "emp_canon"},
     "u1": ["Primimi göster.", "Bu yılki primlerimi listeler misin?"],
     "ask": ["Çalışan numaranızı paylaşır mısınız?", "Hangi personel için prim dökümü istiyorsunuz?"]},
    {"intent": "get_payslip", "tool": "get_bordro", "domain": "maas_finans",
     "slots": ["emp", "donem"], "argmap": {"employee_id": "emp_canon", "donem": "donem_canon"},
     "u1": ["{emp} bordrosunu göster.", "{emp} için maaş pusulası lazım."],
     "ask": ["Hangi döneme ait bordroyu istiyorsunuz? (örn. 2026-07)"],
     "u2mode": "donem"},
    {"intent": "get_timesheet", "tool": "get_puantaj", "domain": "puantaj",
     "slots": ["emp", "month_range"],
     "argmap": {"employee_id": "emp_canon", "baslangic_tarihi": "mrange_b", "bitis_tarihi": "mrange_e"},
     "u1": ["{emp} puantajını göster.", "{emp} çalışma kaydını çıkar."],
     "ask": ["Hangi tarih aralığı için getireyim?"],
     "u2mode": "range"},
]

MT_U2_ID = [
    "{emp}.", "{emp} benim numaram.", "Personel numaram {emp}.", "{emp}, teşekkürler.",
    "Numaram {emp} oluyor.", "{emp} — bu arada acele lazım.",
]

MULTI_INTENT_SPECS: list[dict] = [
    {"intent": "salary_and_leave_balance", "tools": ["get_maas_bilgisi", "get_izin_bakiyesi"],
     "domain": "maas_finans", "slots": ["emp"],
     "calls": [("get_maas_bilgisi", {"employee_id": "emp_canon"}),
               ("get_izin_bakiyesi", {"employee_id": "emp_canon", "izin_tipi": "yillik"})],
     "user": ["{emp} maaşını ve kalan yıllık iznini göster.",
              "{emp} için hem ücret bilgisini hem yıllık izin bakiyesini getir."]},
    {"intent": "balance_and_history", "tools": ["get_izin_bakiyesi", "get_izin_gecmisi"],
     "domain": "izin_yonetimi", "slots": ["emp"],
     "calls": [("get_izin_bakiyesi", {"employee_id": "emp_canon"}),
               ("get_izin_gecmisi", {"employee_id": "emp_canon"})],
     "user": ["{emp} kalan iznini ve geçmiş izin kayıtlarını çıkar.",
              "{emp} için izin bakiyesi ve izin geçmişini birlikte göster."]},
    {"intent": "info_and_manager", "tools": ["get_employee_info", "get_yonetici_bilgisi"],
     "domain": "calisan_bilgileri", "slots": ["emp"],
     "calls": [("get_employee_info", {"employee_id": "emp_canon"}),
               ("get_yonetici_bilgisi", {"employee_id": "emp_canon"})],
     "user": ["{emp} temel bilgilerini ve yöneticisini getir.",
              "{emp} kim, ne iş yapıyor ve kime bağlı — ikisini de söyle."]},
    {"intent": "payslip_and_overtime", "tools": ["get_bordro", "get_mesai_bilgisi"],
     "domain": "maas_finans", "slots": ["emp", "donem"],
     "calls": [("get_bordro", {"employee_id": "emp_canon", "donem": "donem_canon"}),
               ("get_mesai_bilgisi", {"employee_id": "emp_canon", "donem": "donem_canon"})],
     "user": ["{emp} için {donem} bordrosunu ve fazla mesaisini getir.",
              "{emp} {donem}: bordro ve mesai dökümünü birlikte ver."]},
]

# --- Çok-adımlı zincir (§25 "çok zor"): parametre topla -> onay iste -> uygula ---
# 6 tur. Kullanıcı bileşik bir yazma isteği verir (bir parametre eksik). Model:
#   A1: eksik parametreyi ister (uygunluk/onay adımını sonraya bırakır)  -> request_for_info
#   A2: parametre gelince YİNE de yazma için onay ister                   -> request_for_info
#   A3: onay gelince tool_call                                            -> tool_call
# Böylece "önce topla, sonra onayla, sonra uygula" sıralaması öğretilir; hiçbir
# değer uydurulmaz (eksik tarih/kimlik asla varsayılmaz).
MULTI_STEP_SPECS: list[dict] = [
    {
        "intent": "create_leave_request", "tool": "create_izin_talebi", "domain": "ik_islemleri",
        "slots": ["emp", "izin_tipi", "range"], "missing": "range",
        "argmap": {"employee_id": "emp_canon", "izin_tipi": "tip_canon",
                   "baslangic_tarihi": "range_b", "bitis_tarihi": "range_e"},
        "argkeys": ["employee_id", "izin_tipi", "baslangic_tarihi", "bitis_tarihi"],
        "u1": [
            "{emp} için {tip_disp} kullanmak istiyorum, uygunsam talebi de oluştur.",
            "{emp} — {tip} izin planlıyorum; müsaitsem talebi aç.",
            "{emp} için {tip_disp} almak istiyorum, sonra da kaydını gir.",
        ],
        "a1": [
            "Talebi hazırlayabilmem için hangi tarihler arası olduğunu söyler misiniz? Sonra onayınızı alıp oluştururum.",
            "Hangi tarihlerde? Başlangıç ve bitişi verirseniz talebi hazırlar, onayınızla kaydederim.",
        ],
        "u2": ["{range}.", "{range} istiyorum.", "Tarihler: {range}.", "{range} için olsun."],
        "ctx": "{emp} için {range} tarihlerinde {tip_disp} talebi",
    },
    {
        "intent": "create_leave_request", "tool": "create_izin_talebi", "domain": "ik_islemleri",
        "slots": ["emp", "izin_tipi", "range"], "missing": "emp",
        "argmap": {"employee_id": "emp_canon", "izin_tipi": "tip_canon",
                   "baslangic_tarihi": "range_b", "bitis_tarihi": "range_e"},
        "argkeys": ["employee_id", "izin_tipi", "baslangic_tarihi", "bitis_tarihi"],
        "u1": [
            "{range} için {tip_disp} kullanmak istiyorum, uygunsam talebi oluştur.",
            "{tip} izne çıkacağım: {range}. Uygunsam kaydını da aç.",
        ],
        "a1": [
            "Talebi kimin adına oluşturayım? Çalışan numaranızı belirtir misiniz?",
            "Hangi personel için? Çalışan numaranıza ihtiyacım var; sonra onayla kaydederim.",
        ],
        "u2": ["{emp}.", "Personel numaram {emp}.", "{emp} benim numaram.", "Numaram {emp}."],
        "ctx": "{emp} için {range} tarihlerinde {tip_disp} talebi",
    },
    {
        "intent": "update_leave_request", "tool": "update_izin_talebi", "domain": "ik_islemleri",
        "slots": ["talep", "range"], "missing": "range",
        "argmap": {"talep_id": "talep", "yeni_baslangic_tarihi": "range_b", "yeni_bitis_tarihi": "range_e"},
        "argkeys": ["talep_id", "yeni_baslangic_tarihi", "yeni_bitis_tarihi"],
        "u1": [
            "{talep} numaralı iznimin tarihlerini kaydırmak istiyorum, uygunsa güncelle.",
            "{talep} talebini başka tarihlere çekelim, sonra da kaydet.",
        ],
        "a1": [
            "Yeni tarihler ne olsun? Başlangıç ve bitişi verirseniz güncellemeyi hazırlayıp onayınızı alırım.",
            "Hangi tarihlere çekelim? Yeni başlangıç ve bitişi belirtin, sonra onayla uygularım.",
        ],
        "u2": ["{range}.", "Yeni tarihler: {range}.", "{range} olarak."],
        "ctx": "{talep} numaralı iznin {range} olarak güncellenmesi",
    },
]

# --- Çok turlu DIRECT: tanım sorusu + takip sorusu (ikisi de tool'suz) ---
MT_DIRECT_SPECS: list[dict] = [
    {"intent": "define_annual_leave", "domain": "izin_yonetimi",
     "u1": ["Yıllık izin nedir?", "Yıllık ücretli izni kısaca açıklar mısın?"],
     "a1": ["Yıllık izin, çalıştığınız her yıl karşılığında kazandığınız, ücretiniz ödenmeye devam ederken kullanabileceğiniz dinlenme iznidir. Süresi kıdeme göre değişir ve kullandıkça bakiyenizden düşer."],
     "u2": ["Kullanmazsam yanar mı?", "Peki sonraki yıla devreder mi?"],
     "a2": ["Genelde yanmaz; çoğu şirkette bir sonraki döneme devreder, ancak devredilebilecek gün sayısına veya kullanım süresine tavan konabilir. Kesin kural şirketinizin İK politikasına göre değişir."]},
    {"intent": "gross_net_salary", "domain": "maas_finans",
     "u1": ["Brüt maaş ile net maaş arasındaki fark nedir?", "Brüt ve net neden farklı çıkıyor?"],
     "a1": ["Brüt, yasal kesintiler öncesi tutardır. Net ise brütten SGK primi, işsizlik sigortası, gelir vergisi ve damga vergisi düşüldükten sonra elinize geçen tutardır."],
     "u2": ["Yıl içinde netim neden düşüyor peki?", "Aynı brütte net neden değişiyor?"],
     "a2": ["Gelir vergisi kümülatif matrah üzerinden artan oranlıdır. Yıl ilerledikçe biriken kazancınız üst dilime girince kesinti oranı artar ve net maaşınız bir miktar azalır."]},
    {"intent": "define_severance", "domain": "maas_finans",
     "u1": ["Kıdem tazminatı nedir?", "Kıdem tazminatını kısaca anlatır mısın?"],
     "a1": ["En az bir yıllık kıdemi olan çalışana, kanunda sayılan fesih hallerinde her tam yıl için son brüt ücreti tutarında ödenen tazminattır."],
     "u2": ["İstifa edersem de alır mıyım?", "Her ayrılışta ödenir mi?"],
     "a2": ["Hayır. İstifa gibi bazı durumlarda kıdem tazminatı hakkı doğmaz; hak, feshin nedenine bağlıdır. Emeklilik, askerlik, evlilik (kadın çalışan için) gibi istisnalar dışında kendi isteğiyle ayrılan çalışan genelde hak etmez."]},
    {"intent": "leave_request_process", "domain": "ik_islemleri",
     "u1": ["İzin talebini nasıl oluştururum?", "İzin başvurusu nasıl yapılıyor?"],
     "a1": ["İzin türünü seçer, başlangıç ve bitiş tarihlerini girer, gerekiyorsa kısa bir açıklama ekler ve talebi gönderirsiniz. Talep yöneticinize onaya düşer."],
     "u2": ["Onaylanması ne kadar sürer?", "Onaydan sonra ne oluyor?"],
     "a2": ["Süre yöneticinize bağlıdır; genelde birkaç iş günü içinde sonuçlanır. Onaylandığında izniniz kesinleşir ve izin bakiyeniz otomatik güncellenir."]},
    {"intent": "define_bordro", "domain": "maas_finans",
     "u1": ["Bordro nedir?", "Maaş pusulası neyi gösterir?"],
     "a1": ["Bordro, bir aya ait ücret hesabınızın resmi dökümüdür: brüt ücret, SGK ve işsizlik primleri, gelir ve damga vergisi, varsa prim/kesinti kalemleri ve net ödenen tutar satır satır gösterilir."],
     "u2": ["Kümülatif matrah ne demek orada?", "Çalışılan gün neden yazıyor?"],
     "a2": ["Kümülatif matrah, yıl başından itibaren biriken vergiye esas kazancınızdır; arttıkça gelir vergisi oranınız yükselebilir. Çalışılan gün ise o ay fiilen çalıştığınız gün sayısıdır ve ücret ile SGK gününü belirler."]},
    {"intent": "difference_leave_types", "domain": "izin_yonetimi",
     "u1": ["Yıllık izin ile mazeret izni arasındaki fark nedir?"],
     "a1": ["Yıllık izin planlı dinlenme içindir ve bakiyenizden düşer. Mazeret izni evlilik, doğum, vefat gibi belirli olaylara bağlıdır, olayına özgüdür ve yıllık bakiyeyi etkilemez."],
     "u2": ["Hastalık izni bunların neresinde?", "Rapor da mazeret sayılır mı?"],
     "a2": ["Hastalık izni ayrı bir kalemdir: doktor raporuna dayanır, yıllık bakiyeden düşmez ve ödemesi ilgili mevzuata göre yapılır. Rapor mazeret izni değildir; kendi kategorisindedir."]},
    {"intent": "overtime_calculation", "domain": "puantaj",
     "u1": ["Fazla mesai ücreti nasıl hesaplanıyor?", "Fazla çalışma nasıl ödeniyor?"],
     "a1": ["Haftalık 45 saati aşan çalışma fazla mesaidir ve saat ücretinin %50 zamlısıyla ödenir. Hafta tatili veya genel tatilde çalışma için oranlar daha yüksektir."],
     "u2": ["İzne çevirebiliyor muyum onun yerine?", "Ücret yerine izin alınır mı?"],
     "a2": ["Evet, fazla çalışma karşılığı ücret yerine serbest zaman (izin) tercih edilebilir; her fazla mesai saati için 1,5 saat izin hak edilir ve bu izin altı ay içinde kullandırılır."]},
    {"intent": "leave_carryover", "domain": "izin_yonetimi",
     "u1": ["Kullanmadığım yıllık izin sonraki yıla devreder mi?"],
     "a1": ["Genel kural olarak kullanılmayan yıllık izin yanmaz ve sonraki döneme devredilebilir. Ancak birçok şirket devredilebilecek gün sayısına veya kullanım süresine bir tavan koyar."],
     "u2": ["Devreden izinin bir son kullanma tarihi olur mu?"],
     "a2": ["Şirket politikasına göre olabilir: 'devreden gün şu tarihe kadar kullanılmalı' gibi bir kural konabilir. Kendi durumunuz için İK politikanıza bakmanız gerekir."]},
]

# --- Çok turlu CANNOT_ANSWER: kapsam dışı istek + kullanıcı ısrarı ---
MT_CANNOT_PUSH = [
    "Yine de bir fikir ver.", "Olsun, yaklaşık söyle yeter.", "Emin olmasan da söyle.",
    "Kabaca bir tahmin yeterli.", "Hadi ama, yardımcı ol.", "Peki bu konuda ne yapabilirim?",
    "Cidden hiç mi bilgi veremiyorsun?", "Başka bir yolu yok mu?",
]
MT_CANNOT_HOLD = [
    "Maalesef bu konuda yardımcı olamıyorum; doğrulayamadığım bir bilgiyi tahminle de olsa paylaşmam.",
    "Anlıyorum ama yine de yapamam. Yanlış yönlendirmemek için bu isteği geri çevirmem gerekiyor.",
    "Ne yazık ki bu, mevcut araçların dışında kalıyor. İsterseniz yapabildiğim İK konularına geçelim.",
    "Bu talebi karşılayamıyorum; ısrar etseniz de veremeyeceğim bir bilgi bu. İzin, maaş, bordro veya puantaj konularında destek olabilirim.",
    "Bu konuda bir adım atamıyorum. Dilerseniz bunu İK ekibinize iletmenizi öneririm.",
]


DOMAIN_TOOLS = {
    "izin_yonetimi": ["get_izin_bakiyesi", "get_izin_gecmisi", "get_izin_talebi_durumu", "create_izin_talebi", "cancel_izin_talebi"],
    "maas_finans": ["get_maas_bilgisi", "get_bordro", "get_prim_bilgisi", "get_yan_haklar", "create_ucret_degisiklik_talebi"],
    "puantaj": ["get_puantaj", "get_mesai_bilgisi", "get_employee_status"],
    "organizasyon": ["get_departman_bilgisi", "get_calisan_listesi", "get_yonetici_bilgisi", "create_pozisyon_degisiklik_talebi"],
    "calisan_bilgileri": ["get_employee_info", "get_employee_status", "update_employee_contact", "update_employee_information", "get_yonetici_bilgisi"],
    "ik_islemleri": ["create_izin_talebi", "cancel_izin_talebi", "update_izin_talebi", "check_employee_access", "update_employee_contact", "update_employee_information"],
    "meta": [],
    "kapsanmayan": [],
}

NO_FRAME_INTENTS = {"greeting", "thanks", "farewell"}


# ---------------------------------------------------------------------------
# 11. ÜRETİM MOTORU
# ---------------------------------------------------------------------------

ORDER = ["kolay", "orta", "zor", "cok_zor"]


def bump_difficulty(base: str, steps: int) -> str:
    i = ORDER.index(base) + steps
    return ORDER[max(0, min(i, len(ORDER) - 1))]


def detect_register(text: str) -> str:
    t = tr_fold(text)
    if any(w in t for w in [" abi", "abi ", " ya ", "yaa", "valla", " hocam", "reis"]):
        return "konusma_dili"
    if any(w in text for w in ["Sayın", "rica ederim", "arz ederim", "müsaadenizle", "bilgilerinize"]):
        return "resmi"
    if re.search(r"(?<![a-zçğıöşü])(kac|gun|misin|yillik|maasim|iznim|ogrenmek|kalmis|gecmis|dokumu|sirket|calisan)(?![a-z])", text) \
            and text[0].islower():
        return "yazim_hatali"
    words = text.split()
    if len(words) <= 4:
        return "kisa"
    if len(text) > 135:
        return "uzun"
    return "gundelik"


def tools_field(rng, targets, domain=None, k_min=4, k_max=8):
    seed_pool = list(targets)
    if domain and DOMAIN_TOOLS.get(domain):
        seed_pool += [t for t in DOMAIN_TOOLS[domain] if t not in seed_pool]
    return build_tools_list(rng, seed_pool, k_min=k_min, k_max=k_max) if targets or seed_pool \
        else build_tools_list(rng, [], k_min=k_min, k_max=k_max)


def fmt(text: str, slots: dict) -> str:
    try:
        return text.format(**slots)
    except KeyError as e:  # pragma: no cover - şablon hatası yakalama
        raise KeyError(f"'{text}' şablonunda eksik slot: {e}")


def build_args(argkeys, fixed, argmap, slots):
    args = dict(fixed or {})
    for k in argkeys:
        if k == "@ckind":
            args[slots["ckind"]] = slots["cval"]
            continue
        if k == "@bkind":
            args[slots["bkind"]] = slots["bval"]
            continue
        if k in args:
            continue
        src = argmap.get(k)
        if src is None:
            continue
        if src == "@amount":
            args[k] = slots["amount"]
        else:
            args[k] = slots[src]
    return args


CONFIRM_FRAMES = [
    "{ctx} işlemini gerçekleştireceğim. {q}",
    "Özetliyorum: {ctx}. {q}",
    "{ctx} kaydını oluşturmak üzereyim. {q}",
    "Şunu yapacağım: {ctx}. {q}",
]


def confirm_ask_text(rng, ctx: str) -> str:
    frame = rng.choice(CONFIRM_FRAMES)
    return frame.format(ctx=ctx[0].upper() + ctx[1:], q=rng.choice(CONFIRM_ASKS))


class Gen:
    def __init__(self, rng, today: date):
        self.rng = rng
        self.today = today
        self.sfuncs = make_slot_funcs(rng)
        self.seen: set[str] = set()
        self.rows: list[dict] = []
        self.skipped = 0

    # -- yardımcılar --------------------------------------------------------
    def resolve(self, slot_names):
        s = {}
        for name in slot_names:
            s.update(self.sfuncs[name]())
        return s

    def sig(self, user_texts):
        return norm_sig(" || ".join(user_texts))

    def add(self, *, decision, intent, domain, difficulty, messages, tools,
            target_tool=None, target_tools=None, required=None, missing=None,
            is_write=False, confirmation_required=False, multi_turn=False, register=None,
            chain=False):
        users = [m["content"] for m in messages if m["role"] == "user"]
        sg = self.sig(users)
        if sg in self.seen:
            self.skipped += 1
            return False
        self.seen.add(sg)
        first_user = users[0]
        reg = register or detect_register(first_user)
        long_text = len(first_user) > 135 or reg == "uzun"
        diff = difficulty
        if multi_turn:
            diff = bump_difficulty(diff, 1)
        if long_text:
            diff = bump_difficulty(diff, 1)
        self.rows.append({
            "record": {"tools": tools, "messages": messages},
            "meta": {
                "decision": decision,
                "intent": intent,
                "target_tool": target_tool,
                "target_tools": target_tools or ([target_tool] if target_tool else []),
                "required_parameters": required or [],
                "missing_parameters": missing or [],
                "is_write": is_write,
                "confirmation_required": confirmation_required,
                "domain": domain,
                "difficulty": diff,
                "register": reg,
                "multi_turn": multi_turn,
                "chain": chain,
                "turns": len(messages),
            },
        })
        return True

    def style(self, intent, text):
        return style_user_text(self.rng, text, intent)

    # -- sınıf üreticileri ------------------------------------------------
    def gen_direct(self, target):
        specs = list(DIRECT_INTENTS)
        rounds = 0
        while self._count("direct") < target and rounds < 60:
            rounds += 1
            self.rng.shuffle(specs)
            for spec in specs:
                if self._count("direct") >= target:
                    break
                q, reg = self.style(spec["intent"], self.rng.choice(spec["q"]))
                a = self.rng.choice(spec["a"])
                tools = tools_field(self.rng, [], domain=spec["domain"])
                self.add(
                    decision="direct", intent=spec["intent"], domain=spec["domain"],
                    difficulty=spec["difficulty"], register=reg,
                    messages=[{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                    tools=tools,
                )

    def gen_cannot(self, target):
        specs = list(CANNOT_INTENTS)
        rounds = 0
        while self._count("cannot_answer") < target and rounds < 80:
            rounds += 1
            self.rng.shuffle(specs)
            for spec in specs:
                if self._count("cannot_answer") >= target:
                    break
                q, reg = self.style(spec["intent"], self.rng.choice(spec["q"]))
                a = refusal_text(self.rng, spec["pool"])
                tools = tools_field(self.rng, [], domain=spec["domain"])
                _hard = {"historical_bulk_performance_ranking", "predict_exact_future_leave",
                         "external_salary_benchmark", "compare_others_salaries",
                         "bulk_lateness_ranking", "approve_on_behalf_of_manager",
                         "bulk_process_all_requests", "permanently_delete_record",
                         "reset_manager_credentials"}
                diff = "kolay" if spec["pool"] == "plain" and spec["domain"] == "kapsanmayan" \
                    else ("zor" if spec["intent"] in _hard else "orta")
                self.add(
                    decision="cannot_answer", intent=spec["intent"], domain=spec["domain"],
                    difficulty=spec.get("difficulty", diff), register=reg,
                    messages=[{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                    tools=tools,
                )

    def gen_read(self, target):
        specs = list(READ_SPECS)
        rounds = 0
        while self._count_read() < target and rounds < 120:
            rounds += 1
            self.rng.shuffle(specs)
            for spec in specs:
                if self._count_read() >= target:
                    break
                text, argkeys, fixed = self.rng.choice(spec["templates"])
                slots = self.resolve(spec["slots"])
                user, reg = self.style(spec["intent"], fmt(text, slots))
                args = build_args(argkeys, fixed, spec["argmap"], slots)
                call = tool_call_block(spec["tool"], args)
                tools = tools_field(self.rng, [spec["tool"]], domain=spec["domain"])
                self.add(
                    decision="tool_call", intent=spec["intent"], domain=spec["domain"],
                    difficulty=spec["difficulty"], register=reg,
                    messages=[{"role": "user", "content": user}, {"role": "assistant", "content": call}],
                    tools=tools, target_tool=spec["tool"],
                    required=TOOLS[spec["tool"]]["parameters"]["required"],
                )

    def gen_missing(self, target):
        specs = list(MISSING_PARAM_SPECS)
        rounds = 0
        while self._count_missing() < target and rounds < 120:
            rounds += 1
            self.rng.shuffle(specs)
            for spec in specs:
                if self._count_missing() >= target:
                    break
                slots = self.resolve(spec.get("slots", []))
                user, reg = self.style(spec["intent"], fmt(self.rng.choice(spec["q"]), slots))
                ask = fmt(self.rng.choice(spec["ask"]), slots)
                tools = tools_field(self.rng, [spec["tool"]], domain=spec["domain"])
                self.add(
                    decision="request_for_info", intent=spec["intent"], domain=spec["domain"],
                    difficulty="orta", register=reg,
                    messages=[{"role": "user", "content": user}, {"role": "assistant", "content": ask}],
                    tools=tools, target_tool=spec["tool"],
                    required=TOOLS[spec["tool"]]["parameters"]["required"],
                    missing=spec["missing"], is_write=spec.get("is_write", False),
                )

    def gen_confirm_ask(self, target):
        specs = list(WRITE_SPECS)
        rounds = 0
        while self._count_confirm_ask() < target and rounds < 120:
            rounds += 1
            self.rng.shuffle(specs)
            for spec in specs:
                if self._count_confirm_ask() >= target:
                    break
                slots = self.resolve(spec["slots"])
                user, reg = self.style(spec["intent"], fmt(self.rng.choice(spec["user"]), slots))
                ctx = fmt(spec["ctx"], slots)
                ask = confirm_ask_text(self.rng, ctx)
                tools = tools_field(self.rng, [spec["tool"]], domain=spec["domain"])
                self.add(
                    decision="request_for_info", intent=spec["intent"], domain=spec["domain"],
                    difficulty="zor", register=reg,
                    messages=[{"role": "user", "content": user}, {"role": "assistant", "content": ask}],
                    tools=tools, target_tool=spec["tool"],
                    required=TOOLS[spec["tool"]]["parameters"]["required"],
                    is_write=True, confirmation_required=True,
                )

    def gen_confirmed(self, target):
        specs = list(WRITE_SPECS)
        rounds = 0
        while self._count_confirmed() < target and rounds < 120:
            rounds += 1
            self.rng.shuffle(specs)
            for spec in specs:
                if self._count_confirmed() >= target:
                    break
                slots = self.resolve(spec["slots"])
                user, reg = self.style(spec["intent"], fmt(self.rng.choice(spec["user"]), slots))
                ctx = fmt(spec["ctx"], slots)
                ask = confirm_ask_text(self.rng, ctx)
                ack = self.rng.choice(ACK_WORDS)
                args = build_args(spec["argkeys"], {}, spec["argmap"], slots)
                call = tool_call_block(spec["tool"], args)
                tools = tools_field(self.rng, [spec["tool"]], domain=spec["domain"])
                self.add(
                    decision="tool_call", intent=spec["intent"], domain=spec["domain"],
                    difficulty="zor", register=reg,
                    messages=[
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": ask},
                        {"role": "user", "content": ack},
                        {"role": "assistant", "content": call},
                    ],
                    tools=tools, target_tool=spec["tool"],
                    required=TOOLS[spec["tool"]]["parameters"]["required"],
                    is_write=True, confirmation_required=True, multi_turn=True,
                )

    def gen_mt_info(self, target):
        specs = list(MT_INFO_SPECS)
        rounds = 0
        while self._count_mt_info() < target and rounds < 120:
            rounds += 1
            self.rng.shuffle(specs)
            for spec in specs:
                if self._count_mt_info() >= target:
                    break
                slots = self.resolve(spec["slots"])
                mode = spec.get("u2mode", "id")
                u1, reg = self.style(spec["intent"], fmt(self.rng.choice(spec["u1"]), slots))
                ask = fmt(self.rng.choice(spec["ask"]), slots)
                if mode == "id":
                    u2 = fmt(self.rng.choice(MT_U2_ID), slots)
                elif mode == "donem":
                    u2 = fmt(self.rng.choice(["{donem}.", "{donem} dönemi.", "{donem} olsun."]), slots)
                else:  # range
                    u2 = fmt(self.rng.choice(["{mrange}.", "{mrange} arası.", "{mrange} yeterli."]), slots)
                args = build_args(list(spec["argmap"].keys()), {}, spec["argmap"], slots)
                call = tool_call_block(spec["tool"], args)
                tools = tools_field(self.rng, [spec["tool"]], domain=spec["domain"])
                self.add(
                    decision="tool_call", intent=spec["intent"], domain=spec["domain"],
                    difficulty="orta", register=reg,
                    messages=[
                        {"role": "user", "content": u1},
                        {"role": "assistant", "content": ask},
                        {"role": "user", "content": u2},
                        {"role": "assistant", "content": call},
                    ],
                    tools=tools, target_tool=spec["tool"],
                    required=TOOLS[spec["tool"]]["parameters"]["required"], multi_turn=True,
                )

    def gen_multi_intent(self, target):
        specs = list(MULTI_INTENT_SPECS)
        rounds = 0
        while self._count_multi_intent() < target and rounds < 120:
            rounds += 1
            self.rng.shuffle(specs)
            for spec in specs:
                if self._count_multi_intent() >= target:
                    break
                slots = self.resolve(spec["slots"])
                user, reg = self.style(spec["intent"], fmt(self.rng.choice(spec["user"]), slots))
                calls = []
                for tname, amap in spec["calls"]:
                    a = {k: (slots[v] if v in slots else v) for k, v in amap.items()}
                    calls.append((tname, a))
                content = tool_call_blocks(calls)
                tools = tools_field(self.rng, spec["tools"], domain=spec["domain"], k_min=5, k_max=9)
                self.add(
                    decision="tool_call", intent=spec["intent"], domain=spec["domain"],
                    difficulty="zor", register=reg,
                    messages=[{"role": "user", "content": user}, {"role": "assistant", "content": content}],
                    tools=tools, target_tools=spec["tools"],
                )

    def gen_multi_step(self, target):
        """6 tur: eksik parametre iste -> gelince onay iste -> onaylanınca tool_call."""
        specs = list(MULTI_STEP_SPECS)
        rounds = 0
        while self._count_multi_step() < target and rounds < 160:
            rounds += 1
            self.rng.shuffle(specs)
            for spec in specs:
                if self._count_multi_step() >= target:
                    break
                slots = self.resolve(spec["slots"])
                u1, reg = self.style(spec["intent"], fmt(self.rng.choice(spec["u1"]), slots))
                a1 = fmt(self.rng.choice(spec["a1"]), slots)
                u2 = fmt(self.rng.choice(spec["u2"]), slots)
                ctx = fmt(spec["ctx"], slots)
                a2 = confirm_ask_text(self.rng, ctx)
                u3 = self.rng.choice(ACK_WORDS)
                args = build_args(spec["argkeys"], {}, spec["argmap"], slots)
                a3 = tool_call_block(spec["tool"], args)
                tools = tools_field(self.rng, [spec["tool"]], domain=spec["domain"], k_min=5, k_max=9)
                self.add(
                    decision="tool_call", intent=spec["intent"], domain=spec["domain"],
                    difficulty="zor", register=reg,
                    messages=[
                        {"role": "user", "content": u1},
                        {"role": "assistant", "content": a1},
                        {"role": "user", "content": u2},
                        {"role": "assistant", "content": a2},
                        {"role": "user", "content": u3},
                        {"role": "assistant", "content": a3},
                    ],
                    tools=tools, target_tool=spec["tool"],
                    required=TOOLS[spec["tool"]]["parameters"]["required"],
                    missing=[spec["missing"]] if spec.get("missing") else None,
                    is_write=True, confirmation_required=True, multi_turn=True, chain=True,
                )

    def gen_mt_direct(self, target):
        """4 tur: tanım sorusu -> yanıt -> takip sorusu -> yanıt. Tool yok."""
        specs = list(MT_DIRECT_SPECS)
        rounds = 0
        while self._count_mt_direct() < target and rounds < 120:
            rounds += 1
            self.rng.shuffle(specs)
            for spec in specs:
                if self._count_mt_direct() >= target:
                    break
                u1, reg = self.style(spec["intent"], self.rng.choice(spec["u1"]))
                a1 = self.rng.choice(spec["a1"])
                u2 = self.rng.choice(spec["u2"])
                a2 = self.rng.choice(spec["a2"])
                tools = tools_field(self.rng, [], domain=spec["domain"])
                self.add(
                    decision="direct", intent=spec["intent"], domain=spec["domain"],
                    difficulty="orta", register=reg,
                    messages=[
                        {"role": "user", "content": u1},
                        {"role": "assistant", "content": a1},
                        {"role": "user", "content": u2},
                        {"role": "assistant", "content": a2},
                    ],
                    tools=tools, multi_turn=True,
                )

    def gen_mt_cannot(self, target):
        """4 tur: kapsam dışı istek -> kibar ret -> kullanıcı ısrarı -> kararlı ret. Tool yok."""
        specs = [s for s in CANNOT_INTENTS if s["pool"] in ("future", "privacy", "plain", "unsupported")]
        rounds = 0
        while self._count_mt_cannot() < target and rounds < 120:
            rounds += 1
            self.rng.shuffle(specs)
            for spec in specs:
                if self._count_mt_cannot() >= target:
                    break
                u1, reg = self.style(spec["intent"], self.rng.choice(spec["q"]))
                a1 = refusal_text(self.rng, spec["pool"])
                u2 = self.rng.choice(MT_CANNOT_PUSH)
                a2 = self.rng.choice(MT_CANNOT_HOLD)
                tools = tools_field(self.rng, [], domain=spec["domain"])
                self.add(
                    decision="cannot_answer", intent=spec["intent"], domain=spec["domain"],
                    difficulty="orta", register=reg,
                    messages=[
                        {"role": "user", "content": u1},
                        {"role": "assistant", "content": a1},
                        {"role": "user", "content": u2},
                        {"role": "assistant", "content": a2},
                    ],
                    tools=tools, multi_turn=True,
                )

    # -- sayaçlar ---------------------------------------------------------
    def _count(self, decision):
        return sum(1 for r in self.rows if r["meta"]["decision"] == decision)

    def _tag_count(self, pred):
        return sum(1 for r in self.rows if pred(r["meta"]))

    def _count_read(self):
        return self._tag_count(lambda m: m["decision"] == "tool_call" and not m["multi_turn"] and len(m["target_tools"]) == 1 and not m["is_write"])

    def _count_missing(self):
        return self._tag_count(lambda m: m["decision"] == "request_for_info" and not m["confirmation_required"])

    def _count_confirm_ask(self):
        return self._tag_count(lambda m: m["decision"] == "request_for_info" and m["confirmation_required"])

    def _count_confirmed(self):
        return self._tag_count(lambda m: m["decision"] == "tool_call" and m["is_write"]
                               and m["multi_turn"] and not m.get("chain"))

    def _count_mt_info(self):
        return self._tag_count(lambda m: m["decision"] == "tool_call" and not m["is_write"]
                               and m["multi_turn"] and not m.get("chain"))

    def _count_multi_intent(self):
        return self._tag_count(lambda m: m["decision"] == "tool_call" and len(m["target_tools"]) > 1)

    def _count_multi_step(self):
        return self._tag_count(lambda m: m.get("chain"))

    def _count_mt_direct(self):
        return self._tag_count(lambda m: m["decision"] == "direct" and m["multi_turn"])

    def _count_mt_cannot(self):
        return self._tag_count(lambda m: m["decision"] == "cannot_answer" and m["multi_turn"])


# ---------------------------------------------------------------------------
# 12. ÇALIŞTIRMA / YAZMA
# ---------------------------------------------------------------------------

def stratified_split(rows, rng, val_ratio):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["meta"]["decision"], r["meta"]["intent"])].append(r)
    train, val = [], []
    for _, items in sorted(groups.items()):
        rng.shuffle(items)
        k = round(len(items) * val_ratio)
        if len(items) < 8:
            k = 0
        val.extend(items[:k])
        train.extend(items[k:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_jsonl(path: Path, objs):
    # newline="\n": platformdan bağımsız LF satır sonu (JSONL sözleşmesi +
    # işletim sistemleri arası byte-aynı çıktı).
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def build_report(all_rows, train, val, args) -> str:
    n = len(all_rows)
    dec = Counter(r["meta"]["decision"] for r in all_rows)
    dom = Counter(r["meta"]["domain"] for r in all_rows)
    dif = Counter(r["meta"]["difficulty"] for r in all_rows)
    reg = Counter(r["meta"]["register"] for r in all_rows)
    mt = sum(1 for r in all_rows if r["meta"]["multi_turn"])
    mi = sum(1 for r in all_rows if len(r["meta"]["target_tools"]) > 1)
    writes = sum(1 for r in all_rows if r["meta"]["is_write"])
    chains = sum(1 for r in all_rows if r["meta"].get("chain"))
    turn_hist = Counter(r["meta"]["turns"] for r in all_rows)
    tool_hist = Counter()
    for r in all_rows:
        for m in r["record"]["messages"]:
            if m["role"] == "assistant":
                for mm in re.finditer(r'"name"\s*:\s*"([^"]+)"', m["content"]):
                    tool_hist[mm.group(1)] += 1
    sigs = {norm_sig(" || ".join(m["content"] for m in r["record"]["messages"] if m["role"] == "user")) for r in all_rows}

    lines = []
    lines.append(f"# Büyük İK tool-calling dataset — üretim raporu\n")
    lines.append(f"- Üretim tarihi bağlamı (today): `{args.today}`")
    lines.append(f"- Seed: `{args.seed}`  |  Hedef N: `{args.n}`  |  Üretilen: **{n}**")
    lines.append(f"- Train / Val: **{len(train)} / {len(val)}**  (val oranı ~{args.val_ratio})")
    lines.append(f"- Benzersiz kullanıcı-mesajı imzası: **{len(sigs)} / {n}**  (%{100*len(sigs)/max(n,1):.1f})")
    lines.append(f"- Çok turlu örnek: **{mt}**  (bunun {chains}'i 6-turlu 'topla→onay→uygula' zinciri)  "
                 f"|  Çoklu-tool örnek: **{mi}**  |  WRITE örneği: **{writes}**")
    lines.append(f"- Tur dağılımı: " + ", ".join(f"{k} tur: {v}" for k, v in sorted(turn_hist.items())) + "\n")

    lines.append("## Karar dağılımı (hedef vs gerçekleşen)\n")
    lines.append("| decision | hedef | gerçekleşen | oran |")
    lines.append("|---|---|---|---|")
    for k in ["tool_call", "direct", "request_for_info", "cannot_answer"]:
        lines.append(f"| {k} | %{TARGET_MIX[k]*100:.0f} | {dec[k]} | %{100*dec[k]/max(n,1):.1f} |")

    def block(title, counter):
        out = [f"\n## {title}\n", "| değer | adet | oran |", "|---|---|---|"]
        for k, v in counter.most_common():
            out.append(f"| {k} | {v} | %{100*v/max(n,1):.1f} |")
        return "\n".join(out)

    lines.append(block("Domain dağılımı", dom))
    lines.append(block("Zorluk dağılımı", dif))
    lines.append(block("Register (dil kaydı) dağılımı", reg))
    lines.append(block("Tool çağrı histogramı (assistant çıktısında)", tool_hist))
    lines.append("\n## Notlar\n")
    lines.append("- Eğitim dosyası (`*_train.jsonl`) yalnızca `tools` + `messages` içerir.")
    lines.append("- `*_train.meta.jsonl` satırları eğitim dosyasıyla AYNI SIRADADIR; QC ve değerlendirme içindir.")
    lines.append("- Tüm çalışan / ID / maaş / tarih bilgileri sentetiktir.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Büyük İK tool-calling dataset üreticisi")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--today", default=DEFAULT_TODAY)
    ap.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--dry-run", action="store_true", help="dosya yazma, yalnızca rapor")
    ap.add_argument("--sample", type=int, default=0, help="N örnek yazdır")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    today = date.fromisoformat(args.today)

    n = args.n
    n_tool = round(n * TARGET_MIX["tool_call"])
    n_direct = round(n * TARGET_MIX["direct"])
    n_reqinfo = round(n * TARGET_MIX["request_for_info"])
    n_cannot = n - n_tool - n_direct - n_reqinfo

    # tool_call alt kırılımı
    n_multi_intent = max(8, round(n_tool * 0.07))
    n_mt_info = max(8, round(n_tool * 0.09))
    n_confirmed = max(10, round(n_tool * 0.16))
    n_multi_step = max(8, round(n_tool * 0.11))   # gerçek çok-adımlı zincir (§25)
    n_read = n_tool - n_multi_intent - n_mt_info - n_confirmed - n_multi_step
    # request_for_info alt kırılımı
    n_confirm_ask = max(10, round(n_reqinfo * 0.33))
    n_missing = n_reqinfo - n_confirm_ask
    # direct / cannot_answer alt kırılımı (çok turlu pay)
    n_mt_direct = max(6, round(n_direct * 0.12))
    n_mt_cannot = max(6, round(n_cannot * 0.12))

    g = Gen(rng, today)
    g.gen_read(n_read)
    g.gen_confirmed(n_confirmed)
    g.gen_mt_info(n_mt_info)
    g.gen_multi_intent(n_multi_intent)
    g.gen_multi_step(n_multi_step)
    g.gen_missing(n_missing)
    g.gen_confirm_ask(n_confirm_ask)
    g.gen_mt_direct(n_mt_direct)
    g.gen_direct(n_direct)
    g.gen_mt_cannot(n_mt_cannot)
    g.gen_cannot(n_cannot)

    all_rows = g.rows
    rng.shuffle(all_rows)
    train, val = stratified_split(all_rows, rng, args.val_ratio)

    # id ata
    for i, r in enumerate(train, 1):
        r["meta"]["id"] = f"hr_{i:06d}"
    for i, r in enumerate(val, 1):
        r["meta"]["id"] = f"hr_val_{i:05d}"

    report = build_report(all_rows, train, val, args)
    print(report)
    print(f"[i] atlanan (yakın-kopya) örnek sayısı: {g.skipped}")

    if args.sample:
        print("\n" + "=" * 70 + "\n ÖRNEKLER\n" + "=" * 70)
        for r in rng.sample(all_rows, min(args.sample, len(all_rows))):
            print(f"\n--- [{r['meta']['decision']}] {r['meta']['intent']} "
                  f"({r['meta']['domain']}, {r['meta']['difficulty']}, {r['meta']['register']}) ---")
            for m in r["record"]["messages"]:
                print(f"  {m['role'].upper()}: {m['content']}")

    if args.dry_run:
        print("\n[dry-run] dosya yazılmadı.")
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = args.prefix

    def meta_line(r):
        d = dict(r["meta"])
        d["tools"] = r["record"]["tools"]
        d["messages"] = r["record"]["messages"]
        return d

    write_jsonl(out / f"{p}_train.jsonl", [r["record"] for r in train])
    write_jsonl(out / f"{p}_val.jsonl", [r["record"] for r in val])
    write_jsonl(out / f"{p}_train.meta.jsonl", [meta_line(r) for r in train])
    write_jsonl(out / f"{p}_val.meta.jsonl", [meta_line(r) for r in val])
    (out / f"{p}_tools.json").write_text(
        json.dumps(list(TOOLS.values()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")

    # Üretim raporu: KANONİK data/ dizinine yazılıyorsa depo docs/'una, aksi halde
    # (geçici/deneysel çıktı) yalnız o dizine — repo raporu böylece test/deneme
    # koşularından korunur.
    canonical_data = Path(__file__).resolve().parent.parent / "data"
    if out.resolve() == canonical_data.resolve():
        report_path = Path(__file__).resolve().parent.parent / "docs" / "generation_report.md"
    else:
        report_path = out / "generation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")

    print(f"\n[✓] veri -> {out}   |   rapor -> {report_path}")
    for fn in [f"{p}_train.jsonl", f"{p}_val.jsonl", f"{p}_train.meta.jsonl",
               f"{p}_val.meta.jsonl", f"{p}_tools.json"]:
        fp = out / fn
        print(f"    {fn:36s} {fp.stat().st_size/1024:8.1f} KB")


if __name__ == "__main__":
    main()
