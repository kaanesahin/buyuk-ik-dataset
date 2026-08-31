# Büyük İK — Qwen LoRA tool-calling / tool-routing dataset

NVIDIA **When2Call** yaklaşımının **Büyük İK** (sentetik İnsan Kaynakları) alanına
uyarlanmış sentetik veri seti. Amaç modele "hangi cümlede hangi tool vardı"yı
ezberletmek değil; **niyet + mevcut yetenekler + parametre yeterliliği + yetki/onay**
ekseninde doğru aksiyonu seçmeyi öğretmektir.

**Güncel sürüm:** 3000 örnek · 22 tool · dağılım `tool_call` %30 / `direct` %25 /
`request_for_info` %25 / `cannot_answer` %20 · doğrulama 0 hata / 0 uyarı ·
`tests/` altında 25 dosya / ~320 pytest testi.

---

## Depo düzeni

```
buyuk_ik_lora_dataset/
├── data/                                  KANONİK — eğitim için kullanılan dosyalar
│   ├── buyuk_ik_tool_calling_train.jsonl       satır başına {"tools":[...], "messages":[...]}
│   ├── buyuk_ik_tool_calling_val.jsonl         doğrulama (~%10, intent bazında stratifiye)
│   ├── buyuk_ik_tool_calling_train.meta.jsonl  aynı sırada; id + decision/intent/... (yalnız QC)
│   ├── buyuk_ik_tool_calling_val.meta.jsonl
│   ├── buyuk_ik_tool_calling_tools.json        22 tool'luk şema envanteri (bağımsız)
│   └── variants/                               opsiyonel eğitim varyantları (gitignore; üretilir)
│
├── preview/                               OKUNUR — otomatik üretilir, salt-okunur
│   ├── DATASET_PREVIEW.md                      sohbet dökümleri, karar sınıfına göre — BURADAN BAŞLA
│   ├── index.md                                ne nerede + dağılım özeti
│   └── samples/<decision>.sample.json          girintili tam kayıtlar (tools + messages)
│
├── scripts/
│   ├── generate_dataset.py                     üretici (stdlib-only, deterministik, API yok)
│   ├── validate_dataset.py                     bağımsız kalite kontrol (When2Call §31)
│   ├── make_preview.py                         preview/ üreticisi (içeriği değiştirmez)
│   └── build_training_variants.py              opsiyonel: system-turlu (ChatML) kopya üretir
│
├── tests/                                 pytest paketi (25 dosya, ~320 test) — tests/README.md
│   ├── conftest.py · pytest.ini · requirements-test.txt
│   ├── test_*.py  (çekirdek 1–10)              yapı, şema, tool-call, halüsinasyon, karar,
│   │                                           akış, dağılım, sızıntı, gizlilik, tekrarlanabilirlik
│   ├── test_*.py  (ileri 11–20)                istatistiksel denge, sözcüksel çeşitlilik,
│   │                                           tool-ayırt edilebilirliği, parametre grounding,
│   │                                           eksik-param mantığı, WRITE otomatı, tur tutarlılığı,
│   │                                           kodlama/serileştirme, müfredat, anlamsal intent
│   └── test_*.py  (sert 21–25)                 karar orakeli, mutasyon meta-testi, zamansal
│                                               akıl yürütme, ChatML şablon güvenliği, kapsama
│
├── docs/
│   ├── ANALYSIS.md                             yeterlilik analizi + kapatılan eksikler
│   ├── generation_report.md                    üretim istatistikleri (generate_dataset.py yazar)
│   └── validation_report.md                    doğrulama raporu (validate_dataset.py yazar)
│
└── README.md
```

> **`data/*.jsonl` neden tek satır?** JSONL biçimidir: her satır bağımsız bir eğitim
> örneğidir ve `datasets` / streaming yükleyiciler bunu bekler. Gözle okumak için
> `preview/` klasörünü kullan — aynı içeriğin girintili / döküm hâli.

---

## Dört karar davranışı

| decision | ne zaman | assistant çıktısı |
|---|---|---|
| `direct` | tool gerekmiyor (tanım, politika, süreç, selamlaşma) | doğrudan yanıt |
| `tool_call` | tool gerekli **ve** tüm zorunlu parametreler mevcut | `<tool_call>…</tool_call>` |
| `request_for_info` | tool var ama zorunlu bilgi eksik **veya** WRITE için onay gerekiyor | eksik bilgiyi / onayı isteyen soru |
| `cannot_answer` | mevcut araçlarla cevaplanamaz (kapsam dışı, gelecek, gizlilik, desteklenmeyen) | kibar ret + gerekçe |

### Bu sürümde kapsam

- **Çok-adımlı zincir (§25):** 99 örnek — 6 turlu `parametre topla → yazma için onay iste → uygula`.
- **Çok turlu `direct`:** 90 örnek (tanım + takip sorusu). **Çok turlu `cannot_answer`:** 72 örnek (ret + kullanıcı ısrarı + kararlı ret).
- **`cannot_answer` tüm alanlara yayıldı:** `puantaj` ve `ik_islemleri` dâhil 7 domain (§17).
- **WRITE:** `create/cancel/update_izin_talebi`, `update_employee_contact`, **`update_employee_information`** (yeni), `create_ucret/pozisyon_degisiklik_talebi`.
- Zorluk: `kolay` %16 · `orta` %42 · `zor` %29 · `cok_zor` %13.

---

## Format

`messages` yalnızca `user` / `assistant` turlarından oluşur. Tool tanımları ayrı bir
`tools` alanındadır — bu, **Qwen 2.5 sohbet şablonunun beklediği biçimdir**
(`apply_chat_template(messages, tools=tools, …)` araçları system istemine kendisi
yerleştirir). Eğitim şablonunuz araçları ayrı bir `system` turundan bekliyorsa
`scripts/build_training_variants.py` ile system-turlu bir kopya üretebilirsiniz;
kanonik dosyalar değişmez.

Tool çağrıları:

```
<tool_call>
{"name": "get_izin_bakiyesi", "arguments": {"employee_id": "EMP-1042", "izin_tipi": "yillik"}}
</tool_call>
```

Çoklu tool çağrısı = art arda birden fazla `<tool_call>` bloğu. Tool sonucu **taklit
edilmez** (SFT karar davranışını öğretir — When2Call §33); örnekler assistant'ın
kararında biter.

---

## Kullanım

```bash
# 1) dataset üret  (varsayılan: n=3000, seed=20260827, çıktı -> data/)
python scripts/generate_dataset.py

# 2) doğrula  (0 hata beklenir; rapor -> docs/validation_report.md)
python scripts/validate_dataset.py

# 3) okunur önizlemeyi tazele  (-> preview/)
python scripts/make_preview.py

# 4) (opsiyonel) system-turlu eğitim kopyası  (-> data/variants/)
python scripts/build_training_variants.py

# 5) test paketi  (pytest gerekir; bkz. tests/README.md)
pip install -r tests/requirements-test.txt
pytest tests/                     # ~320 test, ~20 sn
pytest tests/ -m "not slow"       # üretici alt süreçleri hariç
```

Yararlı bayraklar: `--n`, `--seed`, `--dry-run`, `--sample 20`, `--today 2026-08-27`,
`--val-ratio 0.1`, `--prefix <ad>`, `--out-dir <yol>`.

Üretici belirlenimcidir: aynı seed → byte-aynı çıktı. Dağılım `--n 5000`'e kadar
±%0 korunur; daha büyük set için `generate_dataset.py` içindeki `*_SPECS` şablon
havuzlarını genişletmek gerekir (bkz. `docs/ANALYSIS.md`).

---

## Tasarım notları

- **Intent generalization** — her niyet resmi/gündelik/konuşma dili/kısa/uzun/yazım
  hatalı kayıtlarda ifade edilir; ID/tarih/tutar token'ları asla bozulmaz.
- **Yakın-kopya engelleme** — tüm kullanıcı turlarının normalize imzası (rakam/ID
  silinmiş) benzersizdir; "sadece EMP-ID değiştir" klonları üretilmez.
- **Halüsinasyon yok** — `tool_call` argümanları yalnızca kullanıcı turlarında geçen
  değerlerden türetilir; eksikse `request_for_info`. Validator bunu bağımsız denetler.
- **Distractor tool'lar** — her örnekte doğru tool + aynı alandan 3–8 çeldirici.
- **Onay akışları** — `confirmation_required` tool'lar 2-turlu `request_for_info`
  (onay iste), 4-turlu `tool_call` (onaylandı) ve 6-turlu zincir olarak öğretilir.
- **Gizlilik** — başkasının maaş/izin/iletişim bilgisi talepleri `cannot_answer`;
  "yetkim var mı" tipi sorular `check_employee_access` tool'una yönlenir.

Tüm çalışan, ID, maaş, tarih, departman bilgisi **sentetiktir**. Gerçek TC kimlik,
banka hesabı, telefon veya hassas kişisel veri üretilmez.

---

## Değerlendirme

`docs/ANALYSIS.md` — setin When2Call yaklaşımına göre yeterlilik analizi, kapatılan
eksikler ve kalan yol haritası.
