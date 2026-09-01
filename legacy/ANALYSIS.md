# Büyük İK dataset — yeterlilik analizi

> İlk değerlendirme: 2026-08-27 (2000 örnek) · Revizyon: 2026-08-27 (3000 örnek,
> eksikler kapatıldı) · Revizyon 2: 2026-08-27 (pytest paketi eklendi, LF hijyeni) ·
> **Revizyon 3: 2026-08-27 (test paketi 25 dosyaya çıktı; `get_overtime` eksik-param
> şablonundaki göreli-ay ifadesi düzeltildi)** · Kapsam:
> `data/buyuk_ik_tool_calling_{train,val}.jsonl`, `scripts/*.py`, `tests/`.

Bu belge, üretilen setin **When2Call** yaklaşımına ve orijinal üretim promptuna göre
ne kadar yeterli olduğunu değerlendirir.

**Sonuç:** İlk değerlendirmede bildirilen 6 eksikten 5'i giderildi, 1'i (system-turlu
şablon) opsiyonel bir araçla karşılandı. Set **3000 örnek hedefinde eğitime hazırdır**;
`tests/` altında **25 dosya / ~320 pytest testinin tamamı geçer** (çekirdek 1–10 +
ileri 11–20 + sert 21–25).

---

## 1. Güçlü yanlar

| Başlık | Bulgu |
|---|---|
| Yapısal geçerlilik | 3000 örnek; `validate_dataset.py` **0 hata / 0 uyarı** + `tests/` (25 dosya, ~320 test) |
| Kodlama hijyeni | Tüm `.jsonl` UTF-8 (BOM'suz), **LF** satır sonu, `\n` ile biter → platformlar arası byte-aynı |
| Karar dağılımı | `tool_call` %30.0 · `direct` %25.0 · `request_for_info` %25.0 · `cannot_answer` %20.0 — prompt §6 hedefiyle birebir |
| 4 davranış ayrımı | Her karar tipi için validator'da ayrı tutarlılık kuralı |
| Halüsinasyon önleme | `trace_arg()` her tool-call argümanını (employee_id, tarih, dönem, tutar, talep_id, enum) kullanıcı turundan **bağımsız olarak** doğrular; assistant düz metninde uydurma ID/sayı ayrıca taranır |
| Tool *seçimi* öğretimi | Her örnekte doğru tool + `CONFUSABLE` haritasından 3–8 çeldirici |
| Tool envanteri | **22 tool** (yeni: `update_employee_information`), flat-function şema, `enum` + `required` |
| Dil kaydı çeşitliliği | resmi / gündelik / konuşma dili / uzun / yazım hatalı / kısa — ID, tarih, tutar token'ları korunur |
| Çok turlu | **486 örnek** (2/4/6 tur) |
| Çok-adımlı zincir (§25) | **99 örnek** — 6 turlu `parametre topla → yazma için onay iste → uygula` |
| Multi-tool | 63 örnek |
| WRITE akışları | 690 örnek; 2-turlu "onay iste", 4-turlu "onaylandı", 6-turlu zincir |
| Train/val sızıntısı | **Sıfır** — normalize-imza düzeyinde bile |
| Belirlenimcilik | Sabit seed; regen byte-aynı çıktı verir |
| Gizlilik | Tümü sentetik; gerçek TC kimlik, IBAN, telefon yok |

### Domain kapsamı (tam veri)

| domain | adet | oran |
|---|---:|---:|
| `ik_islemleri` | 916 | %30.5 |
| `maas_finans` | 625 | %20.8 |
| `izin_yonetimi` | 495 | %16.5 |
| `puantaj` | 315 | %10.5 |
| `organizasyon` | 256 | %8.5 |
| `calisan_bilgileri` | 191 | %6.4 |
| `kapsanmayan` | 125 | %4.2 |
| `meta` | 77 | %2.6 |

**122 benzersiz intent** (ilk sürüm: 97). Zorluk: `kolay` %15.6 · `orta` %42.5 ·
`zor` %29.0 · `cok_zor` %12.9. Register: 6 kayıt, %4.8–%24.9 arası dağılmış.

`cannot_answer` domain yayılımı: 7 alanın tamamı, 39–155 örnek aralığında
(`puantaj` 60, `ik_islemleri` 76).

---

## 2. İlk değerlendirmedeki eksikler — durum

### 2.1 Ölçek tavanı  ✅ GİDERİLDİ (~2500 → ~5000)

Şablon havuzları genişletildi: `DIRECT_INTENTS` 40 → 53, `CANNOT_INTENTS` 31 → 42,
`READ_SPECS` şablonları 72 → 113, `MISSING_PARAM_SPECS` 20 → 24,
`WRITE_SPECS` 6 → 7 aile / user şablonu 29 → 36, slot havuzları (tarih aralıkları
10 → 18, dönemler 14 → 21, gerekçeler 5 → 11, pozisyonlar 8 → 16, isimler 12 → 24,
departman yüzeyleri +12).

`--n` testleri: **3000 → %30/25/25/20 birebir**, **4000 → birebir**, **5000 → birebir**.
5000'in ötesinde yakın-kopya eleme yeniden agresifleşir.

### 2.2 `cannot_answer` domain dengesizliği  ✅ GİDERİLDİ (§17)

Önce: `puantaj` 0, `ik_islemleri` 18. Sonra: **7 domain'in tamamı 40–145 aralığında.**
Eklenen intent'ler:

- `puantaj`: `coworker_timesheet`, `bulk_lateness_ranking`, `predict_future_overtime`, `edit_own_timesheet`
- `ik_islemleri`: `approve_own_leave`, `approve_on_behalf_of_manager`, `bulk_process_all_requests`, `permanently_delete_record`, `reset_manager_credentials`
- ek: `coworker_position_history`, `predict_own_leave_rejection`

### 2.3 Gerçek çok-adımlı akıl yürütme zinciri  ✅ GİDERİLDİ (§25)

`MULTI_STEP_SPECS` + `gen_multi_step`: 6 turlu zincir. Model önce eksik parametreyi
ister (uygunluk/onay adımını sonraya bırakır), parametre gelince **yine de yazma
için onay ister**, onay gelince `tool_call` yapar. 99 örnek, `cok_zor`.

> Sınır: dataset konvansiyonu gereği (§33) tool SONUCU taklit edilmez; bu yüzden
> "izin bakiyesini API'den çek → sonuca göre dallan" gibi tool-cevabına dayanan
> gerçek dallanma bu sette YOK. Zincir, "önce topla, sonra onayla, sonra uygula"
> sıralamasını ve parametre uydurmama davranışını öğretir.

### 2.4 Çok turlu `direct` / `cannot_answer`  ✅ GİDERİLDİ

- `MT_DIRECT_SPECS` + `gen_mt_direct`: tanım sorusu → yanıt → takip sorusu → yanıt (90 örnek).
- `MT_CANNOT_PUSH/HOLD` + `gen_mt_cannot`: kapsam dışı istek → kibar ret → kullanıcı ısrarı → kararlı ret (72 örnek).

### 2.5 WRITE kapsamı  ✅ GİDERİLDİ

- Yeni tool **`update_employee_information`** (medeni durum, öğrenim durumu, acil durum
  kişisi/telefonu) — `WRITE_SPECS`, `MISSING_PARAM_SPECS`, `CONFUSABLE`, `DOMAIN_TOOLS`'a eklendi.
- `update_izin_talebi` için eksik-parametre (`talep_id`) `request_for_info` varyantı eklendi.
- `get_puantaj` / `get_mesai_bilgisi` için ek eksik-parametre varyantları eklendi.

### 2.6 `system` rolü / tools-in-system-message  ✅ KARŞILANDI (opsiyonel araç)

Kanonik `{tools, messages}` biçimi **Qwen 2.5'in native formatıdır** — tokenizer,
`apply_chat_template(messages, tools=tools, ...)` çağrısında araçları system istemine
kendisi koyar. Ek bir şey gerekmez.

Eğitim şablonunuz araçları ayrı bir `system` turundan bekliyorsa:
`python scripts/build_training_variants.py` → `data/variants/*_chatml_system.jsonl`
(system turu araç tanımlarını içerir, ayrı `tools` alanı yoktur). İçerik birebir korunur.

---

## 3. Kalan sınırlamalar (düşük öncelik)

1. **Tool-cevabına dayalı gerçek dallanma yok** — §33 konvansiyonu gereği. İleride
   tool-response trajectory'li ayrı bir alt-set eklenebilir.
2. **`cok_zor` ~%13** — hedeflenen ~%15'in biraz altında. `n_multi_step` payı
   artırılarak yükseltilebilir; şu an kalite/çeşitlilik dengesinde tutuldu.
3. **5000+ ölçek** — bu hacim isteniyorsa şablon havuzları bir tur daha genişletilmeli.
4. **Değerlendirme seti** — ayrı bir hold-out "hard eval" seti (When2Call'ın kendi
   değerlendirme yaklaşımı gibi) henüz yok; `val` split intent-stratifiye ama aynı
   şablon havuzundan geliyor.

---

## 4. Kalite güvence katmanları

| Katman | Ne yapar | Nerede |
|---|---|---|
| `validate_dataset.py` | Hızlı kapı: yapısal + şema + halüsinasyon izleme + dağılım; hata varsa exit 1 | CI'nin ilk adımı |
| `tests/` çekirdek (1–10, ~145 test) | Yapısal/anlamsal temel güvenceler; isimlendirilmiş testler, ayrıntılı hata mesajı | `pytest tests/` |
| `tests/` ileri (11–20, ~95 test) | İstatistiksel (ki-kare), bilgi-kuramsal (entropi/n-gram), biçimsel-otomat (WRITE), grounding-kesinliği, müfredat monotonluğu, anlamsal intent kohezyonu | `pytest tests/` |
| `tests/` sert (21–25, ~80 test) | Orakel (etiketi ilk ilkelerden yeniden türet), mutasyon meta-testi, zamansal akıl yürütme, sohbet-şablonu güvenliği, tasarım-uzayı kapsaması | `pytest tests/` |
| derin QC (üretim sırasında) | `generate_dataset` içi sayaç/imza kontrolleri | otomatik |

İleri + sert kümenin öne çıkan denetimleri: **karar etiketinin kendisi** When2Call
sürecinin bir orakeliyle 3000 örnekte yeniden türetilir; `validate_dataset.check_record`
**mutasyon testine** tabi tutulur (14 kusur türü enjekte → hepsi yakalanmalı); karar
dağılımı hedeften **istatistiksel olarak ayırt edilemiyor** (χ²); her tool argümanı
üreticinin **kapalı slot havuzundan**; tarih argümanları **takvim-geçerli** ve iş
mantığında tutarlı (izinler geleceğe dönük); WRITE akışı geçerli bir **durum-makinesi
yolu**; **intra-intent kohezyon inter-intent ayrışmanın ~17 katı** (Jaccard); tool_call
blokları `tool_call_block` çıktısıyla **birebir yeniden üretilebilir**; 22 tool + tüm
slot/ret havuzları %100 örneklenmiş.

---

## 5. Yeniden üretim

```bash
python scripts/generate_dataset.py      # -> data/ + docs/generation_report.md
python scripts/validate_dataset.py      # -> docs/validation_report.md  (0 hata beklenir)
python scripts/make_preview.py          # -> preview/
python scripts/build_training_variants.py   # opsiyonel -> data/variants/
pip install -r tests/requirements-test.txt && pytest tests/   # 25 dosya / ~320 test
```

Tümü belirlenimcidir; `--seed` sabit tutulduğunda çıktı byte-aynıdır (LF).
