# Büyük İK LoRA Dataset — Fine-Tune Uygunluk Raporu

> Bağımsız dış inceleme · Tarih: 2026-08-28 · Kapsam: deponun **tüm** dosyaları
> (data/, scripts/, tests/, docs/, preview/, kök).
> Yöntem: her dosya elle okundu; veri seti Python ile yeniden çözümlendi;
> `validate_dataset.py`, `pytest tests/` ve deterministik yeniden üretim
> **fiilen çalıştırılıp** doğrulandı. Bu rapordaki her sayı üretilen dosyalardan
> ölçülmüştür (bkz. §11 — Ölçüm günlüğü).

---

## 0. Sonuç (yönetici özeti)

**Hedeflenen yapı:** NVIDIA When2Call'ın 4 karar çerçevesini (`direct` /
`tool_call` / `request_for_info` / `cannot_answer`) Qwen 2.5 LoRA ile,
`{tools, messages}` biçiminde, Türkçe İK alanında öğreten bir **tool-calling /
tool-routing** SFT seti.

**Karar: Bu hedef için veri seti EĞİTİME HAZIR ("yüksek uygunluk").**
Yapısal, şema, dağılım, güvenlik-otomatı ve halüsinasyon eksenlerinde kusur
bulunamadı; format Qwen 2.5'in native biçimidir; üretim deterministiktir.

**Ancak 4 tasarım sınırı, fine-tune sonrası davranışı doğrudan belirler ve
eğitime başlamadan önce bilinçli karar gerektirir:**

| # | Sınır | Etki |
|---|---|---|
| S-1 | **Tool SONUCU turu yok** (`tool` / `<tool_response>` rolü hiç yok) | Model tool çağırıp *durmayı* öğrenir; dönen sonucu yorumlayıp cevap üretmeyi **öğrenmez**. Gerçek agent'ta bu ikinci adım ayrıca gerekir. |
| S-2 | **Kapalı ve küçük slot havuzları** | 18 tarih-aralığı yüzeyi, 21 dönem, 10 talep-ID, ~10 tutar, 16 pozisyon, 15 departman. Model bu eşlemeleri ezberler; havuz dışı ifadeler ("14 Mart 2027") güvenilir çözülmeyebilir. |
| S-3 | **`direct` / `cannot_answer` cevap çeşitliliği düşük** | `direct`: 750 örnekte yalnız **133 benzersiz** cevap (bazıları 14×). Kalıp cevap ezberleme riski. |
| S-4 | **`val` gerçek hold-out değil** | Aynı şablon havuzundan, intent-stratifiye üretiliyor. Genelleme ölçmez; ayrı "hard eval" seti yok. |

Ayrıntı: §5 (uygunluk) ve §6 (riskler).

---

## 1. Neyin ne olduğu — depo envanteri

| Yol | Satır | Boyut | Tür | Durum |
|---|---:|---:|---|---|
| `data/buyuk_ik_tool_calling_train.jsonl` | 2 708 | 10.0 MB | Kanonik eğitim | ✔ geçerli |
| `data/buyuk_ik_tool_calling_val.jsonl` | 292 | 1.1 MB | Kanonik doğrulama | ✔ geçerli |
| `data/buyuk_ik_tool_calling_train.meta.jsonl` | 2 708 | 11.0 MB | QC/etiket (eğitimde kullanılmaz) | ✔ sırası birebir |
| `data/buyuk_ik_tool_calling_val.meta.jsonl` | 292 | 1.2 MB | QC/etiket | ✔ sırası birebir |
| `data/buyuk_ik_tool_calling_tools.json` | 533 | 15.6 KB | 22 tool şema envanteri | ✔ üretici ile birebir |
| `data/variants/*.chatml_system.jsonl` | 2 708 / 292 | — | Opsiyonel system-turlu kopya (gitignore) | ✔ içerik korunmuş |
| `scripts/generate_dataset.py` | 3 286 | 186 KB | Üretici (stdlib-only, deterministik) | ✔ çalışır |
| `scripts/validate_dataset.py` | 500 | 20 KB | Bağımsız QC kapısı | ✔ 0 hata / 0 uyarı |
| `scripts/make_preview.py` | 344 | 14 KB | preview/ üreticisi | ✔ içeriği değiştirmez |
| `scripts/build_training_variants.py` | 102 | 4 KB | system-turlu varyant üreticisi | ✔ opsiyonel |
| `tests/` (25 × `test_*.py` + conftest + pytest.ini) | ~4 900 | — | pytest paketi | ✔ **321 test geçti** |
| `docs/ANALYSIS.md` | 162 | — | Elle yazılan yeterlilik analizi | güncel |
| `docs/generation_report.md` | 83 | — | Üretici çıktısı | güncel |
| `docs/validation_report.md` | 16 | — | Validator çıktısı | güncel |
| `preview/DATASET_PREVIEW.md` + `index.md` + `samples/*.json` | — | — | Okunur önizleme (otomatik) | güncel |
| `README.md`, `.gitignore` | 156 / 8 | — | — | güncel |

**Toplam eğitim örneği: 3 000** (2 708 train + 292 val).
`meta` satırları ile veri satırları arasında **0 uyumsuzluk** (birebir aynı `messages`).

---

## 2. Kanonik veri dosyaları — `data/*.jsonl`

### 2.1 Kayıt yapısı

Her satır bağımsız bir JSON nesnesi: `{"tools": [...], "messages": [...]}`.

- `tools`: o örnekte modele sunulan araç listesi — **doğru tool + 3–8 çeldirici**
  (kayıt başına 4–9 tool; dağılım: 4→157, 5→662, 6→871, 7→697, 8→589, 9→24).
- `messages`: yalnız `user` / `assistant` turları. **`system` turu yok, `tool`
  turu yok** (bilinçli — bkz. §5.1).
- `_meta` alanı kanonik dosyalarda **yok**; yalnız `*.meta.jsonl`'de.

### 2.2 Doğrulanan yapısal nitelikler (kusursuz)

| Kontrol | Sonuç |
|---|---|
| Her satır geçerli JSON | ✔ 3000/3000 |
| Kodlama: UTF-8, BOM yok | ✔ |
| Satır sonu: LF (CRLF sayısı 0), dosya `\n` ile biter | ✔ |
| Rol sırası: ilk `user`, son `assistant`, katı alternasyon | ✔ 3000/3000 (0 ihlal) |
| Boş içerikli mesaj | ✔ yok |
| `tool_call` yalnız son assistant turunda, `tool_call` kararında | ✔ 0 kaçak |
| Assistant turunda `<tool_call>` bloğundan önce düz metin | ✔ 0 (blok tek başına) |
| `<tool_response>` / özel ChatML token'ı sızıntısı | ✔ yok |

### 2.3 Karar dağılımı — hedefle birebir

| decision | hedef | gerçek | oran |
|---|---:|---:|---:|
| `tool_call` | %30 | 900 | **%30.0** |
| `direct` | %25 | 750 | **%25.0** |
| `request_for_info` | %25 | 750 | **%25.0** |
| `cannot_answer` | %20 | 600 | **%20.0** |

`validate_dataset.py` toleransı ±3.5 puan; gerçekleşen sapma **0.0**.
`test_statistical_balance.py` ki-kare ile "hedeften istatistiksel olarak ayırt
edilemez" doğruluyor.

### 2.4 İkincil dağılımlar (tam veri, 3000)

**Domain:**

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

**Zorluk:** `orta` %42.5 · `zor` %29.0 · `kolay` %15.6 · `cok_zor` %12.9
(hedeflenen ~%15 `cok_zor`'un biraz altında — ANALYSIS.md kabul ediyor).

**Register (dil kaydı):** `gundelik` %24.9 · `resmi` %22.2 · `konusma_dili` %18.3
· `uzun` %18.3 · `yazim_hatali` %11.4 · **`kisa` %4.8** (az temsil — bkz. §6.7).

**Tur:** 2 tur 2514 · 4 tur 387 · 6 tur 99. Çok turlu toplam **486**, bunun
**99'u** 6-turlu "topla → onay → uygula" zinciri. Çoklu-tool **63**.

**is_write:** 690 örnek (`direct`/`cannot_answer`'da hiç yok — doğru).
`confirmation_required`: 491.

### 2.5 Karar × domain ızgarası (gözlemlenen tasarım sözleşmesi)

```
domain              tool_call req_info direct cannot
izin_yonetimi            130      55    251     59
maas_finans              222      93    208    102
puantaj                   91      80     90     54
organizasyon             104      30      0    122
calisan_bilgileri         84      45     13     49
ik_islemleri             269     447    111     89
meta                       0       0     77      0
kapsanmayan                0       0      0    125
```

- `meta` (selamlaşma/kimlik/yetenek) → **yalnız `direct`**.
- `kapsanmayan` (hava, yemekhane, yatırım…) → **yalnız `cannot_answer`**.
- `organizasyon` → `direct` yok (tanım sorusu bu domaine atanmamış).
- `ik_islemleri` → `request_for_info` ağırlıklı (447) — WRITE onay akışları burada.

`test_statistical_balance.py` bu bağımlılığı "tasarım sözleşmesi" olarak kilitliyor.

### 2.6 Intent kapsamı

**122 benzersiz intent.** 19 intent hem `tool_call` hem `request_for_info`
altında görünüyor (aynı niyet — parametre var / eksik kontrastı; When2Call'ın
istediği ayrım). Hiçbir intent karar sınıfları arasında karışmıyor (ör. bir
`direct` intent'i asla `tool_call` üretmiyor).

En sık intent'ler: `create_leave_request` 212 · `update_leave_request` 114 ·
`get_payslip` 93 · `get_timesheet` 91 · `update_information` 82. En seyrek:
`expense_advance_difference` 6 · `should_i_resign` 7 · `lactation_leave_general` 7.

### 2.7 Train / val bölmesi

- Intent bazında stratifiye; `len(items) < 8` olan intent'lerde `k=0` (val'a
  hiç örnek gitmez).
- **val'da 119/122 intent var**; eksik 3'ü (`should_i_resign`,
  `lactation_leave_general`, `expense_advance_difference`) hepsi n≤7.
- val karar dağılımı: 30.5 / 26.0 / 23.6 / 19.9 — kabul edilebilir yakınlık.
- val: 8 zincir, 51 çok turlu, 8 domain, 7 meta örneği.
- **Train ↔ val sızıntısı: 0** (normalize edilmiş imza düzeyinde bile;
  `test_diversity_and_leakage.py` ayrıca klasik metin karşılaştırmasıyla teyit).

---

## 3. `data/*.meta.jsonl` — etiket dosyaları

Her satır: `decision`, `intent`, `target_tool(s)`, `required_parameters`,
`missing_parameters`, `is_write`, `confirmation_required`, `domain`, `difficulty`,
`register`, `multi_turn`, `chain`, `turns`, `id`, ayrıca `tools` + `messages`
(kontrol için tekrar).

- **Eğitimde kullanılmaz** — yalnız QC / değerlendirme / hata ayıklama.
- Satır sırası eğitim dosyasıyla **birebir aynı** (doğrulandı: 0 uyumsuzluk).
- `id` biçimi: train `hr_000001…`, val `hr_val_00001…`.
- `difficulty` alanı **türetilmiş** (heuristik): temel zorluk + çok-turlu ise +1
  kademe + uzun metin ise +1 kademe (`bump_difficulty`). Yani "içsel" değil
  "hesaplanmış" bir etiket; `test_curriculum_and_difficulty.py` bir karmaşıklık
  skorunun kesin monoton (kolay<orta<zor<cok_zor) olduğunu doğruluyor.

---

## 4. `data/buyuk_ik_tool_calling_tools.json` — 22 tool

`generate_dataset.py` içindeki `TOOLS` sözlüğüyle **birebir aynı** (ad + sıra).
Şema: düz "function" biçimi — `{name, description, parameters:{type:object,
properties, required}}`, JSON-Schema alt kümesi. `enum` ve `required` kullanılıyor;
iç içe nesne / dizi yok.

| # | Tool | Zorunlu param | Opsiyonel | WRITE? | Onay? |
|---|---|---|---|:--:|:--:|
| 1 | `get_employee_info` | employee_id | — | | |
| 2 | `get_employee_status` | employee_id | — | | |
| 3 | `get_departman_bilgisi` | departman_adi | — | | |
| 4 | `get_calisan_listesi` | departman_adi | durum(enum) | | |
| 5 | `get_yonetici_bilgisi` | employee_id | — | | |
| 6 | `get_izin_bakiyesi` | employee_id | izin_tipi(enum) | | |
| 7 | `get_izin_gecmisi` | employee_id | baslangic/bitis_tarihi | | |
| 8 | `get_izin_talebi_durumu` | employee_id | talep_id | | |
| 9 | `create_izin_talebi` | employee_id, izin_tipi, baslangic, bitis | **aciklama** | ✔ | ✔ |
| 10 | `cancel_izin_talebi` | talep_id | — | ✔ | ✔ |
| 11 | `update_izin_talebi` | talep_id | yeni_baslangic/bitis | ✔ | ✔ |
| 12 | `get_maas_bilgisi` | employee_id | tur(enum) | | |
| 13 | `get_bordro` | employee_id, donem | — | | |
| 14 | `get_prim_bilgisi` | employee_id | donem | | |
| 15 | `get_yan_haklar` | employee_id | — | | |
| 16 | `create_ucret_degisiklik_talebi` | employee_id, yeni_brut_ucret, gerekce | **gecerlilik_tarihi** | ✔ | ✔ |
| 17 | `create_pozisyon_degisiklik_talebi` | employee_id, yeni_pozisyon | gerekce, gecerlilik_tarihi | ✔ | ✔ |
| 18 | `get_puantaj` | employee_id, baslangic, bitis | — | | |
| 19 | `get_mesai_bilgisi` | employee_id, donem | — | | |
| 20 | `update_employee_contact` | employee_id | telefon, email, adres | ✔ | ✔ |
| 21 | `update_employee_information` | employee_id | medeni_durum, ogrenim_durumu, acil_durum_kisisi/telefonu | ✔ | ✔ |
| 22 | `check_employee_access` | requester_id, hedef_employee_id, kaynak_tipi(enum) | — | | |

**Gözlem:** `create_izin_talebi.aciklama`, `create_ucret_degisiklik_talebi.gecerlilik_tarihi`,
`create_pozisyon_degisiklik_talebi.gecerlilik_tarihi`/`gerekce` gibi **opsiyonel
parametreler tüm veri setinde hiç doldurulmuyor** (0 kez). Model, kullanıcı
gerekçe/açıklama verse bile bu alanları argümana taşımayı öğrenmez (bkz. §6.9).

Tool çağrılarında fiilen kullanılan argüman anahtarları ve frekansları:
`employee_id` 789 · `baslangic_tarihi`/`bitis_tarihi` 152'şer · `donem` 142 ·
`izin_tipi` 120 · `talep_id` 81 · `departman_adi` 74 ·
`yeni_baslangic/bitis_tarihi` 54'er · `requester_id`/`hedef_employee_id`/`kaynak_tipi`
26'şar · `durum` 24 · `yeni_pozisyon` 21 · `yeni_brut_ucret`/`gerekce` 20'şer ·
`tur` 17 · `telefon` 10 · `adres` 7 · `ogrenim_durumu`/`acil_durum_telefonu` 7'şer.

**"Distractor" (çeldirici) tasarımı** — `CONFUSABLE` haritası: her tool için
anlamca yakın 3–4 kardeş tanımlı. Örneklerin **%89.9'unda** tool listesinde
hedefin en az bir karıştırılabilir komşusu var; kalanında rastgele doldurma.
Hedef tool **her örnekte** listede (0 eksik). Bu, "hangi cümlede hangi tool"u
ezberletmek yerine ayırt etmeyi öğretir — When2Call §16 ile uyumlu.

---

## 5. Hedeflenen fine-tune yapısına uygunluk

### 5.1 Format — Qwen 2.5 native ✔

Kanonik `{tools, messages}` biçimi tam olarak
`tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=...)`
çağrısının beklediği girdidir; tokenizer araç tanımlarını system istemine kendisi
yerleştirir. **Ek dönüşüm gerekmez.**

Tool çağrı biçimi kesin ve tutarlı:
```
<tool_call>
{"name": "get_bordro", "arguments": {"employee_id": "EMP-5939", "donem": "2026-01"}}
</tool_call>
```
Çoklu çağrı = art arda bloklar (aralarında `\n`). `tool_call_block()` üreticisinin
çıktısıyla **birebir yeniden üretilebilir** (`test_encoding_and_serialization.py`).

> **`tool` rolü / tool sonucu bilinçli olarak yok** (When2Call §33). Örnekler
> assistant'ın *kararında* biter. Bu, "SFT karar davranışını öğretir, sonuç
> uydurmayı öğretmez" ilkesidir — ama S-1'de belirtilen dağıtım sonucu doğurur.

Eğitim şablonunuz araçları ayrı bir `system` turundan bekliyorsa
`scripts/build_training_variants.py` → `data/variants/*_chatml_system.jsonl`
(TR veya Hermes önsözü; içerik birebir korunur). Qwen 2.5 için **gerekmez**.

### 5.2 Şema / tool-call geçerliliği — kusursuz ✔

900 `tool_call` kararının tamamında (fiili blok sayısı > 900, çoklu-tool dâhil):

| Kontrol | İhlal |
|---|---:|
| JSON parse hatası | 0 |
| Bilinmeyen tool adı | 0 |
| Şemada olmayan argüman anahtarı | 0 |
| `enum` ihlali | 0 |
| Eksik zorunlu argüman | 0 |
| Bozuk `<tool_call>` blok biçimi | 0 |

### 5.3 Halüsinasyon önleme — yapısal olarak sağlam ✔ (bir nüansla)

**Bağımsız izleme sonucu: uydurulmuş hiçbir argüman değeri yok.** Her
`tool_call` argümanı bir kullanıcı turundaki bir ifadeye şu yollardan biriyle
bağlanıyor (240 "birebir eşleşmeyen" değerin kategori dökümü):

| Mekanizma | Adet | Örnek |
|---|---:|---|
| Tarih çözümleme (TR ay adı / "N günlük" / çeyrek → ISO) | 402 | "3 Ekim 2026 ile 7 Ekim 2026 arası" → `2026-10-03` / `2026-10-07` |
| Çıplak sayı → `EMP-XXXX` | 307 | "3311 numaralı çalışan" → `EMP-3311` |
| Dönem çözümleme ("geçen yıl" / "Mayıs 2026" → `2025` / `2026-05`) | 126 | — |
| Enum yüzey/anlam eşlemesi ("yıllık"→`yillik`, "izinde olan"→`izinli`, "sağlık"→`hastalik`) | 95 + 14 | — |
| Departman yüzeyi → kanonik ad ("denetim"→`İç Denetim`, "it"→`Bilgi Teknolojileri`) | 27 | — |
| Pozisyon büyük/küçük harf normalizasyonu | 2 | "kidemli is analisti" → `Kıdemli İş Analisti` |
| Adres noktalama/harf normalizasyonu | 1 | kullanıcı adresi birebir verdi, yalnız biçim düzeltildi |

Assistant **düz metninde** (onay özeti vb.) uydurulmuş `EMP-`/`LV-` token'ı: **0**
(onay özetlerindeki tüm ID'ler kullanıcının verdiği çıplak sayıdan türüyor).
`request_for_info` / `cannot_answer` yanıtlarında kullanıcının vermediği
gün/TL/saat sayısı: **0**.

**Nüans (S-2 ile bağlantılı):** Bu "halüsinasyon yok" garantisi, üreticinin
**kapalı yüzey haritalarından** gelir ve `validate_dataset.py` *aynı haritaları*
paylaşır — yani doğrulama kısmen dairesel. Bağımsız katmanlar var
(`test_no_hallucination.py`, `test_decision_oracle.py` — etiketi ilk ilkelerden
yeniden türetir, `test_adversarial_mutation.py` — 14 kusur enjekte edip
validator'ın hepsini yakaladığını kanıtlar), ama hepsi üreticinin ürettiği
kategori uzayı içinde çalışır. Model **açık bir NER + Türkçe tarih ayrıştırma +
enum folding + departman eşleme** görevini birlikte öğrenmek zorunda kalır; bazı
ekipler argümanların birebir (verbatim) olmasını tercih eder.

### 5.4 Karar davranışı ayrımı ✔

- `direct` → tool_call yok, doğrudan cevap. 750/750.
- `tool_call` → son turda blok var. 900/900.
- `request_for_info` → tool_call yok; son tur bilgi/onay isteği. 750/750
  (15'i "?" içermiyor ama hepsi geçerli koşullu istek: "Çalışan numarasını
  paylaşırsanız kaydı getirebilirim." — bkz. §6.10).
  Alt kırılım: eksik-bilgi (is_write=False) 303 · WRITE eksik-param 199 ·
  WRITE onay-iste 248.
- `cannot_answer` → tool_call yok; kibar ret + gerekçe. 600/600. 7 domaine yayılı
  (`puantaj` 54, `ik_islemleri` 89 dâhil — When2Call §17 kapatıldı).

### 5.5 WRITE güvenlik otomatı ✔

- **Onaysız WRITE `tool_call` YOK.** 2-turlu bir örnek asla WRITE ile bitmiyor.
- WRITE `tool_call` tur dağılımı: 4 tur (onay-iste → onay → çağrı) **144** ·
  6 tur (zincir: eksik param → onay → çağrı) **99**. READ `tool_call`: 2 tur 576 · 4 tur 81.
- Onay isteği işlemi **somut** anlatıyor (emp no / talep no / tarih / tutar onay
  turunda geçiyor — `test_write_safety_state_machine.py`).
- Onay sözcükleri olumlu ("Onaylıyorum.", "Evet, doğru. Devam et." — 12 varyant).

### 5.6 Çok-adımlı akıl yürütme — kısmi ✔

99 örnek 6-turlu zincir: **eksik parametreyi iste → parametre gelince YİNE de
yazma için onay iste → onay gelince `tool_call`**. "Önce topla, sonra onayla,
sonra uygula" sıralamasını ve parametre uydurmama davranışını öğretir.

> **Sınır:** Tool SONUCUNA dayalı dallanma ("izin bakiyesini çek → yetersizse
> reddet") bu sette **yok** (S-1'in doğrudan sonucu). Zincir yalnız
> kullanıcı-girdisine dayalı.

### 5.7 Determinizm & hijyen ✔

- `--seed 20260827` → **byte-aynı çıktı** (SHA-256 ile 5 dosyada da doğrulandı).
- Tüm çıktı LF; UTF-8; BOM yok; NFC-normal (`test_encoding_and_serialization.py`).
- Üretici **stdlib-only**; `pytest` yalnız test bağımlılığı.

### 5.8 Gizlilik / güvenlik ✔

- Tüm çalışan/ID/maaş/tarih **sentetik**. Gerçek TCKN / IBAN / telefon deseni yok
  (`test_privacy_and_safety.py`).
- Başkasının maaş/izin/iletişim/puantaj talebi → `cannot_answer`.
- "Yetkim var mı" tipi sorular → `check_employee_access` tool'una yönleniyor.
- Yıkıcı/yetki-dışı istekler ("kaydı sil", "izni sen onayla", "şifre sıfırla") →
  `cannot_answer` (`unsupported` ret havuzu).

---

## 6. Riskler ve sınırlamalar (fine-tune öncesi karar gerektirenler)

### 6.1 [S-1] Tool sonucu turu yok — **en önemli**

Model `<tool_call>` üretip durur. Gerçek bir İK agent'ında tool çalışır, JSON
sonuç döner ve modelin **onu Türkçe cevaba dönüştürmesi** gerekir — bu ikinci
inference turu veri setinde hiç yok.

**Etki:** Fine-tune sonrası model, tool sonucunu özetleme yeteneğini yalnız base
modelin genel becerisinden alır; LoRA bu davranışı pekiştirmez, hatta "assistant
turu = ya tool_call ya kısa ret/soru" kalıbıyla **zayıflatabilir**.

**Seçenekler:**
- (a) Kabul et; değerlendirmede tool-sonucu-özetleme yeteneğini ayrıca ölç ve
  base modelin yeterli olduğundan emin ol.
- (b) Küçük bir ek alt-set üret: `user → assistant(tool_call) → tool(result) →
  assistant(Türkçe özet)` turlu 300–800 örnek. When2Call §33'ün dışına çıkar ama
  dağıtımı tamamlar.

### 6.2 [S-2] Kapalı, küçük slot havuzları → genelleme tavanı

Fiilen kullanılan benzersiz değerler:
- Tarih (ISO, çağrı argümanında): **50** farklı değer, 18 yüzey şablonundan.
- Talep ID: **10** (`LV-2026-0148` … `LV-2025-1140`).
- Maaş tutarı: **10** (`62000` … `141000`).
- Pozisyon: **16**. Departman: **15**. İsim havuzu: **24**.
- EMP-ID: 930 farklı (bunlar ezberlenmez, yalnız referanslanır — sorun değil).

Model "30 Eylül 2026 başlangıçlı 4 günlük" → `2026-09-30`/`2026-10-03`
eşlemesini **ezberler**; havuz dışı ("14 Mart 2027", "önümüzdeki cuma") bir ifade
gelirse çözüm güvenilmez. `docs/ANALYSIS.md` de "5000+ için havuzları genişletmek
gerekir" diyor.

**Öneri:** Eğitim öncesi tarih/dönem/tutar/pozisyon/departman yüzey havuzlarını
en az 3–5×'e çıkar; özellikle tarih ifade çeşitliliği (nispi tarihler, farklı
ay/format kombinasyonları).

### 6.3 [S-3] `direct` / `cannot_answer` cevap çeşitliliği düşük

| decision | benzersiz son yanıt / toplam | oran |
|---|---:|---:|
| `tool_call` | 845 / 900 | 0.94 |
| `cannot_answer` | 343 / 600 | 0.57 |
| `request_for_info` | 336 / 750 | 0.45 |
| **`direct`** | **133 / 750** | **0.18** |

`direct`'te bazı cevaplar 11–14× tekrar ediyor (ör. evlilik izni tanımı 14×,
mesai→izin dönüşümü 13×). **Kullanıcı turu** register ile çeşitleniyor ama
**assistant cevabı** 2–3 sabit paragraftan seçiliyor. Routing için sorun değil;
model doğal açıklama üretecekse kalıp ezberi riski var.

**Öneri:** Her `DIRECT_INTENTS` / `MT_DIRECT_SPECS` girdisine 3–5 paraphrase daha
ekle; `REFUSAL_CORE` havuzlarını genişlet.

### 6.4 [S-4] `val` gerçek hold-out değil

`val`, `train` ile **aynı şablon havuzundan** üretiliyor, yalnız intent-stratifiye
ayrılıyor. İmza düzeyinde sızıntı yok ama şablon/slot düzeyinde aynı dağılım.
**Genelleme ölçmez.** When2Call'ın kendi zorlu değerlendirme setine eşdeğer bir
hold-out yok.

**Öneri:** Ayrı bir `hard_eval.jsonl`: farklı seed + havuz-dışı yüzey ifadeleri +
elle yazılmış 50–100 zor/kenar vaka (belirsiz niyet, kısmi bilgi, çeldirici tool'un
cazip olduğu durumlar, register karışımı).

### 6.5 Register üretimi mekanik

`resmi` = önek ("Sayın yetkili, ") + gövde + sonek (" Saygılarımla.").
`yazim_hatali` = yalnız diyakritik düşürme + noktalama atlama (`tr_fold`).
Gerçek yazım hataları (harf transpozisyonu, fonetik yazım, otomatik düzeltme
hataları) modellenmiyor. Model önekten **sahte sınıf sinyali** öğrenebilir
("'Sayın' görürsem resmi").

### 6.6 Multi-tool sığ

63 çoklu-tool örneğinin tamamı: **2 paralel READ, aynı çalışan, 4 sabit
kombinasyon** (maaş+izin, bakiye+geçmiş, bilgi+yönetici, bordro+mesai). Sıralı
bağımlılık yok, 2'den fazla tool yok, WRITE+READ karışımı yok.

### 6.7 `kisa` register az temsil (%4.8)

Gerçek kullanıcılar sık sık 2–4 kelimelik mesaj yazar; sette 144 örnek. Kısa
mesajda niyet + parametre çıkarımı en zor senaryo olduğu için bu pay düşük.

### 6.8 `cok_zor` hedefin altında (%12.9 vs ~%15)

`n_multi_step` payı artırılarak yükseltilebilir; şu an kalite/çeşitlilik
dengesinde tutulmuş.

### 6.9 Opsiyonel parametre öğretilmiyor

`aciklama`, `gecerlilik_tarihi` hiç doldurulmuyor. Kullanıcı "gerekçem taşınma"
dese bile `create_izin_talebi` çağrısında `aciklama` boş kalır. Model opsiyonel
alan doldurmayı hiç görmez.

### 6.10 15 `request_for_info` sonu "?" içermiyor

Hepsi geçerli koşullu istek ("Personel kimliğinizi iletirseniz maaş bilginizi
kontrol ederim."). Anlamsal sorun değil; istenirse `MISSING_PARAM_SPECS`'teki
`ask` şablonlarında soru işareti garanti edilebilir.

### 6.11 Ölçek

3000 örnek + 122 intent + dar havuzlar, LoRA için orta-küçük. Ezber riski;
2–3 epoch'tan fazlası ve düşük LR gerekli. ANALYSIS ~5000 tavan diyor (havuz
genişletmeden).

---

## 7. `scripts/` — üretim hattı

### 7.1 `generate_dataset.py` (3286 satır) — bölüm bölüm

| Bölüm | Satır | İçerik | Değerlendirme |
|---|---|---|---|
| 0. Yapılandırma | 52–81 | `DEFAULT_N=3000`, `SEED=20260827`, `TODAY=2026-08-27`, `TARGET_MIX` 30/25/25/20 | Net, tek yerden. |
| 1. Tool envanteri | 84–405 | 22 tool + `CONFIRMATION_REQUIRED` (7) + `WRITE_TOOLS` + `CONFUSABLE` haritası | Politika (onay) şemadan ayrı tutulmuş — temiz. |
| 2. Slot havuzları | 408–536 | 15 departman + 40 yüzey, 16 pozisyon, izin-tipi yüzeyleri, 21 dönem, **18 tarih aralığı**, 12 ay aralığı, 10 belirsiz tarih, 10 talep-ID, 24 isim, `emp_ref_forms` (8 biçim) | **Havuzlar dar (S-2).** |
| 3. Yardımcılar | 539–697 | `tr_fold`, `norm_sig` (yakın-kopya imzası), `resolve_relative_period`, `build_tools_list` (hedef + komşu + rastgele, karıştır), `style_user_text` (register) | `norm_sig` rakam+ID siler → "sadece EMP değiştir" klonları engellenir. |
| 4. DIRECT intent havuzu | 800–1536 | 53 tanım/politika/süreç/meta intent'i, her biri 3–8 soru + 2–3 cevap | Sorular çeşitli, **cevaplar dar (S-3)**. |
| 5. CANNOT_ANSWER havuzu | 1539–1772 | 42 intent, `pool` etiketli (future/privacy/plain/career/financial/unsupported) | 7 domaine yayılı; §17 kapatılmış. |
| 6. Slot üreticileri | 1776–1883 | `make_slot_funcs` — her slot tipi için (yüzey, kanonik) çifti döner | Yüzey↔kanonik ayrımı halüsinasyon-izlemenin temeli. |
| 7. READ şablonları | 1886–2110 | 15 intent × 6–14 şablon; `argmap` kanonik değere işaret eder | `get_leave_balance` 14 şablon — iyi. |
| 8. MISSING_PARAM | 2113–2316 | 24 aile; `missing` listesi + hedefli `ask` şablonları | Eksik-param mantığı `test_missing_parameter_logic.py` ile doğrulanıyor. |
| 9. WRITE akışları | 2319–2424 | 7 aile; `user` / `ctx` / `argkeys` (`@ckind`, `@bkind`, `@amount` dinamik) | `ctx` onay özetinin somutluğunu sağlıyor. |
| 10. Multi-turn / multi-intent / zincir | 2427–2606 | `MT_INFO` (5), `MULTI_INTENT` (4), `MULTI_STEP` (3 zincir spec), `MT_DIRECT` (8), `MT_CANNOT` (push/hold havuzları) | Zincir mekaniği §5.6'da açıklandı. |
| 11. Üretim motoru | 2623–3092 | `Gen` sınıfı: 11 üretici (`gen_read`, `gen_confirmed`, `gen_mt_info`, `gen_multi_intent`, `gen_multi_step`, `gen_missing`, `gen_confirm_ask`, `gen_mt_direct`, `gen_direct`, `gen_mt_cannot`, `gen_cannot`) + sayaçlar + `add()` (imza-dedup) | `add()` her eklemede `norm_sig` çakışmasını atlıyor (`skipped` sayacı). |
| 12. Çalıştırma/yazma | 3094–3287 | `stratified_split` (intent bazında, <8 → k=0), `write_jsonl` (`newline="\n"`), `build_report`, `main` (alt kırılım oranları) | Rapor yalnız kanonik `data/`'ya yazılırken `docs/`'a — test koşuları repo raporunu bozmuyor. |

**Alt kırılım oranları (`main`, `n_tool=900` üzerinden):**
`n_multi_intent` = %7 → 63 · `n_mt_info` = %9 → 81 · `n_confirmed` = %16 → 144 ·
`n_multi_step` = %11 → 99 · `n_read` = kalan → 513.
`request_for_info`: `n_confirm_ask` = %33 → 248 · `n_missing` = kalan → 502.
`direct`/`cannot`: çok-turlu pay %12 → 90 / 72.

### 7.2 `validate_dataset.py` (500 satır) — bağımsız QC kapısı

- `generate_dataset`'i import edip **ters yüzey haritaları** kurar
  (`DATE_SURFACES`, `PERIOD_SURFACES`, `IZIN_CANON_SURF`, `DEPT_CANON_SURF`…).
- `check_record`: şema + rol + karar tutarlılığı + `trace_arg` (halüsinasyon).
- `trace_arg` mantığı: EMP → çıplak sayı kabul; LV → birebir; tarih/dönem →
  yüzey haritası + nispi ifade; enum → yüzey haritası (uymadıysa **uyarı**, hata
  değil); `number` → birebir (hata); departman → yüzey haritası (uymadıysa uyarı);
  serbest metin → gevşek altdizi (uymadıysa uyarı).
- `check_distribution`: ±3.5 puan; `cannot_answer` domain yayılımı; zincir = 6 tur.
- `check_diversity`: birebir konuşma tekrarı (hata); intent-içi yüzey <%60 (uyarı);
  aşırı sık ilk-4-kelime öneki (uyarı, stil öneklerini hariç tutarak).
- **Çalıştırıldı: 0 hata, 0 uyarı, exit 0.**

> **Not:** Validator üretici ile yüzey haritalarını paylaştığı için "halüsinasyon
> yok" ölçümü kısmen içsel tutarlılık ölçümüdür (§5.3 nüansı).

### 7.3 `make_preview.py` (344 satır)

`data/` → `preview/` okunur türevler (`DATASET_PREVIEW.md`, `index.md`,
`samples/<decision>.sample.json`). Deterministik seçim; **kanonik veriyi
değiştirmez** (kullanıcının "içeriği değiştirme, düzenle" ilkesiyle uyumlu).

### 7.4 `build_training_variants.py` (102 satır)

`data/*.jsonl` → `data/variants/*_chatml_system.jsonl`: araç tanımlarını bir
`system` turuna taşır (`--style tr` veya `hermes`). `messages` metinleri birebir.
Gitignore'da. **Qwen 2.5 için gereksiz** (dosya kendi de söylüyor).

---

## 8. `tests/` — 25 dosya / **321 test, tamamı geçti** (13.6 sn)

| Küme | Dosyalar | Ne doğrular |
|---|---|---|
| Çekirdek 1–10 | jsonl_structure, tool_schema, tool_calls, no_hallucination, decision_semantics, conversation_flow, distribution, diversity_and_leakage, privacy_and_safety, generator_reproducibility | Yapısal + anlamsal temel. |
| İleri 11–20 | statistical_balance (ki-kare), lexical_diversity (Guiraud TTR≥5, ilk-kelime entropisi≥4.5 bit), tool_selection_discriminability, parameter_grounding_precision, missing_parameter_logic, write_safety_state_machine, turn_coherence, encoding_and_serialization, curriculum_and_difficulty, semantic_intent_consistency (Jaccard) | İstatistiksel + bilgi-kuramsal + biçimsel-otomat. |
| Sert 21–25 | decision_oracle (etiketi When2Call'dan yeniden türet, 3000 örnek), adversarial_mutation (14 kusur enjekte → validator hepsini yakalamalı), temporal_consistency (takvim + iş mantığı), chat_template_rendering (ChatML güvenlik), negative_space_coverage (her tool/havuz/ızgara dolu) | En zorlu denetimler. |

**Öne çıkanlar:**
- `test_decision_oracle.py` — etiketin **kendisini** bağımsız bir orakelle
  yeniden türetip 3000 örnekte tutarlılık arıyor. Bu, etiket gürültüsüne karşı
  en güçlü kontrol.
- `test_adversarial_mutation.py` — güvenlik ağının boş olmadığını kanıtlıyor:
  14 kusur türü enjekte, `validate_dataset.check_record` hepsini yakalıyor,
  mutasyonsuz 3000'de yanlış pozitif yok.
- `test_negative_space_coverage.py` — 22 tool'un tamamı çağrılmış; tüm slot/ret
  havuzları örneklenmiş; register×decision (24 hücre) ve domain×difficulty
  ızgaraları dolu; ≥10 train örnekli her intent val'da.

Bağımlılık: `pytest` yalnız test bağımlılığı (`requirements-test.txt`).
9 test dosyası üreticiye/validator'a bilerek bağlı (yüzey→kanonik çözümleme için);
gerisi yalnız `data/` okur.

---

## 9. `docs/` ve `preview/`

- **`docs/ANALYSIS.md`** — elle yazılmış yeterlilik analizi. İlk değerlendirmedeki
  6 eksikten 5'inin kapatıldığını, 1'inin (system-turlu şablon) opsiyonel araçla
  karşılandığını belgeliyor. Kalan sınırlamaları dürüstçe listeliyor (tool-cevabı
  dallanması yok, `cok_zor` ~%13, 5000+ ölçek, ayrı eval seti yok). **Bu raporun
  bulgularıyla tutarlı.**
- **`docs/generation_report.md`** / **`validation_report.md`** — üretici/validator
  çıktıları; güncel ve doğru (yeniden üretilerek teyit edildi).
- **`preview/DATASET_PREVIEW.md`** (1615 satır) — karar sınıfına göre gruplu
  sohbet dökümleri; her sınıftan 22 örnek. `samples/*.json` — girintili tam
  kayıtlar (tools + messages). Hepsi `make_preview.py` üretimi, salt-okunur.
- **`README.md`** — depo düzeni, 4 karar tablosu, format açıklaması, kullanım,
  tasarım notları. Doğru ve yeterli.
- **`.gitignore`** — `__pycache__/`, `*.pyc`, `.pytest_cache/`, `data/variants/`.

---

## 10. Öneriler — öncelik sırasıyla

**Eğitimden önce (yüksek etki):**

1. **Tool-sonucu davranışına karar ver (S-1).** Ya base modelin özetleme
   yeteneğine güven + değerlendirmede ayrıca test et, ya da `tool` rollü 300–800
   örneklik bir devam alt-seti ekle.
2. **Slot havuzlarını 3–5× genişlet (S-2).** Öncelik: tarih/dönem yüzey
   ifadeleri (nispi tarihler, format çeşitliliği), sonra tutar/pozisyon/departman.
3. **Ayrı `hard_eval` seti oluştur (S-4).** Farklı seed + havuz-dışı yüzeyler +
   50–100 elle yazılmış kenar vaka. Değerlendirmeyi bunun üzerinde yap.

**Eğitim kurulumu:**

4. **Loss masking** — yalnız `assistant` turlarında kayıp; tüm ara assistant
   turları (onay-iste, param-iste) hedef. `apply_chat_template(..., tools=tools)`
   kullan; `build_training_variants` gerekmez.
5. **LoRA hiperparam (öneri):** `r=16–32`, `alpha=2×r`, dropout `0.05`,
   hedef modüller `q,k,v,o,gate,up,down`, **2–3 epoch**, LR `1–2e-4` kosinüs,
   3000 örnekte overfit'e karşı early-stopping (hard_eval kaybına göre).
6. **Değerlendirme metrikleri:** karar başına doğruluk (4×4 kafa karışıklığı
   matrisi), tool-seçim F1 (çeldiriciler arasında), argüman-tam-eşleşme oranı,
   halüsinasyon oranı (hard_eval'de), WRITE-onaysız-çağrı sayısı (0 olmalı).

**Kalite (orta etki):**

7. `direct` / `cannot_answer` cevap havuzlarına paraphrase ekle (S-3).
8. `kisa` register payını ~%10'a çıkar; gerçek yazım-hatası çeşitleri ekle.
9. Opsiyonel parametre (`aciklama`, `gecerlilik_tarihi`) doldurma örnekleri ekle.
10. Multi-tool'u derinleştir (3+ tool, sıralı bağımlılık, WRITE+READ karışımı) —
    yalnız hedef senaryoda bu davranış isteniyorsa.

---

## 11. Ölçüm günlüğü (bu raporun dayanağı)

Bu oturumda fiilen çalıştırılan ve çıktısı doğrulanan komutlar:

```bash
python scripts/validate_dataset.py          # -> 0 HATA / 0 UYARI, exit 0
python -m pytest tests/ -q                   # -> 321 passed in 13.58s
python scripts/generate_dataset.py --out-dir <tmp>   # -> 5 dosya SHA-256 = kanonik (byte-aynı)
```

Ek olarak `data/*.jsonl` + `*.meta.jsonl` Python ile yeniden yüklenip:
şema/enum/JSON geçerliliği, rol yapısı, `tool_call` blok biçimi, argüman
grounding (240 birebir-eşleşmeyen değerin kategori dökümü), assistant düz-metin
halüsinasyon taraması, train/val sızıntı, imza çakışması, slot havuzu kullanımı,
cevap çeşitliliği, tur/register/domain/zorluk dağılımları ve karar×domain
ızgarası bağımsız olarak hesaplandı. Rakamlar §2–§6'da yerinde.

**Kodlama:** `train.jsonl` / `val.jsonl` / `tools.json` → BOM yok, CRLF 0,
dosya `\n` ile biter, geçerli UTF-8.

---

*Rapor sonu.*
