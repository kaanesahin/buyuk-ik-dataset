# Büyük İK — Dataset Revizyon Raporu (v1 → v2)

> `POLICY_UYGUNLUK_RAPORU.md` "NEEDS DATASET REVISION" kararı üzerine yapılan
> kapsamlı yeniden yapılandırma. v1 (22 tool, per-tool elle şablon, 3000 örnek)
> → v2 (105 tool / 13 domain, şema-güdümlü üretim, holdout tool bölmesi,
> tool-sonucu turu, 15.000 train + 2.000 val + ~1.000 hard_eval).
> v1 pipeline `legacy/` altında korunuyor.

Format: **PROBLEM → YAPILAN DEĞİŞİKLİK → UYGULAMA → ÖLÇÜLEN SONUÇ**
Ölçümler: `python scripts/metrics.py` · `python scripts/validate_dataset.py` ·
`pytest tests/` (33 test).

---

## Kritik eksiklikler (K-1 … K-10)

### K-1 — Yüzey kelimesi → tool adı korelasyonu (%95–100)

- **PROBLEM:** v1'de `get_mesai_bilgisi` %100, `get_maas_bilgisi` %97 vb. — model
  açıklamayı okumadan kelime eşleştiriyordu.
- **DEĞİŞİKLİK:** (a) Her tool için "ayırt edici yüzey sözcüğü" (`disc_kw`) katalogdan
  otomatik hesaplanıyor (diğer tool'ların nesne+syn metninde en az geçen 1–2 token).
  (b) 105 tool'un tamamının `syn` (dolaylı ifade) havuzu, tool'un `disc_kw`'sini
  İÇERMEYECEK şekilde elden geçirildi (`_SYN_OVERRIDE`). (c) Kullanıcı ifadelerinin
  ~%72'si "oblique" frame'lerden (baş sözcüksüz betimleme) üretiliyor.
- **UYGULAMA:** `catalog/catalog.py` (`_compute_disc_kw`, `_SYN_OVERRIDE`),
  `scripts/gen/frames.py` (`USER_FRAMES_OBLIQUE`, `USER_FRAMES_WRITE`),
  `scripts/gen/scenarios.py` (`_fill_user` oblique_p=0.72).
- **ÖLÇÜLEN SONUÇ (dürüst, iki ölçüt):**
  - **nesnenin ana adı** kullanıcı turunda geçiyor mu: **~%53** (v1'de de benzerdi;
    asıl patoloji bu değildi) — örneklerin ~yarısında model açıklamayı / aday
    listeyi okumak zorunda.
  - **en nadir ayırt edici token** (`disc_kw`; alt sınır): **~%33** — v1'de bu
    **%95–100** idi; patolojik "fiil+nesne → tekil tool adı" eşlemesi kırıldı.
  - Önceki raporda tek başına verilen **"%35"** rakamı `disc_kw` ölçütüdür;
    yalnız onu göstermek iyimserdi. `metrics.py` / `validate_dataset.py` artık
    **her iki** sayıyı birden raporlar. `test_keyword_to_toolname_shortcut_is_low`
    (disc_kw < %55) geçiyor.

### K-2 — Sabit ve küçük tool evreni (22, holdout 0)

- **PROBLEM:** Model 22 ad↔kalıbı ezberleyebilirdi; yeni-tool genellemesi ölçülemiyordu.
- **DEĞİŞİKLİK:** Katalog **105 tool / 13 domain**'e çıkarıldı (İK, bordro, puantaj,
  finans, CRM, BT destek, lojistik, envanter, satış, takvim, belge, raporlama,
  müşteri destek). Domain-stratifiye deterministik hash ile **train 75 / val 15 /
  test 15** bölündü. val/test tool'ları eğitimde **hedef olarak asla** kullanılmıyor.
- **UYGULAMA:** `catalog/catalog.py` (`assign_splits`), `generate_dataset.py`
  (`build_val` — val_unseen_tool), `build_hard_eval.py` (test_tools).
- **ÖLÇÜLEN SONUÇ:** train hedefinde val/test tool'u **0** (validator + 3 test).
  val_unseen_tool hedefleri 15/15 val-split; hard_eval P1/P9 hedefleri 15/15 test-split.
  Holdout tool'lar çeldirici olarak görülüyor (%80+), yani şeması öğreniliyor,
  altın cevabı değil.

### K-3 — Küçük aday tool listesi (hep 4–9)

- **PROBLEM:** 100-tool / retrieval'lı gerçek senaryo hiç görülmüyordu.
- **DEĞİŞİKLİK:** Aday liste boyutu değişken kova dağılımı:
  %28 → 5–12, %56 → 14–30, %16 → 36–58 tool. Hedefin liste-içi konumu uniform rastgele.
- **UYGULAMA:** `scripts/gen/catalog_index.py` (`Index.candidate_list`).
- **ÖLÇÜLEN SONUÇ:** ≤12 %33 · 13–34 %52 · 35–58 %15 (medyan 19, p90 43, maks 58).
  Hedef konumu ort. **0.50** (ele vermiyor). `test_candidate_size_buckets`,
  `test_candidate_list_target_present_and_position_uniform` geçiyor.

### K-4 — Per-tool elle yazılmış şablonlar

- **PROBLEM:** `READ_SPECS` / `MISSING_PARAM_SPECS` / `WRITE_SPECS` / elle `CONFUSABLE`
  — 100 tool = 78 tool için elle içerik yazmak.
- **DEĞİŞİKLİK:** Üretim tamamen **şema-güdümlü**:
  - kullanıcı ifadesi = tool-agnostik frame havuzu × tool metadata (`obj`, `syn`,
    `verbs`, parametre `human`) — per-tool cümle şablonu YOK;
  - çeldiriciler = elle harita yerine **şema benzerliği** (aynı domain / kategori /
    parametre-kind imzası / ad-nesne sözcük örtüşmesi);
  - eksik-parametre mantığı = `required` listesinden otomatik.
- **UYGULAMA:** `scripts/gen/{frames,synth,scenarios,catalog_index}.py`.
- **ÖLÇÜLEN SONUÇ:** `scenarios.py` hiçbir tool adına özel cümle/cevap kodu içermiyor
  (yalnız 7 satırlık `CHAINS` ilişki tablosu — sub-linear).
  `test_catalog_scales_without_new_templates` geçiyor. Yeni tool eklemek = katalogda
  bir `T(...)` satırı.

### K-5 — Kapalı, küçük slot havuzları

- **PROBLEM:** 18 tarih yüzeyi, 10 talep-ID, ~10 tutar → model yüzey ezberler.
- **DEĞİŞİKLİK:** Tüm parametre değerleri **üretim anında prosedürel sentezlenir**
  (`synth.py`): takvim-geçerli rastgele tarihler + çok sayıda yüzey biçimi
  (ISO / DD.MM.YYYY / "14 Mart 2027" / "yarın" / "önümüzdeki salı" / "3 gün sonra");
  ID biçimleri ("EMP-1234" / "emp_1234" / "#1234" / "1234 numaralı personel");
  geniş tutar aralığı + "76 bin TL" / "₺76.000"; dönem/yıl göreli ifadeler.
  Kanonik ↔ yüzey çözümü kural tabanlı (`resolve.py`), liste tabanlı değil; üretici
  ve validator **aynı** çözümleyiciyi paylaşır.
- **UYGULAMA:** `scripts/gen/{synth,resolve}.py`.
- **ÖLÇÜLEN SONUÇ:** benzersiz ilk-kullanıcı-turu **%96.3**. Halüsinasyon 0 —
  "14 Mart 2027", "önümüzdeki çeyreğin başı" gibi havuz-dışı ifadeler de kurala göre
  çözülüyor (`hard_eval` P3/P5 havuz-dışı yüzeyler).

### K-6 — Tool sonucu turu yok

- **PROBLEM:** Model tool çağırıp duruyordu; sonucu yorumlayıp cevap üretmeyi hiç
  görmüyordu (`tool` rolü 0).
- **DEĞİŞİKLİK:** `tool_call` örneklerinin **~%47'sinde** akış uzatıldı:
  `user → assistant(tool_call) → tool(JSON sonuç) → assistant(Türkçe cevap)`.
  Sonuç modları: `ok` / `empty` / `error` / `partial` — model "sonuç yok / hata"
  durumunu da ele alıyor. Sıralı çoklu-tool'da sonuç → ikinci tool_call parametresi.
- **UYGULAMA:** `scripts/gen/scenarios.py` (`synth_result`, `final_answer`,
  `gen_multi_sequential`).
- **ÖLÇÜLEN SONUÇ:** tool-sonucu turu içeren örnek **%26** (tüm veri) /
  **%46** (tool_call içinde). 4 mod: ok / empty / error / partial.
  Nihai yanıtta kaynak-dışı sayı **0** (`test_final_answer_after_tool_result_grounded`).

### K-7 — Zayıf hard-negative'ler

- **PROBLEM:** Çeldiriciler yalnız aynı-domain; çelişkili/yanıltıcı senaryo yok.
- **DEĞİŞİKLİK:** 4 hard-negative senaryosu (toplam **%9.5**):
  - **A** aynı keyword farklı tool (kardeşler aday listede zorunlu);
  - **D** kullanıcı yanlış tool adını söylüyor, model doğruyu seçmeli;
  - **E** çelişkili parametre (ters tarih aralığı / iki farklı kimlik) → netleştirme;
  - **F** doğru tool aday listede yok → `cannot_answer`.
- **UYGULAMA:** `scripts/gen/scenarios.py` (`gen_hn_*`).
- **ÖLÇÜLEN SONUÇ:** A 450 · F 375 · E 300 · D 300 (n=15000). `test_hard_negatives_present`
  geçiyor. hard_eval'de ayrıca P4 (110) ve P8 (51) probe'ları (rapor sonrası büyütüldü).

### K-8 — Sığ multi-tool (4 sabit kombinasyon)

- **PROBLEM:** Sıralı bağımlılık / görev ayrıştırma öğretilmiyordu.
- **DEĞİŞİKLİK:** İki senaryo, prosedürel:
  - `multi_parallel` — aynı varlık, 2 READ, tek asistan turu (şema-benzerliğinden
    türetilen çiftler);
  - `multi_sequential` — resolver tool sonucundaki id → ikinci tool'un parametresi
    (7 cross/intra-domain ilişki: isim→hesap, fatura→tedarikçi, sipariş→hesap...).
- **UYGULAMA:** `scripts/gen/scenarios.py`, `generate_dataset.py` (`run_multi_*`).
- **ÖLÇÜLEN SONUÇ:** paralel ~600 · sıralı ~450. `test_multi_tool_*` geçiyor.

### K-9 — `direct` cevap ezberi (133 benzersiz / 750)

- **PROBLEM:** Bazı cevaplar 14× tekrar.
- **DEĞİŞİKLİK:** Domain-genel 16 tanım/politika intent'i, her biri 2 çekirdek cevap
  × 5 çerçeveleme sarmalı; greeting/thanks için yapay "takip sorusu" kaldırıldı.
- **UYGULAMA:** `scripts/gen/catalog_index.py` (`DIRECT_POOL`),
  `scripts/gen/frames.py` (`DIRECT_WRAP`).
- **ÖLÇÜLEN SONUÇ:** en sık tekrar eden tek `direct` cevap payı **< %6**
  (`test_direct_answer_not_overly_repeated`).

### K-10 — `val` gerçek holdout değil

- **PROBLEM:** Aynı şablon+slot havuzundan; genelleme ölçmüyordu.
- **DEĞİŞİKLİK:** Üç ayrı değerlendirme yüzeyi:
  - `val_seen_tool` (1000) — train tool'ları, alışılmadık yüzey;
  - `val_unseen_tool` (1000) — **eğitimde hiç görülmemiş** val tool'ları hedef;
  - `hard_eval` (~1000; rapor sonrası ~330'dan büyütüldü) — test tool'ları + 9
    policy-probe (P1…P9), elle yazılmış havuz-dışı sarmallar.
- **UYGULAMA:** `generate_dataset.py` (`build_val`), `build_hard_eval.py`.
- **ÖLÇÜLEN SONUÇ:** train↔val imza kesişimi **1**. val_unseen ve hard_eval P1/P9
  hedefleri %100 holdout-split. Metrik iskeleti `docs/DATASET_STATISTICS.md` sonunda.

---

## Değişiklik planı (D-1 … D-10) — durum

| # | Plan | Durum | Kanıt |
|---|---|:--:|---|
| D-1 | Tool evrenini genişlet + holdout | ✅ | 105 tool, 75/15/15, sızıntı 0 |
| D-2 | Şema-güdümlü üretim | ✅ | per-tool şablon yok; `test_catalog_scales…` |
| D-3 | Değişken/büyük aday liste | ✅ | kovalar %34/%51/%15, maks 58 |
| D-4 | Yüzey–ad kısayolunu kır | ✅ | disc_kw korelasyonu ~%33 · nesne ana-adı ~%53 (iki ölçüt de raporlanır) |
| D-5 | Tool sonucu turu | ✅ | %46 (tool_call içinde), 4 mod |
| D-6 | Slot havuzlarını prosedürelleştir | ✅ | `synth.py` + `resolve.py` |
| D-7 | Hard-negative + belirsizlik | ✅ | %9.5, 4 tür + P4/P8 |
| D-8 | Multi-tool'u derinleştir | ✅ | sıralı sonuç→param 450 |
| D-9 | Ölçek + holdout eval | ✅ | 15k / 2k / ~330; metrik iskeleti |
| D-10 | Opsiyonel parametre davranışı | ⚠️→✅ | %7.2 örnekte opsiyonel dolduruluyor (Durum A); "verilmezse uydurma" (Durum B) tüm read'lerde; dolaylı verme (Durum C) enum yüzeyleriyle |

---

## Rapor sonrası düzeltmeler (R-1 … R-5)

> Bağımsız inceleme kararı **NEEDS MINOR FIX**. Aşağıdaki beş düzeltme yapıldı;
> yapısal karar (READY) değişmedi.

### R-1 — Tool-sonucu sentez hatası (orta önem, kritik tool'larda yoğun)

- **PROBLEM:** `_synth_result_value` fonksiyonunda `kind == "name"` / `kind == "title"`
  için fallback yoktu → bu alanlar `rng.randint(1, 90)` ile doluyordu. Sonuç:
  `hr_get_employee_profile.full_name` "ad 42", `payroll_get_salary.currency`
  "para birimi 38" (üstelik "currency" çevrilmemiş), `hr_get_org_unit.manager`
  "yönetici 24" — İK'nın en çok kullanılan 3 tool'unun result-özet turlarının
  ~%67–77'si bozuktu (~99 asistan NL yanıtı + ~208 tool JSON).
- **DEĞİŞİKLİK:** `_synth_result_value` yeniden yazıldı: `tool` bağlamı alıyor;
  isim/şirket/para-birimi/varlık-türü/rapor-özeti/belge-başlığı anahtarları için
  gerçekçi havuzlar; `kind == "name"/"title"` için tam fallback → **hiçbir alan
  artık `randint`'e düşmüyor**. `_RESULT_LABELS`'e eksik Türkçe etiketler
  (para birimi, yönetici, zimmetli, tür, bağlı hesap, özet, kategori…).
  Ayrıca `_harmonize_result`: net ≤ brüt, kesinti = brüt − net, kalan = ayrılan −
  harcanan (iç tutarlılık — K-minor).
- **UYGULAMA:** `scripts/gen/scenarios.py`.
- **ÖLÇÜLEN SONUÇ:** train'de isim/başlık alanında sayı-değer **0** (önceden ~99).
  validator 0 hata; halüsinasyon 0 korundu.

### R-2 — hard_eval istatistiksel olarak zayıftı

- **PROBLEM:** 338 örnek; genelleme kapısının kontrol kolu P5 yalnız 35 örnek /
  23 tool (tool başına ~2) — "policy taşındı mı" farkı güvenilir ölçülemezdi.
  P4 (21) ve P8 (15) de güçsüzdü.
- **DEĞİŞİKLİK:** `build_hard_eval.py` sayıları ~3× büyütüldü: **338 → ~1000**.
  P5 **35 → 200**, P4 **21 → 110**, P8 **15 → 51**, P1 **95 → 189**. P4 artık
  test + train kw-kardeşli tool'lardan çekiliyor.
- **ÖLÇÜLEN SONUÇ:** `hard_eval` ~1000 örnek, sızıntı 0, validator 0 hata.

### R-3 — Kayıp yoğunluğu ~%1.8 (yapısal; kusur değil ama önemli)

- **PROBLEM:** Her dizinin ~%98'i eğitilmeyen aday-şema bağlamı → gradyan-sinyali
  token başına ~55× hesap yükü.
- **DEĞİŞİKLİK:** (a) `scripts/build_training_variants.py --max-candidates N` —
  küçük-aday-liste **curriculum** kopyası (altın cevap değişmez; bağlam ~2.2×
  kısalır → kayıp yoğunluğu ~%1.8 → ~%4). (b) `docs/TRAINING_EFFICIENCY.md` —
  sequence packing config'leri (TRL / LLaMA-Factory / axolotl) + 2-aşamalı
  ısıtma reçetesi.
- **ÖLÇÜLEN SONUÇ:** `cand10` varyantı: yoğunluk %4.0, ort. 3.710 krk/kayıt
  (kanonik 8.070). Kanonik veri değişmedi.

### R-4 — Anti-kısayol metriği tek sayıyla iyimser raporlanmıştı

- **PROBLEM:** Rapor "%35" diyordu; bu `disc_kw` (tool'un en nadir token'ı)
  ölçütü — kayırmalı. Nesnenin ana adıyla ~%53.
- **DEĞİŞİKLİK:** `metrics.py` ve `validate_dataset.py` artık **üç ölçütü** birden
  raporlar: nesne ana-adı (dürüst üst sınır) / tüm yüzey sözlüğü / en nadir token
  (alt sınır). `USER_FRAMES_WRITE`'a `{syn}` tabanlı (baş-sözcüksüz) kalıplar
  eklendi (WRITE tarafında nesne-adı korelasyonunu düşürmek için).
- **ÖLÇÜLEN SONUÇ:** nesne ana-adı ~%53 · disc_kw ~%33. Dokümanlar düzeltildi.

### R-5 — Birleşen-aksan kalıntısı + agresif typo register (K-minor)

- **PROBLEM:** Python'da `"İ".lower()` → `"i̇"` (i + U+0307). ~154 kayıtta
  (özellikle formal/chat register) sızmıştı. Ayrıca typo register (%15) neredeyse
  okunmaz metinler üretiyordu.
- **DEĞİŞİKLİK:** `frames.py`'ye `tr_lower` / `tr_upper` / `denorm`;
  `Record.__post_init__` tek chokepoint'ten U+0307 temizler. `synth.gen_org_name`
  / `gen_email` Türkçe-güvenli küçültme. typo register %15 → %12, harf-yer-değiştirme
  olasılığı 0.4 → 0.22, soru işareti korunuyor.
- **ÖLÇÜLEN SONUÇ:** train'de U+0307 **0** (önceden ~154). typo register %10.4.
  `test_register_diversity` geçiyor.

---

## Korunan sağlam mimari (bozulmadı)

`{tools, messages}` biçimi · Qwen mesaj yapısı · `<tool_call>\n{tek-satır JSON}\n</tool_call>` ·
4-karar çerçevesi (`direct` / `tool_call` / `request_for_info` / `cannot_answer`) ·
onaydan önce WRITE yapmama · parametre uydurmama ilkesi ve bağımsız izleyici ·
UTF-8 / LF / BOM-suz / deterministik (aynı seed → byte-aynı) · validator + test
felsefesi (bağımsız QC kapısı + isimli pytest'ler).

---

## Doğrulama özeti (v2)

```
python scripts/validate_dataset.py   -> HATA 0 / UYARI 0  (train 15000 + val 2000 + hard_eval ~1000)
pytest tests/                        -> 33 passed
python scripts/generate_dataset.py   -> aynı seed, byte-aynı çıktı (test_regen_byte_identical)
```
