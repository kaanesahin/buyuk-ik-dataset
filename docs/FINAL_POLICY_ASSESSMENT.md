# Büyük İK v2 — Nihai Policy Değerlendirmesi

> Revizyon sonrası (v2) değerlendirme. Öncekiler: `FINETUNE_UYGUNLUK_RAPORU.md`,
> `POLICY_UYGUNLUK_RAPORU.md` (v1 → "NEEDS DATASET REVISION"), `REVISION_REPORT.md`.
> Ölçüm kaynağı: `docs/DATASET_STATISTICS.md`, `scripts/validate_dataset.py` (0 hata),
> `pytest tests/` (33 test).
>
> **Güncelleme:** bağımsız inceleme kararı **NEEDS MINOR FIX** → R-1…R-5 düzeltildi
> (bkz. `REVISION_REPORT.md` "Rapor sonrası düzeltmeler"): tool-sonucu sentez
> hatası, hard_eval 338→~1000, kayıp-yoğunluğu araçları, dürüst anti-kısayol
> raporlaması, aksan/typo temizliği. Yapısal karar (READY) değişmedi.

---

## 1. Değerlendirme tablosu

| Bölüm | Durum | Tespit | Öneri / not |
|---|:--:|---|---|
| **Dataset Formatı** | ✅ Yüksek | `{tools, messages}` Qwen native; `id`/`decision`/`register` yalnız `*.meta.jsonl`'de (eğitim dosyasında yok); `tool` rolü destekli; LF/UTF-8/BOM-suz; `<tool_call>\n{tek-satır JSON}\n</tool_call>`. | `*_train.jsonl` olduğu gibi kullanılabilir; `.meta.jsonl` eğitime yüklenmesin. |
| **Tool Definitions** | ✅ Yüksek | 105 tool / 13 domain; her biri 1 cümle açıklama + parametre açıklamaları + `enum`/`required` + kategori (read/write/action). Şema, üretimin tek kaynağı — tanım kalitesi doğrudan veri kalitesine yansıyor. | Adlandırma İngilizce snake_case + domain öneki (tutarlı). |
| **Tool Selection** | ✅ Yüksek | Yüzey kelimesi → tool korelasyonu **iki ölçüt**: nesnenin ana adı geçiyor mu **~%53** (dürüst; örneklerin ~yarısında açıklama/aday-liste okumak şart) · en nadir ayırt edici token **~%33** (alt sınır; v1'de bu **%95–100** idi — patolojik fiil→tekil-tool eşlemesi kırıldı). Aday liste 5–58 (değişken), hedef konumu ort. 0.50. Çeldiriciler şema-benzerliğinden. Hard-negative A + P4. | Birkaç tool (özellikle WRITE: `create_leave_request`, `create_expense_report`) nesne-adı korelasyonu %80+ — nesnesi doğal olarak anılıyor; `disc_kw` tarafında hepsi düşük. Gerçek genelleme eğitim sonrası P1-vs-P5. |
| **Parameter Extraction** | ✅ Yüksek | Bağımsız izleyici: **0 uydurulmuş argüman** (train + val + hard_eval). Değerler prosedürel sentezleniyor; yüzey → kanonik çözümü kural tabanlı (`resolve.py`), havuz değil → "14 Mart 2027" gibi görülmemiş ifadeler de çözülüyor. Tarih çiftleri tutarlı (start ≤ end). Ortak param kind'leri (id, tarih, dönem, tutar, enum) çok sayıda tool'da. | Opsiyonel parametre: verilince doldurulur (%7.2), verilmeyince uydurulmaz, enum yüzeyiyle dolaylı verilebilir. |
| **General Policy** | ✅ Yüksek | 4-karar (`direct` %11 / `tool_call` %56.5 / `request_for_info` %20 / `cannot_answer` %12.5). Onaydan önce WRITE: **0 ihlal**. Eksik param → soru. "Aday listede tool var ≠ çağır" (tüm `cannot_answer` örneklerinde liste dolu). ≥25 read tool'u hem `tool_call` hem `request_for_info` hedefi (param var/yok kararı ifadeden bağımsız). | `tool_call` payı yüksek — tool-calling policy için savunulabilir; istenirse `direct`/`cannot` payı artırılabilir (`MIX` sözlüğü). |
| **Cross-Tool Generalization** | ✅ Yüksek (yapısal) | Üretim şema-güdümlü — `scenarios.py` hiçbir tool adına özel cümle/cevap kodu içermiyor (yalnız 7-satır `CHAINS` ilişki tablosu). **15 val + 15 test tool'u eğitimde HİÇ hedef değil**; yalnız çeldirici olarak görülüp şeması öğreniliyor. `hard_eval` P1/P2/P3/P5 probe'ları bunu ölçmek için. | **Kesin kanıt eğitim sonrası:** `hard_eval` üzerinde top-1(P5, görülen tool) − top-1(P1, görülmemiş tool) farkı küçükse policy taşınıyor demektir. Bu farkın kabul eşiği ≤ ~10 puan önerilir. |
| **Tool Coverage** | ✅ Yüksek | 75 train tool'unun tamamı hedef (min 103 / medyan 144 / maks 354 örnek); dağılım eğriliği 3.4×. 13 domain'in tamamı. read 66 / write 35 / action 4. | Tool'lar arası ilişki: 7 sıralı-zincir + şema-benzerliği çeldirici grafiği. |
| **Negative / Edge Cases** | ✅ Yüksek | `cannot_answer` %12.5 (7 gerekçe: kapsam-dışı / gelecek / gizlilik / yetki / tavsiye). Hard-negative %9.5: **A** aynı kelime farklı tool, **D** kullanıcı yanlış tool adı söylüyor, **E** çelişkili parametre → netleştirme, **F** doğru tool listede yok → ret. tool-sonucu `empty`/`error`/`partial` modları. | hard_eval'de P4 (110) / P7 (142) / P8 (51) / P9 (74) ayrı probe. |
| **Tool Call Format** | ✅ Yüksek | Katı `<tool_call>` biçimi, tek-satır JSON (streaming-parse güvenli), blok tek başına (önünde metin yok), çoklu çağrı = art arda blok. 0 biçim hatası. | `build_training_variants.py` ile Hermes/TR system-turlu kopya. |
| **Tool Result Handling** | ✅ Yüksek (v1: ❌ yoktu) | tool_call örneklerinin **~%46'sı** `user → tool_call → tool(JSON) → assistant(cevap)` yapısında. 4 sonuç modu. Nihai yanıtta kaynak-dışı sayı **0**. Sıralı çoklu-tool: 1. sonuçtaki id → 2. tool parametresi (450 örnek). **R-1:** result-değer sentezi düzeltildi — isim/başlık/para-birimi alanları artık gerçekçi (önceden İK'nın 3 ana tool'unda ~%70 "ad 42" gibi bozuktu). İç tutarlılık: net ≤ brüt, kalan = ayrılan − harcanan. | Model "sonuç boş/hata" durumunu da öğreniyor. |
| **Fine-Tuning Uyumu** | ✅ Yüksek | 15.000 train (v1: 3.000); kayıp yalnız assistant turlarında; `apply_chat_template(messages, tools=tools)` doğrudan. Tam katalog ~11.000 token; 35–58 tool'lu örneklerde dizi ~6–9k token → `max_seq_len ≥ 16k` önerilir. Determinizm: aynı seed → byte-aynı. **Kayıp yoğunluğu ~%1.8** (dizinin ~%98'i aday-şema bağlamı) → **sequence packing** ya da `--max-candidates` curriculum ısıtması önerilir: `docs/TRAINING_EFFICIENCY.md`. | LoRA: `r=16–32`, 2–3 epoch, lr 1–2e-4, `hard_eval` kaybına göre early-stopping. |
| **100-Tool Ölçeklenebilirliği** | ✅ Yüksek (v1: ❌) | Katalog zaten 105 tool. Yeni tool = `catalog/catalog.py`'de bir `T(...)` satırı → üretici otomatik işler (çeldirici, eksik-param, tool-sonucu, hard-negative dahil). Aday listeler 36–58 tool'u zaten alıştırıyor; tam katalog token bütçesine sığıyor. Üretim maliyeti tool sayısıyla ~sabit. | 200+ tool için token bütçesi (retrieval) ve `MIX` alt-kırılımı gözden geçirilebilir. |

---

## 2. Genel Sonuç

### Soru 1 — Tool-level olarak dataset yeterli mi?

**EVET (Yüksek).** 105 tool'un her biri (train alt kümesinde 75'i) yeterli yoğunlukta
(min 103, medyan 144 örnek), parametreleri kullanıcı metninden uydurma olmadan
çıkarılabiliyor (bağımsız izleyici: 0 halüsinasyon), şeması temiz (0 enum/zorunlu
ihlali), ve her örnekte şema-benzeri çeldiricilerle sunuluyor.

### Soru 2 — Policy-level olarak ~100 tool için genellenebilir mi?

**EVET (yapısal olarak; kesin kanıt eğitim sonrası).** v1'in "22 tool'a özel şablon
kütüphanesi" yapısı ortadan kalktı:

- **Üretim tool'dan değil ŞEMADAN türüyor** — yeni tool eklemek per-tool içerik
  yazmayı gerektirmiyor (`test_catalog_scales_without_new_templates`).
- **Görülmemiş-tool holdout'u var** — 30 tool eğitimde hiç hedef değil; "yeni tool +
  tanıdık mantık" davranışı `hard_eval` P1/P2/P3/P5 ile ölçülebilir (P5 rapor
  sonrası 35→200 → tool başına yeterli örnek).
- **Yüzey-kelime → tool adı kısayolu kırıldı** — patolojik `disc_kw` ölçütü
  %95–100 → ~%33; nesne ana-adı ölçütü ~%53 (örneklerin ~yarısında açıklama şart).
- **Aday liste artık gerçekçi** — ~%15'i 36–58 tool, gerisi 5–34; hedef konumu ele vermiyor.
- **4-karar policy'si tool-agnostik ve iyi temsil edilmiş** — onay-önce-yazma
  0 ihlal, "tool var ≠ çağır", param-var/yok kararı ifadeden bağımsız.
- **Tool sonucu döngüsü tamamlandı** — anla → seç → çağır → **oku → cevapla**.

Yapı, ~100 tool üzerinde genellenebilir bir tool-calling policy öğrenmeye
**hazır ve donanımlı**. Modelin bu policy'yi fiilen taşıyıp taşımadığı ancak
eğitimden sonra `hard_eval` probe'larıyla (özellikle **P1 vs P5 top-1 farkı**)
doğrulanabilir — bu, veri setinin değil eğitimin sınavıdır.

---

## 3. Kritik eksiklik taraması (v2)

| Eski risk | v2 durumu |
|---|---|
| tool'a özel aşırı tekrar | Yok — şema-güdümlü; en sık `direct` cevap < %6 |
| yetersiz cross-tool örnekleri | 30 holdout tool + 9 policy-probe + 7 sıralı zincir |
| dengesiz tool dağılımı | 3.4× eğrilik (v1: benzer); min 103 örnek/tool |
| yetersiz negative examples | `cannot_answer` %12.5 + hard-negative %9.5 |
| aynı intent için az ifade | benzersiz ilk-tur %96.3; oblique frame'ler + 6 register |
| yeni tool'lara genellenemeyen yapı | katalog satırı → otomatik üretim |
| tool eşleştirmesi yapan örnekler | disc_kw korelasyonu ~%33 / nesne ana-adı ~%53 (kısayol kırıldı; iki ölçüt de raporlanır) |
| tool-sonucu değeri bozuk (R-1) | düzeltildi — isim/başlık/para-birimi alanları gerçekçi; isim/başlık alanında sayı-değer 0 |
| hard_eval zayıf (R-2) | 338 → ~1000; P5 35→200, P4 21→110, P8 15→51 |
| kayıp yoğunluğu ~%1.8 (R-3) | curriculum varyantı + packing rehberi (`TRAINING_EFFICIENCY.md`) |

**Kalan izlenecekler (düşük öncelik):**
1. WRITE tool'larında nesne-adı korelasyonu %80+ (`create_leave_request` vb.) —
   `disc_kw` tarafında hepsi düşük; gerçek etki eğitim sonrası P1-vs-P5 ile görülür.
2. `tool_call` payı %56.5 — model tool'a fazla meyilliyse `MIX`'te `direct`/`cannot`
   artırılır (eğitim sonrası ayarlanır).
3. Gerçek genelleme sadece eğitim sonrası P1-vs-P5 ile kanıtlanır (yapısal hazırlık tam).

---

## 4. Son Karar

# → READY

**Gerekçe:** `POLICY_UYGUNLUK_RAPORU.md` §19'daki 13 kabul kriterinin tamamı sağlanıyor
(şema-güdümlü ~100-tool yapısı, keyword kısayolu kırıldı [disc_kw ~%33 / nesne ana-adı ~%53], değişken aday liste,
görülmemiş-tool holdout'u, prosedürel parametre, opsiyonel-parametre davranışı,
hard-negative'ler, çeşitli multi-tool, tool-sonucu turu, gerçek holdout eval,
minimum-manuel-iş ölçeklenme, validator 0 hata, artık NEEDS REVISION/NOT READY değil).

Hem **tool-level** hem **policy-level** yeterli. Tek açık nokta ampiriktir ve veri
setinin kusuru değildir: modelin policy'yi görülmemiş tool'lara fiilen taşıyıp
taşımadığı eğitimden sonra `hard_eval` üzerinde ölçülmelidir. Kabul kapısı:
**top-1(P1, görülmemiş tool) ≥ top-1(P5, görülen tool) − 10 puan** ve
**yetkisiz WRITE = 0**.

---

## 5. Eğitim & değerlendirme akışı

```bash
python scripts/generate_dataset.py       # data/  (15k train + 2k val)
python scripts/build_hard_eval.py        # data/tool_calling_hard_eval.jsonl (~1000)
python scripts/validate_dataset.py       # 0 hata beklenir
pytest tests/                            # 33 test
python scripts/metrics.py                # docs/DATASET_STATISTICS.md
# (opsiyonel) python scripts/build_training_variants.py --style tr        # system-turlu kopya
# (opsiyonel) python scripts/build_training_variants.py --max-candidates 10   # curriculum ısıtma
#             -> bkz. docs/TRAINING_EFFICIENCY.md (kayıp yoğunluğu / packing)

# LoRA eğitimi -> sonra hard_eval üzerinde:
#   4-karar 4x4 matrisi | tool-selection top-1/top-3 (P1 vs P5 ayrı) |
#   argüman tam-eşleşme | halüsinasyon oranı | yetkisiz WRITE (=0) |
#   clarification doğruluğu (P8) | tool-result özet doğruluğu (P9)
#   GENELLEME KAPISI: top-1(P1) ≥ top-1(P5) − 10 puan
```
