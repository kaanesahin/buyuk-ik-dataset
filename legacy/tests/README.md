# Büyük İK dataset — test paketi

`data/` altındaki üretilmiş veri setini **artefakt olarak** doğrulayan `pytest`
paketi: **25 dosya, ~320 test**. `scripts/validate_dataset.py` hızlı bir kapı
(gate) iken bu paket, her niteliği ayrı ayrı, isimlendirilmiş ve hata mesajı
açıklayıcı testlerle kontrol eder — CI'de ve regresyon avında kullanılır.

Üç kümede toplanır:

* **Çekirdek (1–10)** — yapısal/anlamsal temel güvenceler.
* **İleri (11–20)** — istatistiksel, bilgi-kuramsal ve biçimsel-otomat düzeyinde
  derin denetimler.
* **Sert (21–25)** — orakel (etiketi ilk ilkelerden yeniden türet), mutasyon
  meta-testi, zamansal akıl yürütme, sohbet-şablonu güvenliği, tasarım-uzayı kapsaması.

## Kurulum ve çalıştırma

```bash
pip install -r tests/requirements-test.txt        # yalnızca pytest

pytest tests/                     # tüm paket  (~20 sn)
pytest tests/ -m "not slow"       # üretici alt süreçlerini atla
pytest tests/ -m "not statistical" # üretim oranlarına duyarlı testleri atla
pytest tests/test_decision_oracle.py -v
pytest tests/ -k "chain or oracle" # ada göre filtrele
```

> Üretici (`scripts/generate_dataset.py`) **stdlib-only** kalır. `pytest` yalnızca
> bir *test* bağımlılığıdır (`requirements-test.txt`), çalışma zamanı bağımlılığı değil.

Veri dosyaları yoksa testler `skip` olur; önce:

```bash
python scripts/generate_dataset.py
python scripts/make_preview.py
python scripts/build_training_variants.py   # opsiyonel
```

## Çekirdek küme (1–10)

| # | dosya | ne doğrular | prompt bölümü |
|---|---|---|---|
| 1 | `test_jsonl_structure.py` | JSONL geçerliliği, UTF-8/LF hijyeni, eğitim kaydı = `{tools, messages}`, mesaj/rol/içerik şekli, meta satır sayısı eşleşmesi | §22, §37 |
| 2 | `test_tool_schema.py` | 22 tool şeması (JSON-Schema alt kümesi), `required ⊆ properties`, enum temizliği, envanter = `TOOLS`, politika sabitlerinin geçerliliği | §14 |
| 3 | `test_tool_calls.py` | `<tool_call>` biçimi, JSON ayrıştırma, çağrılan tool tanımlı mı, argüman anahtar/tip/enum/zorunlu kontrolü, meta hedef tutarlılığı, WRITE kapsamı, histogram tavanı | §21, §31 |
| 4 | `test_no_hallucination.py` | **her** tool-call argümanı kullanıcı turundan türetilebiliyor mu (ID, tarih, dönem, tutar, enum, serbest metin); assistant düz metninde uydurma kimlik/sayı yok | §18, §30, §31 |
| 5 | `test_decision_semantics.py` | 4 kararın davranışsal doğruluğu; `direct/request_for_info/cannot_answer` → çağrı yok; onay-önce-yazma politikası; aynı intent'in hem çağrı hem soru örneği | §4, §12, §35 |
| 6 | `test_conversation_flow.py` | katı rol alternasyonu, tur sayıları (2/4/6), çok-adımlı zincir mekaniği (topla→onay→uygula), onaylı-WRITE 4-tur akışı, mt_info değer taşıma | §12, §20, §25 |
| 7 | `test_distribution.py` | karar karışımı ±3.5 puan, `cannot_answer` domain yayılımı, zorluk kuyruğu, register çeşitliliği, çok-turlu/WRITE payları, train/val bölmesi | §6, §7, §17 |
| 8 | `test_diversity_and_leakage.py` | birebir tekrar yok, train↔val sızıntısı (klasik + normalize imza), §32 klon üretimi, intent-içi yüzey çeşitliliği, açılış kalıbı yoğunluğu | §7, §31, §32 |
| 9 | `test_privacy_and_safety.py` | gerçek PII kalıbı yok (TCKN/IBAN), sentetik ID biçimleri, telefon/e-posta/isim havuzları; başkasının verisi/gelecek tahmini/yıkıcı işlem → `cannot_answer`; yetki yalnız `check_employee_access` ile | §2, §19, §27, §30 |
| 10 | `test_generator_reproducibility.py` | `slow` — aynı seed → byte-aynı; farklı seed → farklı; ölçekte dağılım; `--dry-run` yazmıyor; `validate_dataset` exit 0; önizleme kaynağı bozmuyor; varyant turları koruyor | §22, §31, §37 |

## İleri küme (11–20) — derin denetimler

| # | dosya | ne doğrular | yöntem |
|---|---|---|---|
| 11 | `test_statistical_balance.py` | karar/register dağılımı hedeften **istatistiksel olarak** ayırt edilemiyor; decision×domain bağımlılığı tasarım sözleşmesine uyuyor; val oranı %99 GA'sı 0.10'u içeriyor | ki-kare uyum iyiliği (stdlib), Wilson skoru, kontenjans artıkları |
| 12 | `test_lexical_diversity.py` | Guiraud kök-TTR ≥ 5; ilk-kelime entropisi ≥ 4.5 bit; val n-gram yeniliği; cevap havuzu yoğunlaşması; **çözülmemiş şablon artefaktı / sızmış slot adı yok** | bilgi-kuramsal ölçümler, n-gram küme farkı |
| 13 | `test_tool_selection_discriminability.py` | çeldirici sayısı 3–8; ≥ %85 örnekte karıştırılabilir komşu var; **kardeş tool'un dışlayıcı sinyali kullanıcı metninde yok** (§16 sınırları); intent → tek target_tool/domain | kelime-sınırlı dışlayıcı sinyal sözlükleri |
| 14 | `test_parameter_grounding_precision.py` | her ISO tarih/dönem/tutar/gerekçe/pozisyon argümanı üreticinin **kapalı slot havuzundan**; tarih çiftleri tek kayıttan, sıralı, ≤ 92 gün; kullanıcı açık ISO tarih verdiyse çağrı onu kullanıyor | havuz üyeliği, çift eşleşmesi, ters grounding |
| 15 | `test_missing_parameter_logic.py` | `request_for_info` temiz partisyon (eksik-bilgi ∪ onay); eksik param gerçekten yok, eksik olmayan zorunlu param var; **soru tam olarak eksik param'ı hedefliyor**; `missing ⊆ properties` | varlık kontrol sözlükleri, soru anahtar kelimeleri |
| 16 | `test_write_safety_state_machine.py` | WRITE akışı biçimsel otomat: EXECUTE'ten önce CONFIRM+ONAY; 2-tur asla WRITE ile bitmez; **CONFIRM işlemi somut anlatıyor** (emp no / talep / tarih / tutar CONFIRM'de); onay olumlu; ask/exec intent kümeleri eşit | durum makinesi yolu doğrulama |
| 17 | `test_turn_coherence.py` | asistan sorduysa kullanıcı TAM onu veriyor; verilen değer çağrıda kullanılıyor; **yanıtlanmış soru tekrar sorulmuyor**; çok-turlu direct: takip sorusu → esaslı yanıt; çok-turlu cannot: ikinci ret ≠ ilki | tur-tur soru↔yanıt eşleştirme |
| 18 | `test_encoding_and_serialization.py` | round-trip kararlılık; Türkçe literal (kaçışsız); **NFC-normal**; yalın vekil yok; tekrar eden JSON anahtarı yok; **tool_call bloğu `tool_call_block` çıktısıyla birebir yeniden üretilebilir**; `tools.json` indent=2 kanonik | yeniden serileştirme, Unicode normalizasyonu |
| 19 | `test_curriculum_and_difficulty.py` | `kolay` gerçekten basit; `cok_zor`'un somut zorluk kaynağı var; **ortalama karmaşıklık skoru kesin monoton** (kolay<orta<zor<cok_zor); zincir → cok_zor; `uzun` register > `kisa` register uzunluğu; dağılım bimodal çökmemiş | karmaşıklık skoru, monotonluk |
| 20 | `test_semantic_intent_consistency.py` | iki farklı intent aynı ilk-turu üretmiyor; **intra-intent kohezyon ≥ 3× inter-intent ayrışma** (Jaccard); her intent izin verilen karar-kümesi kalıbında; intent adı ↔ tool adı; §16 çiftlerinin imza kesişimi < %20 | ikili Jaccard kohezyon/ayrışma analizi |

## Sert küme (21–25) — en zorlu denetimler

| # | dosya | ne doğrular | yöntem |
|---|---|---|---|
| 21 | `test_decision_oracle.py` | **etiketin KENDİSİ doğru mu?** When2Call karar süreci (§36) bir orakel olarak yeniden uygulanır ve 3000 örneğin tamamına karşı çalıştırılır: intent→spec-havuzu→izin verilen karar kümesi; `tool_call`(okuma) ⇒ tüm zorunlu param metinden çıkarılabilir; `request_for_info` ⇒ gerçekten bir eksik var; WRITE yürütme ⇒ onay turu; bilgi/ret intent'leri asla tool çağırmaz | ilk-ilkelerden tam yeniden türetme (paritesiz orakel) |
| 22 | `test_adversarial_mutation.py` | **güvenlik ağının boş olmadığının kanıtı** — gerçek kayıtlara 14 tür kusur enjekte edilir (zorunlu argüman sil, enum boz, kimlik uydur, tarih uydur, tool'u listeden çıkar, `direct`+tool_call, erken tool_call, rol bozulması…) ve `validate_dataset.check_record`'ın HER BİRİNİ yakaladığı; mutasyonsuz 3000 kayıtta yanlış pozitif olmadığı | mutasyon meta-testi |
| 23 | `test_temporal_consistency.py` | ISO tarih argümanları takvim-geçerli (Şubat 2026 = 28 gün); "N günlük" yüzeyleri doğru bitiş (ay/yıl taşması); ay/çeyrek aralıkları takvim sınırında; göreli çözüm (`bu ay`/`geçen ay`) `--today`'e göre; **izin talepleri geleceğe dönük**; bordro/mesai dönemi makul pencerede; geçmiş sorguları geleceğe uzanmıyor; tarih yılı kullanıcı metniyle aynı | takvim aritmetiği, iş-mantığı sınırları |
| 24 | `test_chat_template_rendering.py` | özel token (`<|im_start|>`…) sızıntısı yok; `<tool_call>` işaretleri dengeli + yalnız assistant'ta; `<tool_response>` yok (§33); şablon-kıran dizi (`{{`,`{%`) yok; minimal ChatML renderer iyi biçimli çıktı veriyor (system+araçlar, sıralı roller, `<\|im_start\|>`/`<\|im_end\|>` dengeli); gömülü `tools` round-trip; token bütçesi (~≤3000) | minimal ChatML renderer + işaret dengeleme |
| 25 | `test_negative_space_coverage.py` | **üretilebilecek her şey üretilmiş mi** — 22 tool'un tamamı çağrılmış + her tool erişilebilir; DATE_RANGES/MONTH_RANGES/DONEMLER/GEREKCE/POZISYON/AMOUNT/TALEP yüzeylerinin tamamı kullanılmış; 6 ret havuzu işletilmiş; öksüz spec yok; register×decision (24 hücre) ve domain×difficulty ızgaraları dolu; §27 hard-negative senaryolarının hepsi mevcut; ≥10 train örnekli her intent val'da | kapsama analizi, çapraz ızgara |

`conftest.py` — paylaşılan fixture'lar (`all_records`, `paired`, `all_meta`,
`tools_inventory`, `gen`) ve yardımcılar (`iter_tool_calls`, `user_blob`,
`strip_tool_calls`, `fold`, `TOOLCALL_RE`).

## İşaretçiler (markers)

| marker | anlamı |
|---|---|
| `slow` | üreticiyi alt süreçte koşar (yalnızca `test_generator_reproducibility.py`) |
| `statistical` | üretim oranlarına duyarlı; `TARGET_MIX` / alt-kırılım değişirse eşikler gözden geçirilmeli (11, 12, 19 bölümleri) |

## Bağımsızlık notu

Testlerin çoğu üreticiden **bağımsızdır** (yalnız `data/` dosyalarını okur).
Üreticiye/validator'a bilerek bağlı olanlar — tıpkı `validate_dataset.py` gibi (When2Call §31):

* `test_no_hallucination.py`, `test_parameter_grounding_precision.py`,
  `test_write_safety_state_machine.py`, `test_temporal_consistency.py`,
  `test_decision_oracle.py`, `test_negative_space_coverage.py` — "15-20 Eylül 2026"
  → "2026-09-15" gibi yüzey→kanonik çözümleme ve spec-havuzu erişimi için
  `generate_dataset`'in kapalı slot havuzlarını / spec listelerini kullanır.
* `test_encoding_and_serialization.py` — `tool_call_block` kanoniklik kontrolü için.
* `test_adversarial_mutation.py` — `validate_dataset.check_record`'ı meta-test eder.
* `test_generator_reproducibility.py` — üreticiyi/yardımcı script'leri alt süreçte koşar.

## CI örneği

```yaml
# .github/workflows/tests.yml (örnek)
- run: pip install -r tests/requirements-test.txt
- run: python scripts/generate_dataset.py
- run: python scripts/make_preview.py
- run: pytest tests/ -q
```
