# Büyük İK LoRA Dataset — ~100 Tool İçin Genellenebilir Policy Uygunluk Raporu

> Değerlendirme ekseni: **"Bu veri seti, modele 22 tool'un davranışını ezberletmek
> yerine ~100 tool üzerinde çalışan genellenebilir bir tool-calling policy
> öğretiyor mu?"**
> Tarih: 2026-08-28 · Yöntem: tüm dosyalar okundu, 3000 kayıt yeniden çözümlendi,
> validator + 321 test + deterministik üretim çalıştırıldı, tool-seçim / token /
> kısayol / cross-tool metrikleri ölçüldü. Yapısal temel inceleme:
> [FINETUNE_UYGUNLUK_RAPORU.md](FINETUNE_UYGUNLUK_RAPORU.md).

---

## 0. İki seviyeli çerçeve

| Seviye | Soru | Kısa cevap |
|---|---|---|
| **A — Tool-Level** | Mevcut 22 tool tek tek yeterince temsil edilmiş, parametreleri çıkarılabiliyor mu? | **Evet, yeterli.** |
| **B — Policy-Level** | Model ~100 tool üzerinde genellenebilir bir karar/seçim policy'si öğrenebilir mi? Yeni tool'a aktarılabilir mi? | **Hayır — mevcut haliyle yetersiz.** |

Bu iki sonuç **birbirinden bağımsızdır**: veri seti "her tool için yeterli örnek
var" testini geçer, ama bu tek başına policy hedefi için yeterli değildir.

---

## 1. Değerlendirme tablosu

| Bölüm | Durum | Yeterlilik | Tespit | Öneri |
|---|:--:|:--:|---|---|
| **Dataset Formatı** | ✅ | Yüksek | `{tools, messages}` = Qwen native; yalnız user/assistant; `id`/`intent`/`difficulty` yalnız `*.meta.jsonl`'de, eğitim dosyasında **yok**; LF/UTF-8/BOM temiz; `<tool_call>` katı tek-satır JSON. Policy öğrenmeye engel bir alan yok. | Kanonik `*_train.jsonl` olduğu gibi kullanılabilir; eğitimde `.meta.jsonl` yüklenmesin. |
| **Tool Tanımları** | ⚠️ | Orta | 22 açıklama net, tutarlı, `enum`+`required` var. Ama yalnız **22 tanım** binlerce kez tekrar ediyor → model açıklamayı okumadan adı ezberleyebilir. Tool adları karışık dilli (`get_employee_info` + `get_izin_bakiyesi`) — zararsız ama tutarsız. Opsiyonel param açıklamaları hiç kullanılmıyor. | Tanım sayısını artır; açıklama-okumayı zorunlu kılan örnekler ekle (§ Kritik Eksiklikler). |
| **Tool Selection** | ❌ | Düşük | Yalnız 22 tool; her biri 20–87× hedef. Aday liste **her zaman 4–9 tool** (asla daha fazla). Ölçülen anahtar-kelime→tool adı korelasyonu: `mesai`→`get_mesai_bilgisi` **%100**, `yönetici`→`get_yonetici_bilgisi` **%100**, `maaş`→`get_maas_bilgisi` **%97**, `bordro`→`get_bordro` **%96**. Bu "seçim" değil, **yüzey eşleştirmesi**. Ayırt etme yalnız aynı-domain 2–3 çeldiriciyle. 100 tool arasından seçim hiç görülmemiş. | Aday listeyi değişken ve büyük yap (15–60); anahtar-kelime kısayolunu kır; cross-domain çeldirici oranını artır. |
| **Parameter Extraction** | ⚠️ | Orta | Mekanizma sağlam: 0 uydurulmuş değer, her argüman kullanıcı metninden izlenebiliyor. Ortak param tipleri (tarih, ID, enum, dönem) birden çok tool'da kullanılıyor — iyi. Ama **kapalı, küçük slot havuzları**: 18 tarih yüzeyi, 10 talep-ID, ~10 tutar, 16 pozisyon → model yüzeyleri ezberler, havuz dışı ifade ("14 Mart 2027") çözülmez. Opsiyonel param **hiç** doldurulmuyor. | Havuzları 5–10× büyüt ve prosedürel üret; opsiyonel param örnekleri; aynı tipi farklı param **adlarıyla** öğret. |
| **General Policy** | ✅ | Orta-Yüksek | 4-karar çerçevesi (`direct`/`tool_call`/`request_for_info`/`cannot_answer`) **tool-agnostik** ve iyi temsil edilmiş (30/25/25/20). **19 imza**: aynı kullanıcı ifadesi → farklı karar (param var/yok) — güçlü, transfer edilebilir policy sinyali. "Onay-önce-yazma", "eksik param → sor", "tool listede var ≠ çağır" (600 `cannot_answer`'ın tamamında tool listesi dolu) kuralları tool'dan bağımsız. | Bu iskeleti koru; tool sayısı artınca karar oranlarını sabit tut. |
| **Cross-Tool Generalization** | ❌ | Düşük | Üretici mimarisi **per-tool**: her tool için elle yazılmış `READ_SPECS` / `MISSING_PARAM_SPECS` / `WRITE_SPECS` + elle `CONFUSABLE` haritası + domain'e özel cevap havuzları. 100 tool = 78 tool için elle şablon = doğrusal manuel iş, policy transferi değil. **Görülmemiş tool holdout'u yok.** Bir tool'da öğrenilen mantığın başkasına aktarıldığı gösterilmemiş (tüm tool'lar eğitimde). | Şema-güdümlü (template-free) üretime geç; tool'ların %15–20'sini eğitimden çıkar; "yeni tool + tanıdık mantık" değerlendirmesi. |
| **Tool Coverage** | ⚠️ | Orta | 22 tool dağılımı 4.3× aralıkta (min 20, medyan 46, max 87) — 22 tool için makul denge. READ 15 / WRITE 7. Ama "tool'lar arası ilişki" = yalnız 22-düğümlü `CONFUSABLE` grafiği + **4 sabit** multi-tool kombinasyonu. 100 tool için ilişki yapısı yok. | Tool sayısı + kategori + ilişki çeşitliliği artır; multi-tool kombinasyonlarını prosedürelleştir. |
| **Negative / Edge Cases** | ⚠️ | Orta-Yüksek | "Hiçbir tool uygun değil" iyi (600 `cannot_answer`, 7 domain, tool listesi dolu → okuma zorlar). Eksik param → soru (502). Uydurma engelleniyor (`adversarial_mutation`: 14 kusur enjekte, hepsi yakalanıyor). **Ama:** çelişkili/belirsiz parametre örneği yok (kullanıcı iki tarih verirse?); "yüzeyde uygun ama yanlış tool" çeldiricisi zayıf; "kullanıcı yanlış tool adı söylüyor" senaryosu yok. | Çelişkili/belirsiz girdi; güçlü hard-negative (yüzeyde eşleşen yanlış tool); kullanıcının yanlış tool istemesi. |
| **Tool Call Formatı** | ✅ | Yüksek | `<tool_call>\n{tek satır JSON}\n</tool_call>`; Qwen/Hermes uyumlu; streaming-parse güvenli; 0 biçim hatası; blok tek başına (önünde düz metin yok); çoklu çağrı = art arda blok. Inference formatıyla uyumlu. | Değişiklik gerekmez; çoklu-çağrı ve paralel-çağrı biçimini gerçek hedef runtime ile teyit et. |
| **Fine-Tuning Uyumu** | ⚠️ | Orta | 3000 örnek + 22 tool + LoRA → **ezber çok olası**. Token bütçesi test ile ≤3000'e kilitli; her örnek ≤9 tool. Loss masking standart. Determinizm/hijyen kusursuz. | Örnek sayısını 10–20k'ya çıkar; değişken/büyük aday liste; ayrı holdout eval; düşük LR + early stopping. |
| **100-Tool Ölçeklenebilirliği** | ❌ | Düşük | 100 tool şeması ≈ **14.000 token** (22 tool = ölçülen 3.128). Eğitim ≤3.000 token / ≤9 tool. Inference'ta tam katalog sunulursa model **hiç görmediği bağlam boyutuyla** karşılaşır. Retrieval ile ~10–15 aday sunulacaksa da eğitim bu değişkenliği yansıtmıyor. Üretici per-tool elle şablon → 100 tool = doğrusal manuel emek. | Mimari kararı ver: (a) retrieval + değişken aday sayısı eğitimi, veya (b) tam katalog + büyük token bütçesi. Her iki halde şema-güdümlü üretim. |

---

## 2. Genel Sonuç

### Soru 1 — Tool-level olarak dataset yeterli mi?

**EVET (Orta-Yüksek).** Mevcut 22 tool'un her biri:

- 20–87 kez tool çağrısının hedefi (medyan 46) — 22 tool için yeterli yoğunluk.
- Parametreleri kullanıcı metninden doğru ve uydurma olmadan çıkarılabiliyor
  (0 halüsinasyon, bağımsız izleme ile teyit).
- Şeması temiz (`enum`, `required`, tip kontrolü — 0 ihlal).
- Her örnekte 2–3 karıştırılabilir çeldiriciyle sunuluyor.
- READ/WRITE ayrımı ve onay akışı tutarlı.

Bu 22 tool'luk sistem için veri seti **eğitilebilir durumda**.

### Soru 2 — Policy-level olarak ~100 tool için genellenebilir mi?

**HAYIR (Düşük).** Gerekçeler:

1. **Görülmemiş tool yok.** Model yalnız 22 tool görüyor; hepsi eğitimde hedef.
   "Yeni tool → tanıdık mantık" davranışı ne öğretiliyor ne ölçülüyor. LoRA +
   3000 örnek, 22 ad↔ifade eşlemesini rahatça ezberler.
2. **Ölçülebilir yüzey kısayolu.** Birçok tool için kullanıcı ifadesinde tool'un
   işlev kelimesi %95–100 oranında geçiyor (`mesai`, `bordro`, `maaş`,
   `yönetici`...). Model açıklama okumadan "kelime → ad" öğrenir → bu tam olarak
   istemediğiniz "doğrudan tool eşleştirmesi".
3. **Aday liste hep küçük (4–9).** Gerçek senaryo (100 tool ya da retrieval'lı
   15–40 aday) hiç görülmüyor. "Kalabalık katalogda doğru tool'u bul" becerisi
   eğitilmiyor.
4. **Token bütçesi uyumsuzluğu.** Eğitim ≤3.000 token; 100 tool ≈ 14.000 token.
   Inference dağılımı eğitim dağılımının tamamen dışında.
5. **Per-tool üretici mimarisi.** `READ_SPECS`, `MISSING_PARAM_SPECS`,
   `WRITE_SPECS`, `CONFUSABLE`, cevap havuzları — hepsi elle, tool'a özel.
   100 tool'a çıkmak "policy'yi genişletmek" değil, "78 tool için elle içerik
   yazmak" demek.
6. **Tool sonucu turu yok.** İstediğiniz döngünün son adımı ("tool sonucunu uygun
   şekilde kullan") veri setinde **hiç yok** — `tool` rolü, `<tool_response>`
   hiçbir örnekte geçmiyor.

**Sağlam olan taraf:** 4-karar policy iskeleti, "onay-önce-yazma", "eksik
param → sor", "tool var ≠ çağır", "değer uydurma" kuralları **gerçekten
tool-agnostik** ve iyi öğretilmiş. Yani policy'nin *çerçevesi* doğru; eksik olan
*ölçek, çeşitlilik ve tool-sonucu davranışı*.

---

## 3. Kritik Eksiklikler — tool ezberleme riskini yaratan yapılar

| # | Problem | Kanıt (ölçüm) | Ezberleme etkisi |
|---|---|---|---|
| K-1 | **Yüzey kelimesi → tool adı korelasyonu** | `get_mesai_bilgisi` %100, `get_yonetici_bilgisi` %100, `get_maas_bilgisi` %97, `get_bordro` %96 kullanıcı turunda işlev kelimesi var | Model "kelime eşleştirme" öğrenir, "açıklama okuma" değil. Yeni tool'da kelime tutmazsa policy çöker. |
| K-2 | **Sabit ve küçük tool evreni** | Tüm veride yalnız 22 farklı tool adı; holdout 0 | 22 ad↔kalıp eşlemesi LoRA ağırlıklarına yazılır. |
| K-3 | **Küçük aday liste** | tools[] boyutu 4–9, medyan 6; >9 hiç yok | "Kalabalık katalogda ara" becerisi eğitilmiyor; 100 tool inference'ta dağılım-dışı. |
| K-4 | **Per-tool elle şablonlar** | `READ_SPECS` 15 blok, `MISSING_PARAM_SPECS` 24, `WRITE_SPECS` 7, `CONFUSABLE` 22 giriş — hepsi manuel | Yapı doğrusal ölçekleniyor; her yeni tool ayrı içerik → "policy" değil "tool kütüphanesi". |
| K-5 | **Kapalı slot havuzları** | 18 tarih yüzeyi → 50 ISO değeri; 10 talep-ID; ~10 tutar | Parametre çıkarımı yüzey ezberi; havuz dışı ifade genellenmez. |
| K-6 | **Tool sonucu turu yok** | `tool` rolü 0, `<tool_response>` 0 | Model tool çağırıp durur; sonuç yorumlamayı LoRA öğretmez, hatta bastırır. |
| K-7 | **Zayıf hard-negative** | Çeldiriciler yalnız aynı-domain; "yüzeyde uygun ama yanlış" tool senaryosu yok; çelişkili param yok | Model gerçekten zor ayrımları görmüyor; kendine güveni kalibre değil. |
| K-8 | **Multi-tool sığ** | 4 sabit kombinasyon, hepsi paralel READ, aynı çalışan, 63 örnek | "Birden fazla tool arasında karar / sıralama" öğretilmiyor. |
| K-9 | **`direct` cevap ezberi** | 750 örnekte 133 benzersiz cevap; bazıları 14× | Model kalıp paragraf ezberler. |
| K-10 | **`val` gerçek holdout değil** | Aynı şablon+slot havuzundan, intent-stratifiye | Eğitim sırasında generalization ölçülemez; ezber fark edilmez. |

---

## 4. Düzenleme Kararı — değişiklik planı

**Hedef dönüşüm:** "22 tool'a özel şablon kütüphanesi" → "**tool şemasını
girdi kabul eden, şemadan işlev çıkaran, herhangi bir tool'a uygulanabilen
karar/seçim/parametre policy'si**".

Her madde: **MEVCUT YAPI → PROBLEM → GEREKLİ DEĞİŞİKLİK → NEDEN → BEKLENEN ETKİ**

---

### D-1. Tool evrenini genişlet ve holdout ayır

- **MEVCUT YAPI:** `generate_dataset.py` içinde 22 tool sabit; hepsi hem eğitimde
  hem (varsa) doğrulamada.
- **PROBLEM:** Model 22 adı ezberleyebilir; yeni tool'a genelleme ne öğretiliyor
  ne ölçülüyor (K-2).
- **GEREKLİ DEĞİŞİKLİK:** Tool kataloğunu **≥80–120 tool**'a çıkar (birden çok
  alan: İK + finans + BT destek + CRM + lojistik...). Tool'ları **3 kümeye** böl:
  `train_tools` (~%70), `val_tools` (~%15, eğitimde hiç görülmez),
  `test_tools` (~%15, yalnız final değerlendirme). Aynı intent kalıpları
  train ve val_tools üzerinde ayrı ayrı örneklenir.
- **NEDEN:** "Yeni tool + tanıdık mantık" ancak eğitimde görülmeyen tool üzerinde
  ölçülebilir; policy'nin tool-agnostik olduğunun tek kanıtı budur.
- **BEKLENEN ETKİ:** `val_tools` üzerinde tool-seçim doğruluğu, `train_tools`'a
  yakınsa → policy genelleniyor. Uzaksa → ezber var, erken görülür.

---

### D-2. Per-tool şablonlardan şema-güdümlü üretime geç

- **MEVCUT YAPI:** Her tool için elle `READ_SPECS` / `MISSING_PARAM_SPECS` /
  `WRITE_SPECS` girdisi (metin + argmap + templates), elle `CONFUSABLE` haritası.
- **PROBLEM:** 100 tool = 100 elle blok. Yapı "policy" değil "tool başına içerik"
  (K-4). Yeni tool eklemek = yeni veri yazmak.
- **GEREKLİ DEĞİŞİKLİK:** Üretimi **tool şemasından türet**: (1) tool
  `description` + parametre `description`'larından intent ifadeleri üret
  (parametrik şablonlar: "{fiil} {nesne}", "{emp} için {param_desc}"),
  (2) çeldiricileri elle harita yerine **şema benzerliğiyle** seç (aynı param
  imzası / açıklama gömme yakınlığı / aynı kategori), (3) eksik-param mantığını
  `required` listesinden otomatik türet.
- **NEDEN:** Aynı üretim mantığı 22 veya 220 tool'a fark etmeden uygulanır;
  içerik tool'a değil şemaya bağlanır.
- **BEKLENEN ETKİ:** Yeni tool eklemek = kataloğa bir şema satırı; veri seti
  büyümesi sabit maliyetli. Model "şema oku → davran" ilişkisini görür.

---

### D-3. Aday tool listesini büyüt ve değişkenleştir

- **MEVCUT YAPI:** `build_tools_list` her örnekte 4–9 tool (hedef + 2–3 komşu +
  rastgele).
- **PROBLEM:** Model asla >9 tool görmüyor; 100-tool / retrieval'lı inference
  dağılım-dışı (K-3). Anahtar kelime kısayolu küçük listede işe yarıyor (K-1).
- **GEREKLİ DEĞİŞİKLİK:** Aday liste boyutunu **değişken dağıtım** yap: %20 örnek
  5–10 tool, %50 örnek 15–35 tool, %30 örnek 40–80 tool. Her listede:
  1 hedef + şema-yakını 2–4 güçlü çeldirici + geri kalanı **cross-domain
  rastgele**. Hedefin liste içi konumu uniform rastgele.
- **NEDEN:** Gerçek inference bağlamını (retrieval çıktısı veya tam katalog)
  yansıtır; büyük listede sadece kelime eşleştirmesi yetmez, açıklama okumak
  gerekir.
- **BEKLENEN ETKİ:** Model "kalabalıkta ayırt etme" öğrenir; token bütçesi ve
  liste boyutu inference ile uyumlu olur.

---

### D-4. Yüzey–ad kısayolunu kır

- **MEVCUT YAPI:** Şablon metinleri sık sık tool'un işlev kelimesini birebir
  içeriyor ("fazla mesai" → `get_mesai_bilgisi`), ölçülen korelasyon %95–100.
- **PROBLEM:** Model açıklamayı okumadan kelime → ad eşler (K-1). Yeni tool'da
  kelime tutmazsa policy başarısız.
- **GEREKLİ DEĞİŞİKLİK:** Intent ifadelerinin **en az yarısında** tool
  adının/açıklamasının kök kelimesini kullanma; dolaylı anlatım kullan
  ("ay sonunda kaç saat fazladan çalıştım" yerine "geçen ay normalin üstünde ne
  kadar mesaim oldu, karşılığı ne"). Ayrıca **paylaşılan kelime, farklı tool**
  örnekleri ekle ("izin" kelimesi hem `get_izin_bakiyesi` hem `create_izin_talebi`
  hem `get_izin_gecmisi` için — ayrımı fiil/zaman kipi belirler).
- **NEDEN:** Model niyeti anlamaya, yüzey eşleştirmeye değil, zorlanır.
- **BEKLENEN ETKİ:** Tool-seçim, açıklama semantiğine dayanır → görülmemiş tool'a
  aktarılabilir.

---

### D-5. Tool sonucu turunu ekle

- **MEVCUT YAPI:** Her örnek assistant'ın kararında (tool_call / soru / ret)
  bitiyor. `tool` rolü ve `<tool_response>` hiç yok (When2Call §33 tercihi).
- **PROBLEM:** İstenen davranışın son adımı ("tool sonucunu uygun şekilde kullan")
  hiç öğretilmiyor (K-6). Dağıtımda ikinci inference turu gerekli ve model bunu
  görmemiş.
- **GEREKLİ DEĞİŞİKLİK:** Örneklerin **%30–40'ında** akışı uzat:
  `user → assistant(tool_call) → tool(JSON sonuç) → assistant(Türkçe cevap)`.
  Bazılarında sonuç hata/boş dönsün → model "sonuç yok / hata" durumunu ele
  alsın. Çok-adımlıda: `tool sonucu → ikinci tool_call` (sonuca dayalı dallanma).
- **NEDEN:** Policy'nin son adımı; ayrıca "sonuç geldi, artık uydurmadan özetle"
  davranışı ezber karşıtıdır.
- **BEKLENEN ETKİ:** Model tam döngüyü öğrenir: anla → seç → çağır → **oku →
  cevapla**. Gerçek agent kullanımıyla uyumlu.

---

### D-6. Slot havuzlarını prosedürelleştir

- **MEVCUT YAPI:** `DATE_RANGES` (18), `MONTH_RANGES` (12), `DONEMLER` (21),
  `TALEP_IDS` (10), `AMOUNT_POOL` (10), `POZISYONLAR` (16) — sabit listeler.
- **PROBLEM:** Model 50 ISO tarih değeri ve 18 yüzey ifadesini ezberler; havuz
  dışı parametre çözülmez (K-5).
- **GEREKLİ DEĞİŞİKLİK:** Tarih/dönem/tutar/ID'leri **üretim anında rastgele
  sentezle** (takvim-geçerli tarih üreteci; farklı format/dil kipleriyle yüzey;
  geniş tutar aralığı; ID biçimleri). Yüzey→kanonik çözümü kural tabanlı yap,
  liste tabanlı değil.
- **NEDEN:** Parametre çıkarımı bir *beceri* olarak öğrenilir, yüzey ezberi olarak
  değil.
- **BEKLENEN ETKİ:** "14 Mart 2027", "önümüzdeki çeyreğin son günü" gibi
  görülmemiş ifadeler de doğru çözülür.

---

### D-7. Hard-negative ve belirsizlik örnekleri ekle

- **MEVCUT YAPI:** Çeldiriciler aynı-domain; `cannot_answer` net kapsam-dışı;
  belirsiz/çelişkili param yok.
- **PROBLEM:** Model gerçekten zor ayrımları ve "emin değilim" durumunu görmüyor
  (K-7).
- **GEREKLİ DEĞİŞİKLİK:** Ekle: (a) yüzeyde uygun ama yanlış tool listede
  (ör. "izin durumu" → `get_izin_talebi_durumu` mu `get_izin_bakiyesi` mi;
  ipucu ince), (b) kullanıcı yanlış tool adı söylüyor ("bordro getir" ama aslında
  maaş istiyor), (c) çelişkili parametre ("15 Eylül'den 10 Eylül'e kadar") →
  netleştirme sorusu, (d) kısmi eşleşen tool listede yokken kapsam-dışı.
- **NEDEN:** Policy'nin sınır davranışı; kalibre güven.
- **BEKLENEN ETKİ:** Model niyet–tool arasındaki ince ayrımları semantikten
  çözer; çelişkide uydurmaz, sorar.

---

### D-8. Multi-tool'u derinleştir

- **MEVCUT YAPI:** 4 sabit paralel-READ kombinasyonu, 63 örnek.
- **PROBLEM:** "Birden fazla tool arasında karar / sıralama / bağımlılık"
  öğretilmiyor (K-8).
- **GEREKLİ DEĞİŞİKLİK:** Prosedürel multi-tool: 2–4 tool, bazıları sıralı
  (birinin çıktısı ötekinin girdisi — D-5 ile birlikte), bazıları READ+WRITE
  karışık, farklı domainlerden.
- **NEDEN:** Gerçek görevler çok-tool; policy bunu kapsamalı.
- **BEKLENEN ETKİ:** Model görev ayrıştırma + tool sıralama öğrenir.

---

### D-9. Örnek sayısı, dağılım ve holdout eval

- **MEVCUT YAPI:** 3000 örnek, `val` aynı havuzdan.
- **PROBLEM:** 100 tool + geniş yüzey için 3000 az; `val` generalization ölçmez
  (K-10).
- **GEREKLİ DEĞİŞİKLİK:** **12–20k örnek**; `val` = `val_tools` üzerinden
  (D-1); ayrıca elle yazılmış **200–400 örneklik `hard_eval`** (havuz-dışı
  yüzeyler, `test_tools`, zor ayrımlar). Değerlendirme metrikleri: karar 4×4
  matrisi, tool-seçim top-1/top-3 (train_tools vs val_tools ayrı), argüman-tam-
  eşleşme, halüsinasyon oranı, WRITE-onaysız-çağrı (0 olmalı), tool-sonucu-
  özetleme kalitesi.
- **NEDEN:** Ölçek + bağımsız ölçüm olmadan ezber/generalization ayırt edilemez.
- **BEKLENEN ETKİ:** Eğitim sırasında `val_tools` eğrisi ezberi erken gösterir;
  karar verilebilir.

---

### D-10. Opsiyonel parametre davranışı

- **MEVCUT YAPI:** `aciklama`, `gecerlilik_tarihi` vb. hiç doldurulmuyor.
- **PROBLEM:** Model "opsiyonel alan var, kullanıcı bilgi verdi → doldur"u hiç
  görmüyor; zorunlu/opsiyonel ayrımının yarısı öğretiliyor.
- **GEREKLİ DEĞİŞİKLİK:** Kullanıcının opsiyonel bilgi verdiği örnekler ekle
  (gerekçe, açıklama, geçerlilik tarihi) → argümanda görünsün; vermediğinde
  görünmesin.
- **NEDEN:** Tam parametre policy'si = zorunlu + opsiyonel + yok.
- **BEKLENEN ETKİ:** Model opsiyonel alanları bağlama göre doğru kullanır,
  gereksiz uydurmaz.

---

### Uygulama sırası (önerilen)

1. **D-1 + D-2** (tool evreni + şema-güdümlü üretim) — diğer her şeyin önkoşulu.
2. **D-3 + D-4** (aday liste + kısayol kırma) — policy'nin özü.
3. **D-5** (tool sonucu turu) — eksik davranış.
4. **D-6 + D-10** (parametre becerisi).
5. **D-7 + D-8** (zorluk ve çok-tool).
6. **D-9** (ölçek + eval) — süreç boyunca paralel.

Korunacaklar (yeniden yazma): `{tools, messages}` formatı, `<tool_call>` biçimi,
4-karar çerçevesi ve oranları, onay-önce-yazma otomatı, halüsinasyon-izleme
disiplini, determinizm/LF/UTF-8 hijyeni, test paketi felsefesi (orakel +
mutasyon meta-testi).

---

## 5. Son Karar

# → NEEDS DATASET REVISION

**Tool-level (A):** yeterli — mevcut 22 tool iyi temsil edilmiş, parametreleri
çıkarılabiliyor, format ve güvenlik kusursuz.

**Policy-level (B):** yetersiz — veri seti şu anda 22 tool'a özel bir şablon
kütüphanesidir; ~100 tool üzerinde genellenebilir, yeni tool'a aktarılabilir bir
tool-calling policy öğretmez. Bunun için **önemli veri/örnek düzenlemeleri**
gereklidir (D-1…D-10): tool evrenini genişletmek + holdout ayırmak, per-tool
şablonları şema-güdümlü üretime çevirmek, aday listeyi büyütmek/değişkenleştirmek,
yüzey-ad kısayolunu kırmak, tool sonucu turunu eklemek, slot havuzlarını
prosedürelleştirmek.

**NOT READY değil**, çünkü yeniden kullanılabilir sağlam bir temel var: format,
4-karar çerçevesi, onay otomatı, halüsinasyon disiplini ve test altyapısı
korunabilir — sıfırdan başlamak gerekmez, revize edilip ölçeklenir.

En önemli kriter — "~100 tool arasında genellenebilir policy öğrenme kapasitesi" —
şu an **karşılanmıyor**; "her tool için yeterli örnek var" olması bu sonucu
değiştirmez.

---

*Rapor sonu.*
