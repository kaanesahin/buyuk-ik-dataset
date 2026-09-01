# Büyük İK v2 — Dataset İstatistikleri

> Otomatik üretilir: `python scripts/metrics.py`. Kaynak: `data/`.

## Temel sayılar

| ölçüt | değer |
|---|---|
| train örnek | **15000** |
| val örnek | **2000** (val_seen + val_unseen_tool) |
| hard_eval örnek | **1003** |
| toplam tool (katalog) | **105** |
| train_tools / val_tools / test_tools | **75 / 15 / 15** |
| domain sayısı | **13** (calendar, crm, documents, finance, hr, inventory, it_support, logistics, payroll, reporting, sales, support, timesheet) |
| read / write / action | **66 / 35 / 4** |
| train hedef-tool sayısı | 75 / 75 |
| tool başına örnek (min/medyan/maks) | 101 / 148 / 364 |

## Karar dağılımı (train)

| decision | adet | oran |
|---|---:|---:|
| `tool_call` | 8475 | 56.5% |
| `request_for_info` | 3000 | 20.0% |
| `cannot_answer` | 1875 | 12.5% |
| `direct` | 1650 | 11.0% |

## Senaryo dağılımı (train)

| senaryo | adet | oran |
|---|---:|---:|
| `read_call` | 5175 | 34.5% |
| `missing_param` | 1725 | 11.5% |
| `direct` | 1650 | 11.0% |
| `cannot_scope` | 1500 | 10.0% |
| `write_execute` | 975 | 6.5% |
| `write_confirm` | 975 | 6.5% |
| `multi_parallel` | 600 | 4.0% |
| `write_chain` | 525 | 3.5% |
| `multi_sequential` | 450 | 3.0% |
| `hn_keyword_ambiguous` | 450 | 3.0% |
| `hn_tool_absent` | 375 | 2.5% |
| `hn_user_names_wrong_tool` | 300 | 2.0% |
| `hn_conflict` | 300 | 2.0% |

## Tool çağrı yapısı

| ölçüt | adet | oran |
|---|---:|---:|
| tek-tool tool_call | 7425 | 49.5% |
| çoklu-tool (paralel + sıralı) | 1050 | 7.0% |
|  — bunun sıralısı (sonuç→param) | 450 | 3.0% |
| **tool-sonucu turu içeren** | 3908 | 26.1% |
|  — tool_call örnekleri içinde | 3908 | 46.1% |
| tool-sonucu modu | {None: 1239, 'ok': 1468, 'partial': 322, 'empty': 491, 'error': 388} |  |
| WRITE/action örneği | 3124 | 20.8% |
| onay akışı (confirm) | 2475 | 16.5% |
| 6-turlu zincir (eksik→onay→uygula) | 525 | 3.5% |
| tur dağılımı | {2: 9141, 4: 4095, 5: 298, 6: 1466} |  |

## Parametre davranışı

| ölçüt | adet | oran |
|---|---:|---:|
| opsiyonel parametre KULLANILAN örnek | 1088 | 7.3% |
| eksik-parametre (request_for_info) | 2250 | 15.0% |

## Aday tool listesi boyutu (train)

| kova | adet | oran | hedef |
|---|---:|---:|---:|
| ≤12 tool | 5080 | 33.9% | ~28% |
| 13–34 tool | 7783 | 51.9% | ~56% |
| 35–58 tool | 2137 | 14.2% | ~16% |
| medyan / p90 / maks | 19 / 42 / 58 |  |  |
| hedef tool'un liste-içi konumu (ort., 0=baş 1=son) | 0.49 (uniform ~0.50) |

## Hard-negative örnekleri (train)

Toplam **1425** (9.5%).

| tür | adet |
|---|---:|
| A_keyword_ambiguous | 450 |
| F_tool_absent | 375 |
| D_user_names_wrong_tool | 300 |
| E_conflict | 300 |

## Doğal dil çeşitliliği (train)

| register | adet | oran |
|---|---:|---:|
| `plain` | 3521 | 23.5% |
| `formal` | 3164 | 21.1% |
| `chat` | 2994 | 20.0% |
| `long` | 2702 | 18.0% |
| `typo` | 1645 | 11.0% |
| `short` | 974 | 6.5% |

- benzersiz ilk-kullanıcı-turu (folded): **14461 / 15000** (96.4%)
- **yüzey kelimesi → tool korelasyonu (K-1):**
  - nesnenin ana adı geçiyor mu (dürüst üst sınır): **53%** — örneklerin ~yarısında model açıklamayı/aday listeyi okumak zorunda
  - tüm yüzey sözlüğü: 60%
  - en nadir ayırt edici token (alt sınır): 33% (v1 karşılığı ~%95–100 idi — patolojik fiil→tekil-tool eşlemesi kırıldı)
  - ana-ad korelasyonu en yüksek: find_free_slot 97%, create_expense_report 90%, create_leave_request 89%, create_contact 84%, get_item 80%, export_dataset 79%

## Sızıntı ve tekrar

- train↔val kullanıcı-turu imza kesişimi: **1**
- train hedefinde val/test tool'u: **0**
- val_unseen_tool hedefleri (hepsi split=val): **15/15**
- hard_eval P1/P9 hedefleri (hepsi split=test): **15/15**

## Tool kapsama

- 75 train tool'unun tamamı hedef: **EVET**
- dağılım eğriliği (maks/min): **3.6×**

| domain | tool_call | request_for_info | (train tool) |
|---|---:|---:|---:|
| calendar | 508 | 223 | 5 |
| crm | 1295 | 263 | 7 |
| documents | 477 | 226 | 5 |
| finance | 878 | 266 | 7 |
| hr | 997 | 282 | 7 |
| inventory | 536 | 157 | 5 |
| it_support | 527 | 408 | 7 |
| logistics | 526 | 195 | 5 |
| payroll | 1500 | 252 | 7 |
| reporting | 384 | 118 | 4 |
| sales | 820 | 254 | 7 |
| support | 617 | 192 | 5 |
| timesheet | 460 | 164 | 4 |

## hard_eval probe dağılımı

| probe | ne ölçer | adet |
|---|---|---:|
| `P1_unseen_tool` | eğitimde hiç görülmemiş tool'a doğru çağrı | 190 |
| `P2_seen_intent_new_tool` | bilinen senaryo kalıbı + yeni tool | 73 |
| `P3_category_new_surface` | bilinen kategori + havuz-dışı doğal dil | 100 |
| `P4_same_kw_diff_tool` | aynı kelime → doğru tool ayrımı | 110 |
| `P5_same_tool_new_phrasing` | bilinen tool + yepyeni ifade | 200 |
| `P6_large_candidate_set` | 42–62 aday arasından seçim (kalabalık katalog) | 59 |
| `P7_cannot_answer` | uygun tool yok → kibar ret | 142 |
| `P8_clarification` | çelişkili parametre → netleştirme | 54 |
| `P9_tool_result` | görülmemiş tool sonucunu yorumlama | 75 |

## Eğitim sonrası ölçülecek metrikler (hard_eval üzerinde)

| metrik | nasıl |
|---|---|
| 4-karar doğruluğu | `meta.decision` vs model kararı, 4×4 matris |
| tool-selection top-1 / top-3 | `meta.target_tools[0]` vs model; P1/P5 ayrı raporla |
| argüman tam-eşleşme | tool_call arguments == altın (P1/P6) |
| halüsinasyon oranı | model argümanı kullanıcı/tool metninden izlenemiyorsa |
| yetkisiz WRITE | onay turu olmadan write/action tool_call sayısı (0 olmalı) |
| clarification doğruluğu | P8: model netleştirme sordu mu |
| tool-result özet doğruluğu | P9: yanıt yalnız sonuca dayanıyor mu, sayı uydurma yok |
| **genelleme farkı** | top-1(P5, seen tool) − top-1(P1, unseen tool); küçük fark = policy taşınıyor |
