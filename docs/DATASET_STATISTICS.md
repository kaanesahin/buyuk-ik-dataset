# Büyük İK v2 — Dataset İstatistikleri

> Otomatik üretilir: `python scripts/metrics.py`. Kaynak: `data/`.

## Temel sayılar

| ölçüt | değer |
|---|---|
| train örnek | **15000** |
| val örnek | **2000** (val_seen + val_unseen_tool) |
| hard_eval örnek | **338** |
| toplam tool (katalog) | **105** |
| train_tools / val_tools / test_tools | **75 / 15 / 15** |
| domain sayısı | **13** (calendar, crm, documents, finance, hr, inventory, it_support, logistics, payroll, reporting, sales, support, timesheet) |
| read / write / action | **66 / 35 / 4** |
| train hedef-tool sayısı | 75 / 75 |
| tool başına örnek (min/medyan/maks) | 102 / 145 / 367 |

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
| `write_confirm` | 975 | 6.5% |
| `write_execute` | 975 | 6.5% |
| `multi_parallel` | 600 | 4.0% |
| `write_chain` | 525 | 3.5% |
| `multi_sequential` | 450 | 3.0% |
| `hn_keyword_ambiguous` | 450 | 3.0% |
| `hn_tool_absent` | 375 | 2.5% |
| `hn_conflict` | 300 | 2.0% |
| `hn_user_names_wrong_tool` | 300 | 2.0% |

## Tool çağrı yapısı

| ölçüt | adet | oran |
|---|---:|---:|
| tek-tool tool_call | 7425 | 49.5% |
| çoklu-tool (paralel + sıralı) | 1050 | 7.0% |
|  — bunun sıralısı (sonuç→param) | 450 | 3.0% |
| **tool-sonucu turu içeren** | 3903 | 26.0% |
|  — tool_call örnekleri içinde | 3903 | 46.1% |
| tool-sonucu modu | {'ok': 1488, 'empty': 503, 'error': 413, None: 1189, 'partial': 310} |  |
| WRITE/action örneği | 3135 | 20.9% |
| onay akışı (confirm) | 2475 | 16.5% |
| 6-turlu zincir (eksik→onay→uygula) | 525 | 3.5% |
| tur dağılımı | {2: 9129, 4: 4157, 5: 285, 6: 1429} |  |

## Parametre davranışı

| ölçüt | adet | oran |
|---|---:|---:|
| opsiyonel parametre KULLANILAN örnek | 1051 | 7.0% |
| eksik-parametre (request_for_info) | 2250 | 15.0% |

## Aday tool listesi boyutu (train)

| kova | adet | oran | hedef |
|---|---:|---:|---:|
| ≤12 tool | 5129 | 34.2% | ~28% |
| 13–34 tool | 7594 | 50.6% | ~56% |
| 35–58 tool | 2277 | 15.2% | ~16% |
| medyan / p90 / maks | 19 / 43 / 58 |  |  |
| hedef tool'un liste-içi konumu (ort., 0=baş 1=son) | 0.49 (uniform ~0.50) |

## Hard-negative örnekleri (train)

Toplam **1425** (9.5%).

| tür | adet |
|---|---:|
| A_keyword_ambiguous | 450 |
| F_tool_absent | 375 |
| E_conflict | 300 |
| D_user_names_wrong_tool | 300 |

## Doğal dil çeşitliliği (train)

| register | adet | oran |
|---|---:|---:|
| `plain` | 3625 | 24.2% |
| `formal` | 3043 | 20.3% |
| `chat` | 2876 | 19.2% |
| `long` | 2552 | 17.0% |
| `typo` | 2078 | 13.9% |
| `short` | 826 | 5.5% |

- benzersiz ilk-kullanıcı-turu (folded): **14493 / 15000** (96.6%)
- **ayırt edici yüzey kelimesi → tool korelasyonu: 35%** (K-1; hedef < 55%; eski sürüm ~97%)
  - en yüksek: apply_discount_approval 84%, create_contact 83%, approve_expense 80%, check_service_status 79%, report_damage 78%, get_expense_status 78%

## Sızıntı ve tekrar

- train↔val kullanıcı-turu imza kesişimi: **0**
- train hedefinde val/test tool'u: **0**
- val_unseen_tool hedefleri (hepsi split=val): **15/15**
- hard_eval P1/P9 hedefleri (hepsi split=test): **14/14**

## Tool kapsama

- 75 train tool'unun tamamı hedef: **EVET**
- dağılım eğriliği (maks/min): **3.6×**

| domain | tool_call | request_for_info | (train tool) |
|---|---:|---:|---:|
| calendar | 504 | 231 | 5 |
| crm | 1290 | 253 | 7 |
| documents | 473 | 225 | 5 |
| finance | 876 | 257 | 7 |
| hr | 996 | 283 | 7 |
| inventory | 532 | 154 | 5 |
| it_support | 533 | 402 | 7 |
| logistics | 513 | 200 | 5 |
| payroll | 1506 | 251 | 7 |
| reporting | 394 | 121 | 4 |
| sales | 816 | 259 | 7 |
| support | 632 | 198 | 5 |
| timesheet | 460 | 166 | 4 |

## hard_eval probe dağılımı

| probe | ne ölçer | adet |
|---|---|---:|
| `P1_unseen_tool` | eğitimde hiç görülmemiş tool'a doğru çağrı | 95 |
| `P2_seen_intent_new_tool` | bilinen senaryo kalıbı + yeni tool | 30 |
| `P3_category_new_surface` | bilinen kategori + havuz-dışı doğal dil | 35 |
| `P4_same_kw_diff_tool` | aynı kelime → doğru tool ayrımı | 21 |
| `P5_same_tool_new_phrasing` | bilinen tool + yepyeni ifade | 35 |
| `P6_large_candidate_set` | 36–58 aday arasından seçim | 40 |
| `P7_cannot_answer` | uygun tool yok → kibar ret | 43 |
| `P8_clarification` | çelişkili parametre → netleştirme | 15 |
| `P9_tool_result` | görülmemiş tool sonucunu yorumlama | 24 |

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
