# preview/ — ne nerede

| dosya | içerik |
|---|---|
| `DATASET_PREVIEW.md` | Karar sınıfına göre gruplanmış sohbet dökümleri — **buradan başlayın** |
| `samples/tool_call.sample.json` | `tool_call` — girintili tam kayıtlar (`tools`+`messages`) |
| `samples/direct.sample.json` | `direct` — girintili tam kayıtlar |
| `samples/request_for_info.sample.json` | `request_for_info` — girintili tam kayıtlar |
| `samples/cannot_answer.sample.json` | `cannot_answer` — girintili tam kayıtlar |

Hepsi `scripts/make_preview.py` ile `data/` üzerinden üretilir; elle düzenlemeyin.


## Domain

| değer | adet | oran |
|---|---:|---:|
| `ik_islemleri` | 916 | %30.5 |
| `maas_finans` | 625 | %20.8 |
| `izin_yonetimi` | 495 | %16.5 |
| `puantaj` | 315 | %10.5 |
| `organizasyon` | 256 | %8.5 |
| `calisan_bilgileri` | 191 | %6.4 |
| `kapsanmayan` | 125 | %4.2 |
| `meta` | 77 | %2.6 |

## Difficulty

| değer | adet | oran |
|---|---:|---:|
| `orta` | 1276 | %42.5 |
| `zor` | 869 | %29.0 |
| `kolay` | 467 | %15.6 |
| `cok_zor` | 388 | %12.9 |

## Register

| değer | adet | oran |
|---|---:|---:|
| `gundelik` | 748 | %24.9 |
| `resmi` | 666 | %22.2 |
| `konusma_dili` | 550 | %18.3 |
| `uzun` | 549 | %18.3 |
| `yazim_hatali` | 343 | %11.4 |
| `kisa` | 144 | %4.8 |

