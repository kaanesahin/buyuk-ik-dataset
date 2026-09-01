# Büyük İK tool-calling dataset — üretim raporu

- Üretim tarihi bağlamı (today): `2026-08-27`
- Seed: `20260827`  |  Hedef N: `3000`  |  Üretilen: **3000**
- Train / Val: **2708 / 292**  (val oranı ~0.1)
- Benzersiz kullanıcı-mesajı imzası: **3000 / 3000**  (%100.0)
- Çok turlu örnek: **486**  (bunun 99'i 6-turlu 'topla→onay→uygula' zinciri)  |  Çoklu-tool örnek: **63**  |  WRITE örneği: **690**
- Tur dağılımı: 2 tur: 2514, 4 tur: 387, 6 tur: 99

## Karar dağılımı (hedef vs gerçekleşen)

| decision | hedef | gerçekleşen | oran |
|---|---|---|---|
| tool_call | %30 | 900 | %30.0 |
| direct | %25 | 750 | %25.0 |
| request_for_info | %25 | 750 | %25.0 |
| cannot_answer | %20 | 600 | %20.0 |

## Domain dağılımı

| değer | adet | oran |
|---|---|---|
| ik_islemleri | 916 | %30.5 |
| maas_finans | 625 | %20.8 |
| izin_yonetimi | 495 | %16.5 |
| puantaj | 315 | %10.5 |
| organizasyon | 256 | %8.5 |
| calisan_bilgileri | 191 | %6.4 |
| kapsanmayan | 125 | %4.2 |
| meta | 77 | %2.6 |

## Zorluk dağılımı

| değer | adet | oran |
|---|---|---|
| orta | 1276 | %42.5 |
| zor | 869 | %29.0 |
| kolay | 467 | %15.6 |
| cok_zor | 388 | %12.9 |

## Register (dil kaydı) dağılımı

| değer | adet | oran |
|---|---|---|
| gundelik | 748 | %24.9 |
| resmi | 666 | %22.2 |
| konusma_dili | 550 | %18.3 |
| uzun | 549 | %18.3 |
| yazim_hatali | 343 | %11.4 |
| kisa | 144 | %4.8 |

## Tool çağrı histogramı (assistant çıktısında)

| değer | adet | oran |
|---|---|---|
| create_izin_talebi | 87 | %2.9 |
| get_izin_bakiyesi | 82 | %2.7 |
| get_bordro | 70 | %2.3 |
| get_maas_bilgisi | 67 | %2.2 |
| get_puantaj | 54 | %1.8 |
| update_izin_talebi | 54 | %1.8 |
| get_mesai_bilgisi | 54 | %1.8 |
| get_prim_bilgisi | 53 | %1.8 |
| get_employee_info | 52 | %1.7 |
| get_yonetici_bilgisi | 48 | %1.6 |
| get_izin_gecmisi | 46 | %1.5 |
| get_calisan_listesi | 38 | %1.3 |
| get_departman_bilgisi | 36 | %1.2 |
| get_employee_status | 32 | %1.1 |
| get_yan_haklar | 32 | %1.1 |
| get_izin_talebi_durumu | 30 | %1.0 |
| check_employee_access | 26 | %0.9 |
| update_employee_information | 21 | %0.7 |
| create_pozisyon_degisiklik_talebi | 21 | %0.7 |
| update_employee_contact | 20 | %0.7 |
| cancel_izin_talebi | 20 | %0.7 |
| create_ucret_degisiklik_talebi | 20 | %0.7 |

## Notlar

- Eğitim dosyası (`*_train.jsonl`) yalnızca `tools` + `messages` içerir.
- `*_train.meta.jsonl` satırları eğitim dosyasıyla AYNI SIRADADIR; QC ve değerlendirme içindir.
- Tüm çalışan / ID / maaş / tarih bilgileri sentetiktir.
