"""Schema-driven tool-calling dataset generator (policy-level rebuild).

Bu paket, ~100 tool ölçeğinde genellenebilir bir tool-calling policy öğreten
sentetik veri üretir. Üretim TOOL ŞEMASINDAN türetilir; per-tool elle şablon YOKTUR.
Modüller:
    resolve   - takvim / göreli tarih çözümleme (üretici + validator paylaşır)
    synth     - prosedürel parametre/deger sentezi (kanonik + yüzey)
    frames    - tool-agnostik cümle kalıpları (kullanıcı + asistan)
    catalog   - ~100 tool'luk şema kataloğu + train/val/test bölmesi
    catalog_index - katalog yükleme, şema-benzerliği, aday-liste kurma
    scenarios - karar/senaryo üreticileri (schema-driven)
    generate  - orkestrasyon, split, çıktı
Tümü stdlib-only ve deterministiktir (seed).
"""
