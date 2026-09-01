# Büyük İK v2 — Eğitim verimliliği (kayıp yoğunluğu / K-3)

> Bu bir **veri hatası değil**, yapısal bir özellik — ama LoRA maliyetini
> doğrudan etkilediği için burada açıkça belgelenir ve iki somut hafifletme
> yolu verilir. Kaynak ölçüm: `data/tool_calling_train.jsonl`.

---

## 1. Sorun: kayıp yoğunluğu ~%1.8

Her eğitim dizisi şu yapıda:

```
[tools: 105 tool'dan seçili aday listesi — JSON şema]   ~8.000 karakter  (~%97)
user / assistant / tool mesajları                        ~150–400 karakter
```

Kayıp yalnız **assistant** turlarında hesaplanır. Yani işlenen her dizinin
~%98'i (aday tool şema bağlamı) ileri/geri geçişten geçer ama **gradyan sinyali
üretmez**.

| ölçüt (n=1500, karakter-proxy) | değer |
|---|---|
| ortalama aday-şema (JSON) | ~7.960 krk/kayıt |
| ortalama assistant içeriği | ~150 krk/kayıt |
| **assistant / toplam** | **~%1.8** |

Pratik sonuç: 1 epoch'ta işlenen ~N token'ın yalnız ~%1.8'i öğrenme sinyali —
kabaca **~55× hesap yükü** her gradyan-sinyali token başına. `train.jsonl`'in
~%95'i 105 şemanın tekrar tekrar yazılmasıdır (katalog tek kopya ~38 KB).

Bu **bilinçli** bir tasarım: "kalabalık katalog içinden doğru tool'u seç"
sinyalinin gerçekçi olması için aday listeler büyük ve değişken tutulur
(bkz. K-3, `catalog_index.candidate_list`). Ama ısıtma/ilk epoch'larda bu
yükü ödemeye gerek yok.

---

## 2. Hafifletme A — sequence packing (trainer tarafı, ÖNERİLEN)

Mesajlar dizinin çok küçük bir kısmı olduğundan, packing yapılmadan **dolgu
(padding) token'larına** çok kayıp verilir. Packing bunu ortadan kaldırır ve
adım başına gerçek içeriği artırır.

- **TRL `SFTTrainer`**: `packing=True` (+ `max_seq_length` ≥ 16384). FA2 varlen
  ile blok-diyagonal dikkat otomatik.
- **LLaMA-Factory**: `cutoff_len: 16384`, `packing: true`, `neat_packing: true`
  (örnekler arası dikkat sızıntısını engeller).
- **axolotl**: `sample_packing: true`, `pad_to_sequence_len: true`,
  `sequence_len: 16384`.

Packing kayıp *yoğunluğunu* birebir değiştirmez (aynı token'lar), ama dolgu
israfını sıfırlar ve adım başına ~3–5× daha fazla örnek sığdırır → wall-clock
ve $ maliyeti belirgin düşer.

---

## 3. Hafifletme B — küçük-aday-liste curriculum (bu repo'dan)

`scripts/build_training_variants.py --max-candidates N` her kaydın `tools`
listesini **çağrılan tool(lar) + rastgele N-e-kadar çeldirici** olacak şekilde
kırpar. Altın cevap (hedef tool ve argümanları) **değişmez**; yalnız bağlam
kısalır.

```bash
python scripts/build_training_variants.py --max-candidates 10
# -> data/variants/tool_calling_{train,val,hard_eval}.cand10.jsonl
```

| dosya | kayıp yoğunluğu | ort. boyut |
|---|---|---|
| `tool_calling_train.jsonl` (kanonik) | ~%1.8 | ~8.070 krk/kayıt |
| `…train.cand10.jsonl` | **~%4.0** | ~3.710 krk/kayıt (~2.2× kısa) |

**Önerilen reçete:**

1. 1–2 **ısıtma** epoch'u `…cand10.jsonl` üzerinde (ucuz; tool-seçimi + argüman
   çıkarımı + 4-karar politikası hızlı oturur).
2. Kalan epoch'lar **kanonik** `tool_calling_train.jsonl` üzerinde (tam çeldirici
   baskısı — "kalabalık katalogdan seç" ve "aday listede tool var ≠ çağır"
   sinyalleri burada).
3. `hard_eval` (tam aday listeleriyle) her zaman değerlendirmede kullanılır —
   curriculum dosyası eğitimde, kanonik hard_eval ölçümde.

> Not: `data/variants/` **gitignore**'da; deterministik olarak yeniden üretilir.
> Curriculum yalnız ısıtma içindir — tek başına eğitilirse "az çeldiricili kolay
> seçim" öğrenilir ve K-3'ün amacı kaybolur.

---

## 4. max_seq_len

Tam katalog ~11k token. 35–58 tool'lu örneklerde dizi ~6–9k token.
**`max_seq_len ≥ 16384`** önerilir (packing ile birlikte). `cand10` curriculum
dosyasında örnekler ~2–3k token → ısıtmada daha küçük `max_seq_len` (4k) yeterli.

---

## 5. Özet

| yol | ne kazandırır | maliyet |
|---|---|---|
| sequence packing | dolgu israfı 0; adım başına ~3–5× örnek | trainer config (1 satır) |
| cand10 curriculum ısıtma | kayıp yoğunluğu ~%1.8 → ~%4; dizi ~2.2× kısa | 1 komut + 2-aşamalı eğitim |
| ikisi birlikte | ısıtma çok ucuz, ana eğitim gerçekçi | — |

Hiçbiri kanonik veriyi değiştirmez; ikisi de opsiyoneldir. Değiştirmeden tam
`tool_calling_train.jsonl` üzerinde eğitmek de çalışır — sadece daha pahalıdır.
