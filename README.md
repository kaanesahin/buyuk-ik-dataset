# Büyük İK — Şema-Güdümlü Tool-Calling Policy Dataset (v2)

~105 tool / 13 domain üzerinde **genellenebilir bir tool-calling policy** öğreten
sentetik Türkçe veri seti. Amaç modele "hangi cümlede hangi tool" ezberletmek
değil; **niyeti anla → tool şemasını oku → doğru tool'u seç → parametreleri çıkar
→ doğru formatta çağır → sonucu oku → cevapla** akışını, herhangi bir tool'a
uygulanabilecek şekilde öğretmektir.

> v1 (22 tool, per-tool elle şablon, 3000 örnek) `legacy/` altındadır. v2'nin
> neden ve nasıl üretildiği: `docs/REVISION_REPORT.md`, `docs/FINAL_POLICY_ASSESSMENT.md`.

**Güncel sürüm:** 15.000 train · 2.000 val · ~330 hard_eval · 105 tool
(train 75 / val 15 / test 15) · doğrulama **0 hata** · `tests/` 33 pytest.

---

## Depo düzeni

```
buyuk_ik_lora_dataset/
├── catalog/
│   └── catalog.py            105 tool'luk ŞEMA KATALOĞU — üretimin tek kaynağı.
│                             Yeni tool = buraya bir T(...) satırı.
├── scripts/
│   ├── gen/
│   │   ├── resolve.py        takvim / göreli-tarih çözümleme (üretici+validator paylaşır)
│   │   ├── synth.py          prosedürel parametre/deger sentezi (kanonik + yüzey)
│   │   ├── frames.py         tool-agnostik cümle kalıpları + dil-kaydı stili
│   │   ├── catalog_index.py  şema-benzerliği + aday-liste kurma + direct/cannot havuzları
│   │   └── scenarios.py      senaryo üreticileri (hepsi şemadan türetir)
│   ├── generate_dataset.py   orkestrasyon → data/  (train + val)
│   ├── build_hard_eval.py    → data/tool_calling_hard_eval.jsonl (P1..P9 probe)
│   ├── validate_dataset.py   bağımsız QC: 0 hata beklenir
│   ├── metrics.py            → docs/DATASET_STATISTICS.md
│   └── build_training_variants.py   opsiyonel: system-turlu ChatML kopya (gitignore)
├── data/
│   ├── tool_calling_train.jsonl        satır başına {"tools":[...], "messages":[...]}
│   ├── tool_calling_val.jsonl          val_seen_tool + val_unseen_tool
│   ├── tool_calling_hard_eval.jsonl    test tool'ları + policy probe'ları
│   ├── tool_calling_*.meta.jsonl       aynı sırada; QC/eval alanları (eğitimde KULLANILMAZ)
│   ├── tools_{all,train,val,test}.json 105 / 75 / 15 / 15 tool şeması
│   └── tool_splits.json                tool -> {domain, cat, split}
├── tests/                    pytest paketi (33 test) — tests/README.md
├── docs/
│   ├── DATASET_STATISTICS.md          (metrics.py üretir)
│   ├── validation_report.md           (validate_dataset.py üretir)
│   ├── REVISION_REPORT.md             v1→v2 değişiklik günlüğü (K-1..K-10 / D-1..D-10)
│   ├── FINAL_POLICY_ASSESSMENT.md     12-bölüm değerlendirme + karar
│   ├── POLICY_UYGUNLUK_RAPORU.md      v1 policy incelemesi (revizyon kaynağı)
│   └── FINETUNE_UYGUNLUK_RAPORU.md    v1 yapısal inceleme
└── legacy/                   v1 pipeline (22 tool) — arşiv
```

---

## Dört karar davranışı

| decision | ne zaman | assistant çıktısı |
|---|---|---|
| `direct` | tool gerekmiyor (tanım, politika, selam) | doğrudan yanıt |
| `tool_call` | tool gerekli **ve** tüm zorunlu parametreler mevcut | `<tool_call>…</tool_call>` (+ gerekiyorsa tool sonucu → NL cevap) |
| `request_for_info` | zorunlu bilgi eksik **veya** WRITE için onay gerek **veya** çelişkili parametre | eksik bilgiyi / onayı / netleştirmeyi isteyen soru |
| `cannot_answer` | uygun tool yok (kapsam dışı, gizlilik, gelecek, yetki) | kibar ret + gerekçe |

Politika tool-agnostiktir: aynı kurallar 100 tool için de aynı şekilde uygulanır.

---

## Format

`messages` = `user` / `assistant` / `tool` turları. Tool tanımları ayrı `tools`
alanında — **Qwen 2.5 native biçimi** (`apply_chat_template(messages, tools=tools, …)`).

Tool çağrısı (tek satır JSON, blok tek başına):
```
<tool_call>
{"name": "hr_get_leave_balance", "arguments": {"employee_id": "EMP-1042", "leave_type": "annual"}}
</tool_call>
```
Tool sonucu:
```
{"role": "tool", "content": "{\"annual_left\": 12, \"excuse_left\": 3}"}
```
Ardından assistant **yalnız o sonuca dayanarak** Türkçe yanıt verir (uydurma yok).

Çoklu çağrı = art arda `<tool_call>` blokları. Sıralı zincirde bir tool sonucundaki
kimlik, sonraki tool'un parametresi olabilir.

---

## Kullanım

```bash
python scripts/generate_dataset.py          # -> data/ (n=15000, seed=20260901)
python scripts/build_hard_eval.py           # -> data/tool_calling_hard_eval.jsonl
python scripts/validate_dataset.py          # 0 hata beklenir
python scripts/metrics.py                    # -> docs/DATASET_STATISTICS.md
pip install pytest && pytest tests/          # 33 test

python scripts/build_training_variants.py    # opsiyonel: system-turlu kopya
```

Bayraklar: `--n`, `--seed`, `--today`, `--val-seen`, `--val-unseen`, `--out`, `--dry-run`.
Üretici belirlenimcidir: aynı seed → byte-aynı çıktı.

---

## Yeni tool eklemek

`catalog/catalog.py` içindeki `_add(...)` bloklarından birine bir satır:

```python
T("crm_get_lead_score", "crm", "read",
  "Bir aday müşterinin (lead) güncel skor ve etkenlerini getirir.",
  obj="lead skorunu", obj_nom="lead skoru", kw=["lead", "skor", "aday"],
  verbs=["getir", "göster", "hesapla"],
  params=[ID("lead_id", "LEAD", "lead", True)],
  result=[("score", "pct"), ("stage", "enum")],
  syn=["o adayın ne kadar sıcak olduğu", "bu kişi ne kadar dönüşür"]),
```

Üretici bu tool için otomatik olarak: read çağrıları, eksik-parametre soruları,
tool-sonucu turları, çeldirici listeleri ve hard-negative örnekleri üretir.
`assign_splits()` onu domain-stratifiye biçimde train/val/test'ten birine koyar.
**Per-tool cümle şablonu yazmak GEREKMEZ.**

---

## Değerlendirme

Eğitimden sonra `data/tool_calling_hard_eval.jsonl` üzerinde:

| metrik | kaynak |
|---|---|
| 4-karar doğruluğu (4×4) | `meta.decision` |
| tool-selection top-1 / top-3 | `meta.target_tools` — **P1 (görülmemiş tool) ve P5 (görülen tool) AYRI** |
| argüman tam-eşleşme | `meta` altın argümanlar (P1/P6) |
| halüsinasyon oranı | `validate_dataset.trace_value` |
| yetkisiz WRITE (=0 olmalı) | onay turu olmadan write/action çağrısı |
| clarification / tool-result özet doğruluğu | P8 / P9 |
| **genelleme farkı** | top-1(P1) vs top-1(P5); ≤ ~10 puan hedef |

Ayrıntı: `docs/FINAL_POLICY_ASSESSMENT.md`.
