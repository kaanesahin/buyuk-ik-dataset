# Büyük İK v2 — test paketi

`data/` altındaki üretilmiş veri setini **artefakt olarak** doğrulayan pytest
paketi (33 test). `scripts/validate_dataset.py` hızlı QC kapısıyken bu paket her
niteliği isimlendirilmiş, açıklayıcı testlerle kontrol eder.

```bash
pip install pytest
pytest tests/                  # tümü (~15 sn)
pytest tests/ -m "not slow"    # üretici alt süreçlerini atla
```

| dosya | ne doğrular |
|---|---|
| `test_structure_and_schema.py` | JSONL/rol yapısı, `tool` mesajı akışı, katı `<tool_call>` biçimi, şema uyumu (enum/zorunlu/bilinmeyen arg), UTF-8/LF hijyeni |
| `test_no_hallucination.py` | her tool_call argümanı kullanıcı turundan **veya** önceki tool sonucundan izlenebilir; tool-sonrası yanıtta kaynak-dışı sayı yok |
| `test_tool_holdout_and_policy.py` | **val/test tool'u train'de hedef değil**; val_unseen→val tool, hard_eval P1/P9→test tool; holdout tool'lar çeldirici olarak görülür; call+ask kararı ifadeden bağımsız; **onaysız WRITE yok**; keyword→tool korelasyonu < %55; aday liste kovaları + hedef konumu uniform |
| `test_coverage_and_diversity.py` | 75 train tool'un tamamı hedef; 13 domain; karar karışımı; tool-sonucu turu payı + 4 mod; multi-tool (paralel+sıralı); hard-negative türleri; opsiyonel parametre; register çeşitliliği; ilk-tur sözcüksel çeşitlilik; `direct` cevap tekrarı; train↔val sızıntısı |
| `test_reproducibility.py` | `slow` — aynı seed → byte-aynı; `validate_dataset` exit 0; **`scenarios.py` tool adına özel şablon içermiyor** (şema-güdümlü ölçeklenme kanıtı) |

`conftest.py` — `train`/`val`/`hard_eval` (+meta), `catalog`, `calls_of`, `fold` fixture'ları.

## Bağımsızlık

Çoğu test yalnız `data/` dosyalarını okur. `test_no_hallucination` ve
`test_reproducibility`, `scripts/validate_dataset.py` + `catalog/` + `scripts/gen/`
modüllerini bilerek kullanır (yüzey→kanonik çözümleme ve şema erişimi için).
`pytest` yalnız bir TEST bağımlılığıdır; üretici (`scripts/`) stdlib-only kalır.
