# Büyük İK v2 — doğrulama raporu

- train 15000 | val 2000
- HATA 0 | UYARI 0

## Bilgi
```
hard_eval: 1003 örnek doğrulandı
aşırı-benzer (collapsed) kayıt: 40 / 17000 (%0.24)
train distinct target tools: 75 / 75
  per-tool örnek: min 101 / medyan 148 / max 364
train decision: {'tool_call': 8475, 'direct': 1650, 'cannot_answer': 1875, 'request_for_info': 3000}
aday-liste kovaları: ≤12 %34 | 13-34 %52 | 35+ %14 (medyan 19, maks 58)
tool-result turu: 3908/15000 (%26) | tool_call içinde %46
K-1 yüzey→tool korelasyonu: nesne ana-adı %53 (dürüst) | en nadir ayırt edici token %33 (alt sınır) | ana-ad en yüksek: find_free_slot %97, create_expense_report %90, create_leave_request %89, create_contact %84, get_item %80, export_dataset %79, log_interaction %77, request_leave_of_absence %74
register: {'typo': 1645, 'formal': 3164, 'long': 2702, 'plain': 3521, 'chat': 2994, 'short': 974}
val eval_kind: {'val_seen_tool': 1000, 'val_unseen_tool': 1000}
```
