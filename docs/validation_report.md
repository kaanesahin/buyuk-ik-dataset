# Büyük İK v2 — doğrulama raporu

- train 15000 | val 2000
- HATA 0 | UYARI 0

## Bilgi
```
hard_eval: 338 örnek doğrulandı
aşırı-benzer (collapsed) kayıt: 15 / 17000 (%0.09)
train distinct target tools: 75 / 75
  per-tool örnek: min 102 / medyan 145 / max 367
train decision: {'tool_call': 8475, 'request_for_info': 3000, 'direct': 1650, 'cannot_answer': 1875}
aday-liste kovaları: ≤12 %34 | 13-34 %51 | 35+ %15 (medyan 19, maks 58)
tool-result turu: 3903/15000 (%26) | tool_call içinde %46
keyword->tool korelasyonu: genel %35 (hedef < %55) | en yüksek: apply_discount_approval %84, create_contact %83, approve_expense %80, check_service_status %79, report_damage %78, get_expense_status %78, list_folder %71, list_open_cases %69
register: {'chat': 2876, 'typo': 2078, 'formal': 3043, 'long': 2552, 'plain': 3625, 'short': 826}
val eval_kind: {'val_unseen_tool': 1000, 'val_seen_tool': 1000}
```
