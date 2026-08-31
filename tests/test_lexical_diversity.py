# -*- coding: utf-8 -*-
"""
test_lexical_diversity.py — SÖZCÜKSEL/BİLGİ-KURAMSAL çeşitlilik (§7, §28, §29, §32)
============================================================================

`test_diversity_and_leakage.py` imza-tabanlı kopya arar. Bu dosya metnin
*bilgi içeriğini* ölçer: Guiraud kök-TTR'si, ilk-kelime entropisi, n-gram
yeniliği, cevap havuzu yoğunlaşması. Amaç: dataset yapay/şablonik görünmemeli
(§29) ve val, train'in n-gram uzayında yeni yüzeyler getirmeli.

Ayrıca **şablon sızıntısı** taraması: `{emp}`, `{tip_disp}`, `None`, `][`,
çözülmemiş slot adları hiçbir metinde bulunmamalı.

Not: Bu bir *domain* korpusudur (İK). Ham TTR düşük çıkar (tekrar eden alan
sözcükleri); bu yüzden uzunluğa-dayanıklı **Guiraud R = V/√N** kullanılır.

Kapsam
------
* Kullanıcı turları: Guiraud kök-TTR ≥ 5.0; sözcük dağarcığı ≥ 1000 tür.
* İlk-kelime dağılımının Shannon entropisi ≥ 4.5 bit; ≥ 150 farklı açılış kelimesi.
* Bigram yeniliği: val bigramlarının ≥ %8'i train'de HİÇ geçmiyor.
* Trigram yeniliği: val trigramlarının ≥ %20'si train'de yeni.
* `direct` cevap havuzu: ≥ 100 farklı cevap; hiçbir tekil cevap ≤ %5.
* `request_for_info` / `cannot_answer` son turları: ≥ 200 farklı metin; en sık
  kalıp (ilk 6 kelime) ≤ %12.
* Şablon artefaktı yok: `{...}`, `None`, `][`, çift boşluk, sızmış slot adı.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import pytest

from conftest import fold, strip_tool_calls, user_turns

WORD_RE = re.compile(r"[a-zçğıöşü]+", re.IGNORECASE)
# Not: `\s` yerine düz boşluk — çok-tool assistant turlarındaki ' \n ' yanlış pozitif vermesin.
TEMPLATE_ARTIFACT_RE = re.compile(r"\{[a-z_]+\}|\bNone\b|\]\[|(?<!\d) {2,}(?!\d)")

SLOT_TOKENS = {
    "emp_canon", "tip_disp", "tip_canon", "donem_canon", "donemy_canon", "range_b",
    "range_e", "mrange_b", "mrange_e", "dept_canon", "tur_canon", "kaynak_canon",
    "bkind", "bkind_disp", "ckind", "cval", "bsurf", "bval", "mrange",
}


def words(text: str) -> list[str]:
    return WORD_RE.findall(fold(text))


def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    return -sum((c / total) * math.log2(c / total) for c in counter.values()) if total else 0.0


def ngrams(tokens: list[str], n: int) -> set[tuple]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


@pytest.fixture(scope="session")
def user_tokens(all_records):
    return [w for rec in all_records for u in user_turns(rec["messages"]) for w in words(u)]


# --------------------------------------------------------------------------
# Guiraud kök-TTR ve dağarcık
# --------------------------------------------------------------------------

def test_guiraud_root_ttr(user_tokens):
    v, n = len(set(user_tokens)), len(user_tokens)
    r = v / math.sqrt(n)
    assert r >= 5.0, f"Guiraud R = {r:.2f} (< 5.0) — sözcüksel çeşitlilik zayıf (V={v}, N={n})"


def test_vocabulary_size(user_tokens):
    v = len(set(user_tokens))
    assert v >= 1000, f"kullanıcı sözcük dağarcığı yalnızca {v} tür (< 1000)"


# --------------------------------------------------------------------------
# Entropi
# --------------------------------------------------------------------------

def test_opening_word_entropy_and_spread(all_records):
    firsts = Counter()
    for rec in all_records:
        w = words(user_turns(rec["messages"])[0])
        if w:
            firsts[w[0]] += 1
    h = shannon_entropy(firsts)
    assert len(firsts) >= 150, f"yalnızca {len(firsts)} farklı açılış kelimesi (< 150)"
    assert h >= 4.5, f"ilk-kelime entropisi {h:.2f} bit (< 4.5)"


# --------------------------------------------------------------------------
# n-gram yeniliği
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def split_ngrams(train_records, val_records):
    def collect(records, n):
        acc: set[tuple] = set()
        for rec in records:
            for u in user_turns(rec["messages"]):
                acc |= ngrams(words(u), n)
        return acc
    return {
        2: (collect(train_records, 2), collect(val_records, 2)),
        3: (collect(train_records, 3), collect(val_records, 3)),
    }


def test_val_introduces_novel_bigrams(split_ngrams):
    tr, va = split_ngrams[2]
    ratio = len(va - tr) / max(len(va), 1)
    assert ratio >= 0.08, f"val bigram yeniliği %{100*ratio:.1f} (< %8) — val, train'in yüzey kopyası"


def test_val_introduces_novel_trigrams(split_ngrams):
    tr, va = split_ngrams[3]
    ratio = len(va - tr) / max(len(va), 1)
    assert ratio >= 0.20, f"val trigram yeniliği %{100*ratio:.1f} (< %20)"


# --------------------------------------------------------------------------
# Cevap havuzu yoğunlaşması
# --------------------------------------------------------------------------

def test_direct_answer_pool_is_wide_and_flat(paired):
    answers = [
        next(m["content"] for m in rec["messages"] if m["role"] == "assistant")
        for rec, meta in paired if meta["decision"] == "direct"
    ]
    counts = Counter(answers)
    assert len(counts) >= 100, f"yalnızca {len(counts)} farklı direct cevabı (< 100)"
    top_share = counts.most_common(1)[0][1] / len(answers)
    assert top_share <= 0.05, f"tek bir direct cevabı payı %{100*top_share:.1f} (> %5)"


@pytest.mark.parametrize("decision", ["request_for_info", "cannot_answer"])
def test_terminal_turn_pool_is_wide(paired, decision):
    lasts = [rec["messages"][-1]["content"] for rec, meta in paired if meta["decision"] == decision]
    assert len(set(lasts)) >= 200, f"'{decision}' son turlarında yalnızca {len(set(lasts))} farklı metin"
    prefixes = Counter(" ".join(fold(t).split()[:6]) for t in lasts)
    top = prefixes.most_common(1)[0][1] / len(lasts)
    assert top <= 0.12, f"'{decision}' retlerinin %{100*top:.0f}'i aynı 6-kelime kalıbıyla başlıyor"


# --------------------------------------------------------------------------
# Şablon artefaktı
# --------------------------------------------------------------------------

def test_no_unrendered_template_artifacts(all_records):
    failures = []
    for i, rec in enumerate(all_records):
        for m in rec["messages"]:
            text = strip_tool_calls(m["content"]) if m["role"] == "assistant" else m["content"]
            hit = TEMPLATE_ARTIFACT_RE.search(text)
            if hit:
                ctx = text[max(0, hit.start() - 25):hit.end() + 15]
                failures.append(f"kayıt {i} ({m['role']}): {hit.group(0)!r} … {ctx!r}")
    assert not failures, "Şablon/format artefaktları:\n  " + "\n  ".join(failures[:30])


def test_no_literal_slot_names_leaked(all_records):
    for i, rec in enumerate(all_records):
        blob = " ".join(m["content"] for m in rec["messages"])
        leaked = {s for s in SLOT_TOKENS if re.search(rf"\b{re.escape(s)}\b", blob)}
        assert not leaked, f"kayıt {i}: sızmış slot adı/ları {leaked}"
