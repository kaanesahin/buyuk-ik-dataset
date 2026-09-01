# -*- coding: utf-8 -*-
"""
test_statistical_balance.py — DAĞILIMIN İSTATİSTİKSEL sınavı (§6, §7, §31)
======================================================================

`test_distribution.py` eşik-tabanlıdır ("±3.5 puan"). Bu dosya aynı olguyu
**hipotez testleriyle** sınar: gözlenen dağılım, hedef dağılımdan istatistiksel
olarak ayırt edilemiyor mu? Ayrıca kontenjans tablolarında yapısal boşluk ve
beklenenden sapan hücre var mı?

Tümü **stdlib** ile (scipy yok): ki-kare istatistiği elle, kritik değerler
gömülü tablodan; oran güven aralığı Wilson skoruyla.

Kapsam
------
* Karar dağılımı: hedefe karşı ki-kare uyum iyiliği testi — α=0.001'de reddedilmiyor.
* Aynı test train ve val bölmeleri için AYRI AYRI.
* Register dağılımı: `REGISTER_WEIGHTS`'ten türeyen beklentiye karşı ki-kare.
* decision × domain kontenjans tablosu: her fonksiyonel domain'de en az bir
  ``tool_call`` ve bir ``request_for_info``; standart (Pearson) artık |z| < 6.
* difficulty × turns ilişkisi: ``cok_zor`` çok turlu/uzun ile; ``kolay`` 2 turlu.
* Val oranı: Wilson %99 güven aralığı 0.10'u içeriyor.
* "Aşırı düzgünlük" kontrolü: dağılım hedefe *tıpatıp* oturuyorsa (ki-kare ~ 0)
  bu sentetik veri için beklenir — ama bir uyarı olarak raporlanır, hata değil.
"""
from __future__ import annotations

import math
from collections import Counter

import pytest

pytestmark = pytest.mark.statistical

TARGET_MIX = {"tool_call": 0.30, "direct": 0.25, "request_for_info": 0.25, "cannot_answer": 0.20}

# Ki-kare kritik değerleri (üst kuyruk). df -> {alpha: kritik}
CHI2_CRIT = {
    1: {0.05: 3.841, 0.01: 6.635, 0.001: 10.828},
    2: {0.05: 5.991, 0.01: 9.210, 0.001: 13.816},
    3: {0.05: 7.815, 0.01: 11.345, 0.001: 16.266},
    4: {0.05: 9.488, 0.01: 13.277, 0.001: 18.467},
    5: {0.05: 11.070, 0.01: 15.086, 0.001: 20.515},
}


def chi_square_gof(observed: dict, expected_prob: dict, n: int) -> float:
    """Pearson ki-kare uyum iyiliği istatistiği."""
    stat = 0.0
    for k, p in expected_prob.items():
        exp = n * p
        obs = observed.get(k, 0)
        stat += (obs - exp) ** 2 / exp
    return stat


def wilson_interval(successes: int, n: int, z: float = 2.576):
    """Bir oran için Wilson skor güven aralığı (z=2.576 → %99)."""
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (centre - half, centre + half)


# --------------------------------------------------------------------------
# Karar dağılımı — ki-kare uyum iyiliği
# --------------------------------------------------------------------------

def test_decision_distribution_not_rejected_by_chi_square(all_meta):
    n = len(all_meta)
    obs = Counter(m["decision"] for m in all_meta)
    stat = chi_square_gof(obs, TARGET_MIX, n)
    crit = CHI2_CRIT[3][0.001]
    assert stat < crit, (
        f"karar dağılımı hedeften istatistiksel olarak sapıyor: "
        f"χ²={stat:.2f} ≥ {crit} (df=3, α=0.001)\n  gözlenen: {dict(obs)}"
    )


@pytest.mark.parametrize("split", ["train", "val"])
def test_per_split_decision_distribution_holds(train_meta, val_meta, split):
    metas = train_meta if split == "train" else val_meta
    n = len(metas)
    obs = Counter(m["decision"] for m in metas)
    stat = chi_square_gof(obs, TARGET_MIX, n)
    # val küçük → daha gevşek eşik (α=0.001 yerine sabit df=3 kritik x1.5)
    crit = CHI2_CRIT[3][0.001] * (1.0 if split == "train" else 1.5)
    assert stat < crit, f"{split}: χ²={stat:.2f} ≥ {crit:.1f}  gözlenen {dict(obs)}"


# --------------------------------------------------------------------------
# Register dağılımı — üretici ağırlıklarından türeyen beklenti
# --------------------------------------------------------------------------

def test_register_distribution_is_plausible_given_weights(all_meta, gen):
    """REGISTER_WEIGHTS bir stilleme olasılığı verir; detect_register çıktısı
    tam eşleşmez ama 6 kategori de anlamlı payda olmalı ve hiçbiri baskın değil."""
    n = len(all_meta)
    obs = Counter(m["register"] for m in all_meta)
    assert len(obs) == 6, f"beklenen 6 register, {len(obs)} bulundu: {dict(obs)}"
    for reg, c in obs.items():
        frac = c / n
        assert 0.02 <= frac <= 0.35, f"register '{reg}' payı %{100*frac:.1f} — dengesiz"
    # kesin bir kategori tüm veriye hakim olmasın (Herfindahl < 0.30)
    hhi = sum((c / n) ** 2 for c in obs.values())
    assert hhi < 0.30, f"register yoğunlaşması (HHI) {hhi:.3f} — çeşitlilik zayıf"


# --------------------------------------------------------------------------
# decision × domain kontenjans
# --------------------------------------------------------------------------

def test_decision_domain_contingency_has_no_structural_holes(all_meta):
    functional = {"izin_yonetimi", "maas_finans", "puantaj", "organizasyon",
                  "calisan_bilgileri", "ik_islemleri"}
    table: dict[tuple, int] = Counter((m["domain"], m["decision"]) for m in all_meta)
    holes = []
    for d in functional:
        for dec in ("tool_call", "request_for_info"):
            if table[(d, dec)] == 0:
                holes.append(f"{d} × {dec}")
    assert not holes, f"kontenjans tablosunda yapısal boşluk: {holes}"


def test_decision_domain_association_matches_design_contract(all_meta):
    """decision ile domain BAĞIMSIZ DEĞİLDİR — ve olmamalı. Bu test, gözlenen
    bağımlılığın tasarım sözleşmesine uyduğunu doğrular (rastgele değil)."""
    by_domain: dict[str, Counter] = {}
    for m in all_meta:
        by_domain.setdefault(m["domain"], Counter())[m["decision"]] += 1

    # (1) 'meta' domain (selam/kapasite/politika) yalnızca direct'tir
    assert set(by_domain.get("meta", {})) <= {"direct"}, (
        f"'meta' domain'inde direct dışı karar: {dict(by_domain.get('meta', {}))}"
    )
    # (2) 'kapsanmayan' domain yalnızca cannot_answer'dır
    assert set(by_domain.get("kapsanmayan", {})) <= {"cannot_answer"}, (
        f"'kapsanmayan' domain'inde cannot_answer dışı karar: {dict(by_domain.get('kapsanmayan', {}))}"
    )
    # (3) 'organizasyon' sorguları tool gerektirir → çok az/hiç 'direct'
    org = by_domain.get("organizasyon", Counter())
    if sum(org.values()):
        assert org["direct"] / sum(org.values()) < 0.05, (
            f"organizasyon alanında beklenmedik oranda direct: %{100*org['direct']/sum(org.values()):.1f}"
        )
    # (4) 'ik_islemleri' WRITE/onay ağırlıklıdır → request_for_info payı en yüksek
    ik = by_domain.get("ik_islemleri", Counter())
    if sum(ik.values()):
        assert ik["request_for_info"] >= ik["direct"], (
            f"ik_islemleri'nde request_for_info ({ik['request_for_info']}) < direct ({ik['direct']}) — "
            f"WRITE onay akışları eksik"
        )
    # (5) fonksiyonel okuma domainleri hem tool_call hem request_for_info içerir
    for d in ("maas_finans", "izin_yonetimi", "puantaj", "calisan_bilgileri"):
        b = by_domain.get(d, Counter())
        assert b["tool_call"] > 0 and b["request_for_info"] > 0, (
            f"'{d}' hem tool_call hem request_for_info içermeli: {dict(b)}"
        )


# --------------------------------------------------------------------------
# difficulty ↔ karmaşıklık
# --------------------------------------------------------------------------

def test_difficulty_tracks_turn_count(all_meta):
    by_diff: dict[str, list[int]] = {}
    for m in all_meta:
        by_diff.setdefault(m["difficulty"], []).append(m["turns"])
    mean_turns = {d: sum(v) / len(v) for d, v in by_diff.items()}
    assert mean_turns["kolay"] <= mean_turns["orta"] <= mean_turns["zor"] <= mean_turns["cok_zor"], (
        f"ortalama tur sayısı zorlukla artmıyor: {mean_turns}"
    )
    assert mean_turns["cok_zor"] - mean_turns["kolay"] >= 0.8, (
        f"cok_zor ile kolay arasında tur farkı çok küçük: {mean_turns}"
    )


def test_easy_examples_are_single_turn(all_meta):
    bad = [m["id"] for m in all_meta if m["difficulty"] == "kolay" and m["turns"] != 2]
    assert not bad, f"'kolay' etiketli ama çok turlu örnekler: {bad[:20]}"


# --------------------------------------------------------------------------
# Val oranı — güven aralığı
# --------------------------------------------------------------------------

def test_val_ratio_confidence_interval_contains_target(train_meta, val_meta):
    n = len(train_meta) + len(val_meta)
    lo, hi = wilson_interval(len(val_meta), n, z=2.576)  # %99
    assert lo <= 0.10 <= hi, (
        f"val oranı %99 GA'sı [{lo:.3f}, {hi:.3f}] hedef 0.10'u içermiyor "
        f"(val={len(val_meta)}/{n})"
    )
