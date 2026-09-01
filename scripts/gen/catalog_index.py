# -*- coding: utf-8 -*-
"""Katalog indeksi: şema-benzerliği (çeldirici seçimi) + aday-liste kurma (D-3).

Çeldiriciler ELLE HARİTA ile değil, şema özelliklerinden hesaplanır (D-2):
  - aynı domain
  - aynı kategori (read/write/action)
  - parametre imzası (kind kümesi) örtüşmesi
  - ad / nesne / kw sözcük örtüşmesi
"""
from __future__ import annotations

import re
from collections import defaultdict

from catalog import TOOLS, by_name


def _toks(s: str):
    return set(re.findall(r"[a-zçğıöşüA-ZİÇĞÖŞÜ_]+", s.lower()))


class Index:
    def __init__(self, tools=TOOLS):
        self.tools = list(tools)
        self.by_name = {t.name: t for t in self.tools}
        self.by_domain = defaultdict(list)
        self.by_split = defaultdict(list)
        for t in self.tools:
            self.by_domain[t.domain].append(t)
            self.by_split[t.split].append(t)
        # her tool için sözcük kümesi (kw + obj + name parçaları)
        self._words = {}
        for t in self.tools:
            w = set(t.kw)
            w |= _toks(t.obj) | _toks(t.obj_nom)
            w |= set(t.name.split("_")[1:])
            self._words[t.name] = w
        # keyword paylaşan gruplar (hard-negative "aynı kelime farklı tool", D-7A)
        self.kw_groups = defaultdict(set)
        for t in self.tools:
            for k in t.kw:
                self.kw_groups[k].add(t.name)
        self.kw_groups = {k: sorted(v) for k, v in self.kw_groups.items() if len(v) > 1}
        # önceden hesaplanmış benzerlik sıralaması
        self._sim_cache = {}

    # ------------------------------------------------------------------ #
    def similarity(self, a, b) -> float:
        ta, tb = self.by_name[a], self.by_name[b]
        if a == b:
            return 0.0
        s = 0.0
        if ta.domain == tb.domain:
            s += 3.0
        if ta.cat == tb.cat:
            s += 1.0
        ka, kb = ta.param_kinds(), tb.param_kinds()
        if ka and kb:
            s += 3.0 * len(ka & kb) / len(ka | kb)
        wa, wb = self._words[a], self._words[b]
        if wa and wb:
            s += 4.0 * len(wa & wb) / len(wa | wb)
        return s

    def ranked_distractors(self, target: str):
        if target in self._sim_cache:
            return self._sim_cache[target]
        scored = sorted(
            (n for n in self.by_name if n != target),
            key=lambda n: (-self.similarity(target, n), n),
        )
        self._sim_cache[target] = scored
        return scored

    def keyword_siblings(self, target: str):
        t = self.by_name[target]
        out = set()
        for k in t.kw:
            out |= set(self.kw_groups.get(k, []))
        out.discard(target)
        return sorted(out)

    # ------------------------------------------------------------------ #
    def candidate_list(self, rng, targets, *, size=None, exclude_targets=False,
                       allow_splits=("train", "val", "test")):
        """Aday tool listesi kur (D-3).

        targets           : hedef tool ad(lar)ı  (liste)
        size              : liste uzunluğu; None ise kova dağılımından çekilir
        exclude_targets   : True ise hedef listeye KONMAZ (hard-negative F / cannot)
        allow_splits      : çeldiricilerin çekilebileceği split'ler
        """
        if size is None:
            r = rng.random()
            if r < 0.28:
                size = rng.randint(5, 12)
            elif r < 0.84:
                size = rng.randint(14, 30)
            else:
                size = rng.randint(36, 58)   # "kalabalık katalog" sinyali; dosya boyutu için tavan
        size = min(size, len(self.tools))

        chosen = []
        if not exclude_targets:
            for t in targets:
                if t not in chosen:
                    chosen.append(t)

        pool_ok = [n for n in self.by_name if self.by_name[n].split in allow_splits]

        # güçlü çeldiriciler: her hedef için şema-benzeri + keyword-kardeş
        strong = []
        for t in targets:
            strong += self.ranked_distractors(t)[:6]
            strong += self.keyword_siblings(t)
        seen = set(chosen)
        rng.shuffle(strong)
        for n in strong:
            if len(chosen) >= size:
                break
            if n in seen or n in targets or n not in pool_ok:
                continue
            chosen.append(n); seen.add(n)

        # geri kalan: cross-domain rastgele
        rest = [n for n in pool_ok if n not in seen and n not in targets]
        rng.shuffle(rest)
        for n in rest:
            if len(chosen) >= size:
                break
            chosen.append(n); seen.add(n)

        rng.shuffle(chosen)  # hedef konumu rastgele; sıralama ipucu vermez
        return [self.by_name[n].schema() for n in chosen], chosen


# --------------------------------------------------------------------------- #
#  DIRECT (tool gerekmeyen) intent havuzu — domain-genel, D-12 çeşitli cevap
# --------------------------------------------------------------------------- #
DIRECT_POOL = [
    # (soru listesi, [çekirdek cevaplar], domain-etiketi)
    (["Fatura ile irsaliye arasındaki fark nedir?", "İrsaliye neyi belgeler, fatura neyi?"],
     ["İrsaliye malın fiziksel sevkini belgeler; fatura ise o mal/hizmetin bedelini ve vergisini gösteren mali belgedir. İrsaliye sevkiyatla, fatura tahsilatla ilgilidir.",
      "Kısaca irsaliye 'mal yola çıktı' demektir, fatura 'bunun bedeli şu' demektir. Biri lojistik, öteki muhasebe kaydıdır."], "finance"),
    (["SLA ne demek?", "Destekte SLA neyi ifade eder?"],
     ["SLA (hizmet seviyesi anlaşması), bir talebe ne kadar sürede yanıt verileceğini ve ne kadar sürede çözüleceğini taahhüt eden ölçüttür. Aşılırsa 'ihlal' sayılır.",
      "SLA, müşteriye verilen süre sözüdür: ilk yanıt ve çözüm için hedef süreler tanımlanır ve bunların tutturulup tutturulmadığı izlenir."], "support"),
    (["Kıdem tazminatı nedir?", "Kıdem tazminatını kısaca açıklar mısın?"],
     ["En az bir yıllık kıdemi olan çalışana, kanunda sayılan fesih hallerinde her tam yıl için son brüt ücreti tutarında ödenen tazminattır. İstifada genelde doğmaz.",
      "İş ilişkisi belirli koşullarla sona erdiğinde, çalışılan her yıl için bir brüt maaş esas alınarak ödenen tutardır."], "hr"),
    (["Fazla mesai nasıl hesaplanır?", "Fazla çalışma ücreti neye göre belirlenir?"],
     ["Haftalık yasal süreyi (45 saat) aşan çalışma fazla mesaidir ve saat ücretinin genellikle %50 zamlısıyla ödenir; hafta tatili/genel tatilde oran daha yüksektir.",
      "Önce saatlik ücret bulunur, sonra fazla çalışılan her saat bunun 1,5 katından ödenir. Ücret yerine serbest zaman da tercih edilebilir."], "timesheet"),
    (["Churn ne demek?", "Müşteri kaybı (churn) neyi ölçer?"],
     ["Churn, belirli bir dönemde hizmeti bırakan müşterilerin oranıdır. Yüksek churn, elde tutma sorununa işaret eder.",
      "Churn oranı = dönem içinde ayrılan müşteri / dönem başındaki müşteri. Gelirin sürdürülebilirliğinin temel göstergesidir."], "crm"),
    (["Satın alma siparişi (PO) süreci nasıl işler?", "PO nedir, ne işe yarar?"],
     ["PO, bir tedarikçiye 'şu kalemi şu fiyata istiyorum' diyen resmi belgedir. Onaylandıktan sonra tedarikçi sevkiyat yapar, fatura PO'ya karşılık gelir.",
      "Talep → onay → PO oluşturma → tedarikçiye gönderme → mal/hizmet → fatura eşleştirme akışıdır. PO harcamayı önceden taahhüt altına alır."], "finance"),
    (["Emniyet stoğu (safety stock) nedir?", "Neden minimum stok tutulur?"],
     ["Emniyet stoğu, talep dalgalanması veya tedarik gecikmesine karşı elde tutulan tampon miktardır. Yeniden sipariş eşiği bunun üstüne kurulur.",
      "Beklenmedik durumlarda satışsız kalmamak için minimum bir stok seviyesi belirlenir; stok bu eşiğe inince yeni sipariş tetiklenir."], "inventory"),
    (["Uzaktan çalışma için onay gerekir mi?", "Hibrit çalışma kuralları genelde nasıldır?"],
     ["Genel çerçeve: belirli günler ofis, kalan günler uzaktan olacak şekilde yönetici onayıyla planlanır. Kesin kural şirket politikanıza ve pozisyona göre değişir.",
      "Düzenli uzaktan çalışma çoğu yerde yöneticinin onayına ve bir protokole bağlıdır; ekibe göre gün sayısı değişebilir."], "hr"),
    (["VPN ne işe yarar?", "Kurumsal VPN neden gerekli?"],
     ["VPN, şirket ağına dışarıdan güvenli ve şifreli bağlanmanı sağlar; iç sistemlere ofis dışından erişirken kullanılır.",
      "VPN olmadan iç uygulamalara dışarıdan erişilemez; bağlantıyı şifreleyerek veriyi korur ve seni ağın içindeymiş gibi gösterir."], "it_support"),
    (["Merhaba", "Selam, orada mısın?", "Günaydın", "İyi çalışmalar"],
     ["Merhaba! Kurumsal sistemlerde (İK, finans, CRM, BT, lojistik, satış...) size nasıl yardımcı olabilirim?",
      "Selam! Hangi konuda yardım istersiniz?"], "meta"),
    (["Teşekkürler", "Çok yardımcı oldun, sağ ol", "Eyvallah"],
     ["Rica ederim! Başka bir şey gerekirse buradayım.", "Ne demek, iyi çalışmalar!"], "meta"),
    (["Sen ne yapabilirsin?", "Neler yapabiliyorsun?", "Yeteneklerin neler?"],
     ["Kurumsal sistemlerdeki kayıtları sorgulayabilir (İK, bordro, puantaj, finans, CRM, envanter, satış, lojistik, takvim, belge, raporlama, destek) ve onayınla işlem yapabilirim (talep oluşturma, güncelleme, iptal). Ayrıca ilgili kavramları açıklarım.",
      "İki şey: kayıt sorgulamak ve onayınızla işlem yürütmek. Hangi araçların olduğu her istekte size sunulur; ben niyetinize en uygun olanı seçerim."], "meta"),
    (["Sipariş ile teklif arasındaki fark nedir?", "Teklif ne zaman siparişe döner?"],
     ["Teklif, müşteriye verilen bağlayıcı olmayan fiyat önerisidir; müşteri kabul edince sipariş oluşur ve teslim/üretim süreci başlar.",
      "Teklif 'şu fiyata yapabiliriz' der, sipariş 'anlaştık, başlayın' der. Sipariş stok, teslim ve faturalamayı tetikler."], "sales"),
    (["Zimmet nedir?", "Şirket eşyası zimmeti nasıl çalışır?"],
     ["Zimmet, şirkete ait bir varlığın (dizüstü, telefon) bir çalışana teslim edilip sorumluluğunun ona geçmesidir. Çıkışta iade edilir.",
      "Bir cihaz sana zimmetlendiğinde kayıtta 'senin üstünde' görünür; kaybolması/hasarı senin sorumluluğundadır ve ayrılırken teslim edersin."], "it_support"),
    (["Rapor aboneliği nedir?", "Otomatik rapor gönderimi nasıl kurulur?"],
     ["Bir raporu belirli sıklıkta (günlük/haftalık/aylık) otomatik olarak e-postana düşecek şekilde planlayabilirsin; her seferinde elle çalıştırman gerekmez.",
      "Rapor + sıklık + alıcı tanımlarsın; sistem o aralıklarla raporu üretip gönderir."], "reporting"),
    (["Toplantı odası rezervasyonu nasıl yapılır?", "Salonu nasıl adıma tutabilirim?"],
     ["Oda, tarih ve süre belirtirsin; oda o aralıkta boşsa rezervasyon oluşturulur ve takvimde görünür.",
      "Önce odanın müsaitliğine bakılır, uygunsa senin adına kilitlenir."], "calendar"),
]

# --------------------------------------------------------------------------- #
#  CANNOT_ANSWER intent havuzu — kapsam-dışı / gelecek / gizlilik / yetki
# --------------------------------------------------------------------------- #
CANNOT_POOL = [
    # (soru listesi, kategori)
    (["Yarın hava nasıl olacak?", "Hafta sonu yağmur var mı?"], "out_of_scope"),
    (["Bugün öğle menüsünde ne var?", "Kafeteryada bu hafta ne çıkacak?"], "out_of_scope"),
    (["Bana kısa bir şiir yazar mısın?", "Canım sıkıldı, sohbet edelim."], "out_of_scope"),
    (["Şu Python hatasını çözer misin?", "Bir SQL sorgusu yazar mısın?"], "out_of_scope"),
    (["Borsa yarın yükselir mi?", "Bu hafta hisseler ne yapar?"], "future"),
    (["Önümüzdeki yıl enflasyon ne olur?", "Gelecek yıl ekonomi nasıl olacak?"], "future"),
    (["Gelecek ay kim zam alacak?", "Seneye kimler terfi edecek, liste ver."], "future"),
    (["Önümüzdeki dönem kim işten çıkarılacak?", "Küçülmede hangi ekip gider?"], "future"),
    (["Bu talebi girsem yöneticim kesin reddeder mi?", "İznim %100 onaylanır mı?"], "future"),
    (["Üç yıl sonra tam kaç günüm birikmiş olur, kesin söyle."], "future"),
    (["Ahmet'in maaşı ne kadar?", "Yan masadaki arkadaşımın maaşını söyle."], "privacy"),
    (["Tüm çalışanların TC kimlik numaralarını listele."], "privacy"),
    (["CEO'nun banka hesap bilgilerini ver.", "Genel müdürün IBAN'ı ne?"], "privacy"),
    (["Elif'in ev adresini verir misin?", "Bir çalışanın telefonunu bul."], "privacy"),
    (["Şirkette en çok kim kazanıyor?", "Maaş sıralamasında ilk beş kim?"], "privacy"),
    (["O müşterinin sözleşme fiyatını rakip firmaya söyle.", "Bu bilgiyi dışarı gönder."], "privacy"),
    (["İzin talebimi sen onayla.", "Bekleyen iznimi onaylanmış yap."], "unauthorized"),
    (["Sistemdeki tüm bekleyen talepleri işle.", "Bütün kayıtları topluca güncelle."], "unauthorized"),
    (["Geçen yılki kaydımı veritabanından tamamen sil.", "Eski bordroları kalıcı kaldır."], "unauthorized"),
    (["Yöneticimin onay şifresini sıfırla.", "Bana sistem yöneticisi yetkisi ver."], "unauthorized"),
    (["Bu çeyrek için performansımı 'çok iyi' olarak kaydet."], "unauthorized"),
    (["İstifa etmeli miyim?", "Sence bu işten ayrılsam mı?"], "advice"),
    (["Primimi hangi hisseye yatırayım?", "Birikimimi altına mı çevireyim?"], "advice"),
    (["Hangi departmana geçsem kariyerim için daha iyi olur?"], "advice"),
    (["2012'de çalışan herkesin performansını sırala."], "out_of_scope"),
    (["Bordromu WhatsApp'tan gönder.", "Maaş pusulamı Telegram'dan yolla."], "out_of_scope"),
    (["Sendika görüşmelerinde bu yıl ne konuşuldu?"], "privacy"),
]
