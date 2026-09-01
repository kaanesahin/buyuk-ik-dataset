# -*- coding: utf-8 -*-
"""Tool-agnostik cümle kalıpları ve dil-kaydı stillendirmesi.

Frame'ler hiçbir tool'a özel değildir; `{obj}`, `{verb}`, `{syn}`, `{subj}`,
`{plist}`, `{p:<param>}` gibi yer tutucularla çalışır ve KATALOG metadatasıyla
doldurulur. Böylece yeni bir tool eklemek yeni frame yazmayı GEREKTİRMEZ (D-2/D-4).
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
#  KULLANICI FRAME'LERİ
#  kategori: "direct"  -> baş sözcük (obj/verb) geçebilir
#            "oblique" -> baş sözcük GEÇMEZ (syn / betimleme) — anti-kısayol
# --------------------------------------------------------------------------- #
USER_FRAMES_DIRECT = [
    "{subj}{obj} {verb}",
    "{subj}{obj} {verb}, {plist_phrase}",
    "{obj_nom} {q_nom}",
    "{subj}{obj} {verb2} ihtiyacım var",
    "{plist_phrase} — {obj} {verb}",
    "{obj} {verb} {subj_tail}",
    "{subj}{obj_nom} hakkında bilgi ver",
    "{obj} {verb}: {plist_bare}",
]
USER_FRAMES_OBLIQUE = [
    "{subj}{syn}?",
    "{subj}{syn} — {plist_phrase}",
    "{syn}, bir bakar mısın{subj_tail}",
    "{plist_phrase}; {syn}?",
    "{subj_cap}{syn} öğrenmek istiyorum",
    "şunu merak ediyorum: {syn}",
    "{syn}? {plist_bare}",
    "elimde {plist_bare} var; {syn}",
]
# eksik parametre (request_for_info): parametre listesinden biri ÇIKARILIR
USER_FRAMES_MISSING = [
    "{subj}{obj} {verb}",
    "{subj}{syn}?",
    "{obj} {verb} {subj_tail}",
    "{subj}{obj_nom} lazım",
    "{syn} — {plist_phrase}",
]

# çok-adımlı zincir 1. tur (kullanıcı işi ister ama bir parametre eksik)
USER_FRAMES_CHAIN_START = [
    "{subj}{obj} {verb}, uygunsam gerekeni yap",
    "{syn}; sonra da kaydını gir",
    "{subj}{obj} {verb} — müsaitsem işleme al",
    "{subj}{syn}, olduysa da işle",
]

# WRITE istekleri: kullanıcı bir işlem + ayrıntı verir (doğal)
# NOT: {syn} tabanlı (baş sözcük GEÇMEYEN) kalıp oranı bilinçli yüksek tutulur —
# WRITE tarafında nesne-adı → tool kısayolunu kırmak için (anti-shortcut).
USER_FRAMES_WRITE = [
    "{subj}{obj} {verb}. Bilgiler: {plist_phrase}",
    "{subj}{syn} — {plist_phrase}. {verb2}",
    "{subj}{obj} {verb}: {plist_phrase}",
    "{plist_phrase} olacak şekilde {subj}{obj} {verb}",
    "{subj}{syn}. Ayrıntılar: {plist_phrase}",
    "{plist_phrase} — {subj}{syn}; gerekeni yap",
    "{subj}{syn}? Elimde şunlar var: {plist_bare}",
    "{subj}{syn}. {plist_phrase} — gerekeni yap",
    "{plist_phrase}. {subj_cap}{obj} {verb}",
    # parametresiz (yalnız birincil kimlik özneye gömülü)
    "{subj}{obj} {verb}",
    "{subj}{syn} — {verb2}",
    "{subj}{syn}, {verb} lütfen",
]

# --------------------------------------------------------------------------- #
#  ÖZNE / KUYRUK
# --------------------------------------------------------------------------- #
SUBJ_SELF = ["", "benim ", "bana ait ", ""]
SUBJ_EMP = ["{emp} için ", "{emp} adına ", "{emp} — "]
SUBJ_NAME = ["{name} için ", "{name} adlı kişi için "]
SUBJ_TAIL = ["", "", " lütfen", " acaba", " bugün lazım", " mümkünse"]

Q_NOM = ["nedir?", "ne durumda?", "ne oldu?", "hangi aşamada?", "ne kadar?"]

# --------------------------------------------------------------------------- #
#  ASİSTAN FRAME'LERİ
# --------------------------------------------------------------------------- #
# request_for_info: eksik parametre iste (parametrenin `human` alanıyla)
ASK_MISSING = [
    "Bunu getirebilmem için {human} bilgisine ihtiyacım var. Paylaşır mısın?",
    "{human_cap} olmadan devam edemem; iletir misin?",
    "Hangi {human} için bakayım?",
    "{human_cap} nedir? Onu alınca hemen kontrol ederim.",
    "Tamam — {human} bilgisini verirsen işlemi tamamlarım.",
]
ASK_MISSING_MULTI = [
    "Bunun için {human_list} bilgilerine ihtiyacım var. Paylaşır mısın?",
    "Şu eksikleri tamamlayabilir misin: {human_list}?",
    "{human_list} olmadan ilerleyemem; iletebilir misin?",
]
# request_for_info: WRITE onayı iste
CONFIRM_ASK = [
    "Özetliyorum: {summary}. Bu işlemi yapmamı onaylıyor musun?",
    "{summary_cap} işlemini gerçekleştireceğim. Devam edeyim mi?",
    "Şunu yapacağım: {summary}. Onaylıyor musun?",
    "{summary_cap} kaydını oluşturmak üzereyim. Uygun mu?",
]
# request_for_info: çelişkili parametre
CONFLICT_ASK = [
    "Verdiğin bilgide bir tutarsızlık var: {conflict}. Doğrusu ne olmalı?",
    "{conflict_cap} — bunu netleştirir misin? Bu haliyle işleme alamam.",
    "Burada bir çelişki görüyorum: {conflict}. Hangisi geçerli?",
]
ACK = ["Evet.", "Onaylıyorum.", "Evet, onaylıyorum.", "Tamam, devam et.", "Olur, yap.",
       "Evet lütfen.", "Devam edebilirsin.", "Uygundur.", "Onaylıyorum, ilerle.",
       "Tabii, işleme al.", "Evet, doğru; devam et."]

CHAIN_ASK_PARAM = [
    "Bunu yapabilmem için önce {human} bilgisini almam gerekiyor; sonra onayınla uygularım.",
    "{human_cap} nedir? Onu alınca özetleyip onayına sunarım.",
    "Hangi {human}? Söylersen hazırlar, onayınla kaydederim.",
]

# tool sonucu -> asistan nihai yanıtı (SONUÇ ALANLARINA dayanır, uydurma yok)
RESULT_OK = [
    "{obj_nom_cap}: {result_phrase}.",
    "{subj_res}{obj_nom} sonucu — {result_phrase}.",
    "Kontrol ettim: {result_phrase}.",
    "{result_phrase}. Başka bir şey ister misin?",
]
RESULT_EMPTY = [
    "{obj_nom_cap} için kayıt bulunamadı.",
    "Bu kriterlerle {obj_nom} çıkmadı; farklı bir aralık/kimlik dener misin?",
    "Sistemde eşleşen bir sonuç yok.",
]
RESULT_ERROR = [
    "{obj_nom_cap} alınamadı ({err}). Biraz sonra tekrar denenebilir.",
    "İşlem sırasında bir hata oluştu: {err}. İlgili ekibe iletmek ister misin?",
    "Şu an {obj_nom} servisine ulaşamadım ({err}).",
]
RESULT_PARTIAL = [
    "{obj_nom_cap} kısmen döndü: {result_phrase}. Kalan bilgi şu an gelmedi.",
    "Elimde şu kadarı var: {result_phrase}. Geri kalanı için tekrar bakmam gerekebilir.",
]
# WRITE sonrası onay
WRITE_DONE = [
    "Tamamlandı — {summary}. Kayıt numarası: {ref}.",
    "{summary_cap} işlendi. Referans: {ref}.",
    "Oldu. {summary_cap} kaydı oluşturuldu ({ref}).",
]

# direct (tool gerekmeyen) — çeşitli çerçeveleme
DIRECT_WRAP = [
    "{core}",
    "Kısaca: {core_low}",
    "{core} İstersen ilgili işlemi senin için başlatabilirim.",
    "Şöyle özetleyeyim: {core_low}",
    "{core} Ayrıntı şirket politikanıza göre değişebilir.",
]

# cannot_answer — kibar ret + yönlendirme
CANNOT = {
    "no_tool": [
        "Bunu yapabilecek bir aracım yok. Elimdeki tool'lar {domains} gibi işlemleri kapsıyor.",
        "Bu talebi karşılayan bir tool tanımlı değil; dolayısıyla gerçekleştiremem.",
        "Mevcut araç setiyle bu isteğe güvenilir bir yanıt üretemiyorum.",
    ],
    "out_of_scope": [
        "Bu konu benim kapsamımın dışında; yalnızca kurumsal sistem işlemlerinde yardımcı olabilirim.",
        "Ne yazık ki bu, yapabileceklerimin arasında değil.",
        "Bu isteği yerine getirebilecek bir yeteneğim yok.",
    ],
    "future": [
        "Bu geleceğe dönük ve doğrulanamaz bir tahmin; araçlarım yalnızca mevcut ve geçmiş kayıtları getiriyor.",
        "Henüz gerçekleşmemiş bir durumu kesin olarak bildiremem.",
        "Bu tür bir öngörü spekülasyon olur; yanlış yönlendirmemek için yapmıyorum.",
    ],
    "privacy": [
        "Bu, başka bir kişiye ait özel/mali bir bilgi; yetki doğrulaması olmadan paylaşamam.",
        "Kişisel verilerin korunması gereği bu bilgiyi üçüncü biriyle paylaşamam.",
        "Bu kayda erişim özel yetki gerektiriyor; bu asistan üzerinden açamam.",
    ],
    "unauthorized": [
        "Bu işlem için yetkim yok; onay/işlem yetkisi ilgili yönetici ya da İK tarafında.",
        "Böyle bir toplu/yetki-dışı işlemi yürütemem; her kayıt kendi onay akışından geçer.",
        "Bu değişikliği yapacak bir aracım yok; adım yönetici tarafında ilerler.",
    ],
    "advice": [
        "Bu kişisel bir karar; yerinize karar vermem ya da yönlendirmem doğru olmaz.",
        "Kariyer/finans kararları için bir uzmana ya da İK iş ortağınıza danışmanız daha doğru.",
    ],
}
CANNOT_REDIRECT = ["", "", " Bir kurumsal işlem varsa yardımcı olabilirim.",
                   " Mevcut kayıtlarla ilgili soruları yanıtlayabilirim."]

# çok turlu cannot: ısrar + kararlı ret
CANNOT_PUSH = ["Yine de bir fikir ver.", "Yaklaşık söyle yeter.", "Emin olmasan da söyle.",
               "Hadi ama, yardımcı ol.", "Başka bir yolu yok mu?", "Cidden hiç mi bilgi yok?"]
CANNOT_HOLD = [
    "Anlıyorum ama yine de yapamam; doğrulayamadığım bir bilgiyi tahminle paylaşmam.",
    "Israrına rağmen bu isteği geri çevirmem gerekiyor. Yapabildiğim konulara geçebiliriz.",
    "Bu konuda bir adım atamıyorum. Dilersen ilgili ekibe iletmeni öneririm.",
]

# çok turlu direct: takip sorusu
MT_DIRECT_FOLLOWUP_A = [
    "Buna ek olarak bir şey daha sorabilir miyim?",
    "Peki bir de şunu merak ettim.",
    "Bir noktada kafam karıştı.",
]


# --------------------------------------------------------------------------- #
#  STİL / DİL KAYDI  (üstüne uygulanır)
# --------------------------------------------------------------------------- #
def _fold(s):
    s = s.replace("İ", "i").replace("I", "ı").lower()
    for a, b in (("ı", "i"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("ç", "c")):
        s = s.replace(a, b)
    return s


def tr_lower(s):
    """Türkçe-güvenli küçük harf. Python'da 'İ'.lower() == 'i\\u0307' (i + birleşen
    nokta) — bu artefaktı üretmeden 'İ'->'i', 'I'->'ı' yapar."""
    return s.replace("İ", "i").replace("I", "ı").lower()


def tr_upper(s):
    return s.replace("i", "İ").replace("ı", "I").upper()


_COMBINING_DOT = chr(0x0307)


def denorm(s):
    """Metne sızmış birleşen noktayı (U+0307) temizle — son güvenlik ağı.
    Python'da 'İ'.lower() bunu üretir; Türkçe düz metinde bu işaret kullanılmaz."""
    return s.replace(_COMBINING_DOT, "") if _COMBINING_DOT in s else s


FORMAL_PRE = ["Sayın yetkili, ", "İlgili birime iletilmek üzere: ", "Merhaba, ",
              "Konu hakkında bilgi rica ediyorum: ", "İyi çalışmalar; "]
FORMAL_POST = [" Gereğini rica ederim.", " Yardımlarınız için teşekkürler.",
               " Bilginize sunarım.", " Saygılarımla.", " İyi çalışmalar."]
CHAT_PRE = ["Abi ", "Ya ", "Bak şimdi, ", "Şunu bi baksana, ", "Pardon ya, ", "Hocam ", "Selam, "]
CHAT_POST = [" ya", " bu arada", ", olur mu", " hemen lazım", ", ne dersin"]
LONG_PRE = ["Bir toplantı öncesi kontrol ediyorum. ", "Aylık kapanışa hazırlanıyorum. ",
            "Yöneticimle görüşmeden önce netleştirmem gerekiyor. ",
            "Sistemde bulamadım, o yüzden buradan soruyorum. ",
            "Birkaç işi toparlıyorum bugün. "]
LONG_POST = [" Buna göre ilerleyeceğim.", " Teyit alınca rahatlayacağım.",
             " Acele etmiyorum ama bugün lazım.", " Yanlış bir şey yapmak istemiyorum."]

# yaygın Türkçe yazım hataları (ID/sayı token'ları KORUNUR)
_TYPO_MAP = [("mış", "mis"), ("miş", "mis"), ("değil", "degil"), ("bir", "bi"),
             ("acaba", "acaaba"), ("misin", "mısın"), ("yapar mısın", "yaparmisin"),
             ("nasıl", "nasil"), (" crm", "crm")]


def apply_typos(rng, text):
    """Yaygın Türkçe yazım hataları — ID/sayı token'ları korunur. Metni okunmaz
    hale getirmeyecek kadar ölçülü (K-minor): en çok bir harf yer değiştirmesi,
    soru işareti korunur."""
    prot = []
    def _p(m):
        prot.append(m.group(0)); return f"\x00{len(prot)-1}\x00"
    t = re.sub(r"[A-Z]{2,}-\d+|#?\d[\d.:/]*\d|\d+", _p, text)
    t = _fold(t)
    for a, b in _TYPO_MAP:
        if rng.random() < 0.28:
            t = t.replace(a, b)
    t = t.replace(".", "")
    if rng.random() < 0.22:
        # rastgele bir harf çiftini yer değiştir (tek sefer)
        i = rng.randint(0, max(0, len(t) - 3))
        if t[i].isalpha() and t[i + 1].isalpha() and t[i] != " " and t[i+1] != " ":
            t = t[:i] + t[i + 1] + t[i] + t[i + 2:]
    t = re.sub(r"\x00(\d+)\x00", lambda m: prot[int(m.group(1))], t)
    return t.strip()


REGISTER_WEIGHTS = [("plain", 30), ("formal", 18), ("chat", 18), ("long", 15),
                    ("typo", 12), ("short", 7)]


def style(rng, text, register=None):
    """Bir kullanıcı ifadesine dil kaydı uygular. (styled, register) döner."""
    if register is None:
        register = rng.choices([r for r, _ in REGISTER_WEIGHTS],
                               weights=[w for _, w in REGISTER_WEIGHTS])[0]
    t = text.strip().rstrip(" .?!")
    if register == "plain":
        out = text
    elif register == "formal":
        head = tr_lower(t[0]) + t[1:]
        out = rng.choice(FORMAL_PRE) + head + "." + rng.choice(FORMAL_POST)
    elif register == "chat":
        if rng.random() < 0.5:
            out = rng.choice(CHAT_PRE) + (tr_lower(t[0]) + t[1:])
        else:
            out = t + rng.choice(CHAT_POST) + "?"
    elif register == "long":
        core = tr_upper(t[0]) + t[1:] + "."
        out = rng.choice(LONG_PRE) + core + rng.choice(LONG_POST)
    elif register == "typo":
        out = apply_typos(rng, text)
    elif register == "short":
        # yalnız çekirdeği bırak: ilk 6 kelime, noktalama yok
        words = t.split()
        out = " ".join(words[: rng.randint(3, 6)])
    else:
        register, out = "plain", text
    return denorm(out), register
