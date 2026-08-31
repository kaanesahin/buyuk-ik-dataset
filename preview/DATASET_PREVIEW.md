# Büyük İK dataset — okunur önizleme

Bu dosya **otomatik üretilir** (`scripts/make_preview.py`) ve salt-okunurdur. Kanonik veri `data/*.jsonl` içindedir; buradaki örnekler birebir aynı içeriğin gözle okunur dökümüdür. Örnek seçimi deterministiktir.

- Kaynak kayıt: **3000** (train+val)
- Önizlenen: her karar sınıfından **22** örnek (intent / register / difficulty ekseninde yayılmış)

## Karar dağılımı (tam veri)

| decision | adet | oran |
|---|---:|---:|
| `tool_call` | 900 | %30.0 |
| `direct` | 750 | %25.0 |
| `request_for_info` | 750 | %25.0 |
| `cannot_answer` | 600 | %20.0 |

## İçindekiler

- [Çok-adımlı zincir ve çok turlu örnekler](#cok-adimli-ve-cok-turlu)
- [tool_call](#tool-call)
- [direct](#direct)
- [request_for_info](#request-for-info)
- [cannot_answer](#cannot-answer)



<a id="cok-adimli-ve-cok-turlu"></a>

## Çok-adımlı zincir ve çok turlu örnekler

_6 turlu `tool_call` zinciri (parametre topla → onay iste → uygula), çok turlu `direct` (tanım + takip sorusu) ve çok turlu `cannot_answer` (ret + kullanıcı ısrarı + kararlı ret)._

### 1. `create_leave_request`  ·  tool_call
**domain** ik_islemleri  ·  **difficulty** cok_zor  ·  **register** resmi  ·  **turns** 6  ·  **split** train  ·  **write** ✔  ·  **onay gerekir** ✔
**eksik parametre:** `range`

_Araçlar:_ 6 araç — update_employee_contact, check_employee_access, update_employee_information, cancel_izin_talebi, update_izin_talebi, **create_izin_talebi**

**🧑 Kullanıcı**
> Konu: bilgi talebi. EMP-2889 kodlu personel için mazeret izni kullanmak istiyorum, uygunsam talebi de oluştur. İyi çalışmalar dilerim.

**🤖 Asistan**  ·  _text_
> Hangi tarihlerde? Başlangıç ve bitişi verirseniz talebi hazırlar, onayınızla kaydederim.

**🧑 Kullanıcı**
> Tarihler: 3 Ekim 2026 ile 7 Ekim 2026 arası.

**🤖 Asistan**  ·  _text_
> Şunu yapacağım: EMP-2889 kodlu personel için 3 Ekim 2026 ile 7 Ekim 2026 arası tarihlerinde mazeret izni talebi. Aşağıdaki bilgilerle devam etmemi onaylıyor musunuz?

**🧑 Kullanıcı**
> Evet lütfen oluştur.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "create_izin_talebi", "arguments": {"employee_id": "EMP-2889", "izin_tipi": "mazeret", "baslangic_tarihi": "2026-10-03", "bitis_tarihi": "2026-10-07"}}
</tool_call>
```

---

### 2. `update_leave_request`  ·  tool_call
**domain** ik_islemleri  ·  **difficulty** cok_zor  ·  **register** yazim_hatali  ·  **turns** 6  ·  **split** train  ·  **write** ✔  ·  **onay gerekir** ✔
**eksik parametre:** `range`

_Araçlar:_ 8 araç — get_employee_status, update_employee_contact, get_izin_bakiyesi, create_izin_talebi, cancel_izin_talebi, update_employee_information, **update_izin_talebi**, check_employee_access

**🧑 Kullanıcı**
> lv-2026-0206 talebini baska tarihlere cekelim, sonra da kaydet

**🤖 Asistan**  ·  _text_
> Yeni tarihler ne olsun? Başlangıç ve bitişi verirseniz güncellemeyi hazırlayıp onayınızı alırım.

**🧑 Kullanıcı**
> 10 Aralık 2026 ile 24 Aralık 2026.

**🤖 Asistan**  ·  _text_
> Şunu yapacağım: LV-2026-0206 numaralı iznin 10 Aralık 2026 ile 24 Aralık 2026 olarak güncellenmesi. Aşağıdaki bilgilerle devam etmemi onaylıyor musunuz?

**🧑 Kullanıcı**
> Evet bunu uygula.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "update_izin_talebi", "arguments": {"talep_id": "LV-2026-0206", "yeni_baslangic_tarihi": "2026-12-10", "yeni_bitis_tarihi": "2026-12-24"}}
</tool_call>
```

---

### 3. `update_leave_request`  ·  tool_call
**domain** ik_islemleri  ·  **difficulty** cok_zor  ·  **register** gundelik  ·  **turns** 6  ·  **split** train  ·  **write** ✔  ·  **onay gerekir** ✔
**eksik parametre:** `range`

_Araçlar:_ 6 araç — **update_izin_talebi**, update_employee_information, cancel_izin_talebi, update_employee_contact, check_employee_access, create_izin_talebi

**🧑 Kullanıcı**
> LV-2026-0206 numaralı iznimin tarihlerini kaydırmak istiyorum, uygunsa güncelle.

**🤖 Asistan**  ·  _text_
> Yeni tarihler ne olsun? Başlangıç ve bitişi verirseniz güncellemeyi hazırlayıp onayınızı alırım.

**🧑 Kullanıcı**
> 6-8 Ekim 2026 olarak.

**🤖 Asistan**  ·  _text_
> LV-2026-0206 numaralı iznin 6-8 Ekim 2026 olarak güncellenmesi işlemini gerçekleştireceğim. Bu değişikliği şimdi uygulamam uygun mu?

**🧑 Kullanıcı**
> Onaylıyorum.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "update_izin_talebi", "arguments": {"talep_id": "LV-2026-0206", "yeni_baslangic_tarihi": "2026-10-06", "yeni_bitis_tarihi": "2026-10-08"}}
</tool_call>
```

---

### 4. `create_leave_request`  ·  tool_call
**domain** ik_islemleri  ·  **difficulty** cok_zor  ·  **register** resmi  ·  **turns** 6  ·  **split** train  ·  **write** ✔  ·  **onay gerekir** ✔
**eksik parametre:** `range`

_Araçlar:_ 7 araç — update_izin_talebi, cancel_izin_talebi, update_employee_contact, **create_izin_talebi**, update_employee_information, get_izin_bakiyesi, check_employee_access

**🧑 Kullanıcı**
> İlgili birime iletilmek üzere: 3458 numaralı personel için hastalık izni almak istiyorum, sonra da kaydını gir. İyi çalışmalar dilerim.

**🤖 Asistan**  ·  _text_
> Hangi tarihlerde? Başlangıç ve bitişi verirseniz talebi hazırlar, onayınızla kaydederim.

**🧑 Kullanıcı**
> 15-20 Eylül 2026 için olsun.

**🤖 Asistan**  ·  _text_
> Şunu yapacağım: 3458 numaralı personel için 15-20 Eylül 2026 tarihlerinde hastalık izni talebi. Bu değişikliği şimdi uygulamam uygun mu?

**🧑 Kullanıcı**
> Tabii, işleme al.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "create_izin_talebi", "arguments": {"employee_id": "EMP-3458", "izin_tipi": "hastalik", "baslangic_tarihi": "2026-09-15", "bitis_tarihi": "2026-09-20"}}
</tool_call>
```

---

### 5. `define_annual_leave`  ·  direct
**domain** izin_yonetimi  ·  **difficulty** zor  ·  **register** gundelik  ·  **turns** 4  ·  **split** train

_Araçlar:_ 5 araç — get_izin_bakiyesi, get_izin_gecmisi, get_izin_talebi_durumu, cancel_izin_talebi, create_izin_talebi

**🧑 Kullanıcı**
> Yıllık ücretli izni kısaca açıklar mısın?

**🤖 Asistan**  ·  _text_
> Yıllık izin, çalıştığınız her yıl karşılığında kazandığınız, ücretiniz ödenmeye devam ederken kullanabileceğiniz dinlenme iznidir. Süresi kıdeme göre değişir ve kullandıkça bakiyenizden düşer.

**🧑 Kullanıcı**
> Peki sonraki yıla devreder mi?

**🤖 Asistan**  ·  _text_
> Genelde yanmaz; çoğu şirkette bir sonraki döneme devreder, ancak devredilebilecek gün sayısına veya kullanım süresine tavan konabilir. Kesin kural şirketinizin İK politikasına göre değişir.

---

### 6. `leave_request_process`  ·  direct
**domain** ik_islemleri  ·  **difficulty** cok_zor  ·  **register** uzun  ·  **turns** 4  ·  **split** train

_Araçlar:_ 6 araç — update_izin_talebi, update_employee_information, check_employee_access, create_izin_talebi, cancel_izin_talebi, update_employee_contact

**🧑 Kullanıcı**
> Bir toplantıya gireceğim, öncesinde şunu teyit etmem lazım. İzin başvurusu nasıl yapılıyor? Teyit alınca rahatlayacağım.

**🤖 Asistan**  ·  _text_
> İzin türünü seçer, başlangıç ve bitiş tarihlerini girer, gerekiyorsa kısa bir açıklama ekler ve talebi gönderirsiniz. Talep yöneticinize onaya düşer.

**🧑 Kullanıcı**
> Onaydan sonra ne oluyor?

**🤖 Asistan**  ·  _text_
> Süre yöneticinize bağlıdır; genelde birkaç iş günü içinde sonuçlanır. Onaylandığında izniniz kesinleşir ve izin bakiyeniz otomatik güncellenir.

---

### 7. `define_severance`  ·  direct
**domain** maas_finans  ·  **difficulty** zor  ·  **register** yazim_hatali  ·  **turns** 4  ·  **split** train

_Araçlar:_ 6 araç — get_mesai_bilgisi, get_yan_haklar, get_bordro, get_maas_bilgisi, get_prim_bilgisi, create_ucret_degisiklik_talebi

**🧑 Kullanıcı**
> kidem tazminatini kisaca anlatir misin

**🤖 Asistan**  ·  _text_
> En az bir yıllık kıdemi olan çalışana, kanunda sayılan fesih hallerinde her tam yıl için son brüt ücreti tutarında ödenen tazminattır.

**🧑 Kullanıcı**
> İstifa edersem de alır mıyım?

**🤖 Asistan**  ·  _text_
> Hayır. İstifa gibi bazı durumlarda kıdem tazminatı hakkı doğmaz; hak, feshin nedenine bağlıdır. Emeklilik, askerlik, evlilik (kadın çalışan için) gibi istisnalar dışında kendi isteğiyle ayrılan çalışan genelde hak etmez.

---

### 8. `predict_layoffs`  ·  cannot_answer
**domain** organizasyon  ·  **difficulty** zor  ·  **register** gundelik  ·  **turns** 4  ·  **split** val

_Araçlar:_ 8 araç — create_ucret_degisiklik_talebi, get_calisan_listesi, get_employee_status, get_employee_info, create_izin_talebi, create_pozisyon_degisiklik_talebi, get_yonetici_bilgisi, get_departman_bilgisi

**🧑 Kullanıcı**
> Küçülme olursa hangi departmandan kimler gider?

**🤖 Asistan**  ·  _text_
> Bu konuda yardımcı olamıyorum. Gelecekte ne olacağını kestiremem; araçlarım yalnızca mevcut ve geçmiş kayıtları getiriyor. Dilerseniz mevcut kayıtlara bakabilirim.

**🧑 Kullanıcı**
> Cidden hiç mi bilgi veremiyorsun?

**🤖 Asistan**  ·  _text_
> Bu talebi karşılayamıyorum; ısrar etseniz de veremeyeceğim bir bilgi bu. İzin, maaş, bordro veya puantaj konularında destek olabilirim.

---

### 9. `predict_exact_future_leave`  ·  cannot_answer
**domain** izin_yonetimi  ·  **difficulty** zor  ·  **register** resmi  ·  **turns** 4  ·  **split** train

_Araçlar:_ 7 araç — get_izin_bakiyesi, get_izin_gecmisi, get_yan_haklar, get_izin_talebi_durumu, update_izin_talebi, create_izin_talebi, cancel_izin_talebi

**🧑 Kullanıcı**
> Merhaba, aşağıdaki hususta bilgilendirilmek istiyorum: 2030'da kac gun yillik iznim olacak net rakam ver. Saygılarımla.

**🤖 Asistan**  ·  _text_
> Bu konuda yardımcı olamıyorum. Gelecekte ne olacağını kestiremem; araçlarım yalnızca mevcut ve geçmiş kayıtları getiriyor.

**🧑 Kullanıcı**
> Kabaca bir tahmin yeterli.

**🤖 Asistan**  ·  _text_
> Ne yazık ki bu, mevcut araçların dışında kalıyor. İsterseniz yapabildiğim İK konularına geçelim.

---

### 10. `all_employees_id_numbers`  ·  cannot_answer
**domain** calisan_bilgileri  ·  **difficulty** zor  ·  **register** gundelik  ·  **turns** 4  ·  **split** train

_Araçlar:_ 8 araç — get_employee_status, update_employee_information, update_employee_contact, get_calisan_listesi, get_employee_info, get_izin_talebi_durumu, get_departman_bilgisi, get_yonetici_bilgisi

**🧑 Kullanıcı**
> herkesin kimlik no telefon bilgilerini disari aktar

**🤖 Asistan**  ·  _text_
> Başka bir çalışanın bu bilgisini paylaşamam; kişisel ve mali veriler yalnızca ilgili kişiye ve yetkili İK ekibine açıktır. Yetki durumunuzu birlikte kontrol etmemi isterseniz söyleyin.

**🧑 Kullanıcı**
> Kabaca bir tahmin yeterli.

**🤖 Asistan**  ·  _text_
> Ne yazık ki bu, mevcut araçların dışında kalıyor. İsterseniz yapabildiğim İK konularına geçelim.

---


<a id="tool-call"></a>

## TOOL_CALL — tool çağır (tüm zorunlu parametreler mevcut)

_Bu bölümde 22 örnek, 22 farklı intent._

### 1. `balance_and_history`  ·  tool_call
**domain** izin_yonetimi  ·  **difficulty** zor  ·  **register** gundelik  ·  **turns** 2  ·  **split** val

_Araçlar:_ 6 araç — get_izin_talebi_durumu, **get_izin_gecmisi**, **get_izin_bakiyesi**, cancel_izin_talebi, create_izin_talebi, get_puantaj

**🧑 Kullanıcı**
> EMP-1434 kalan iznini ve geçmiş izin kayıtlarını çıkar.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "get_izin_bakiyesi", "arguments": {"employee_id": "EMP-1434"}}
</tool_call>
<tool_call>
{"name": "get_izin_gecmisi", "arguments": {"employee_id": "EMP-1434"}}
</tool_call>
```

---

### 2. `cancel_leave_request`  ·  tool_call
**domain** ik_islemleri  ·  **difficulty** cok_zor  ·  **register** resmi  ·  **turns** 4  ·  **split** train  ·  **write** ✔  ·  **onay gerekir** ✔

_Araçlar:_ 7 araç — **cancel_izin_talebi**, get_izin_talebi_durumu, check_employee_access, create_izin_talebi, update_employee_information, update_izin_talebi, update_employee_contact

**🧑 Kullanıcı**
> Sayın yetkili, LV-2026-1057 numaralı izin talebini iptal et. Yardımlarınız için şimdiden teşekkür ederim.

**🤖 Asistan**  ·  _text_
> Şunu yapacağım: LV-2026-1057 numaralı izin talebinin iptali. Devam etmemi ister misiniz?

**🧑 Kullanıcı**
> Tamam, devam et.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "cancel_izin_talebi", "arguments": {"talep_id": "LV-2026-1057"}}
</tool_call>
```

---

### 3. `check_access`  ·  tool_call
**domain** ik_islemleri  ·  **difficulty** zor  ·  **register** gundelik  ·  **turns** 2  ·  **split** train

_Araçlar:_ 6 araç — update_employee_information, cancel_izin_talebi, **check_employee_access**, update_employee_contact, create_izin_talebi, update_izin_talebi

**🧑 Kullanıcı**
> EMP-2886 olarak EMP-6344 çalışanının maaş bilgisine erişme yetkim var mı?

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "check_employee_access", "arguments": {"requester_id": "EMP-2886", "hedef_employee_id": "EMP-6344", "kaynak_tipi": "maas"}}
</tool_call>
```

---

### 4. `create_leave_request`  ·  tool_call
**domain** ik_islemleri  ·  **difficulty** cok_zor  ·  **register** gundelik  ·  **turns** 4  ·  **split** train  ·  **write** ✔  ·  **onay gerekir** ✔

_Araçlar:_ 6 araç — check_employee_access, update_employee_information, update_employee_contact, update_izin_talebi, cancel_izin_talebi, **create_izin_talebi**

**🧑 Kullanıcı**
> 3388 numaralı personel için 1 Eylül - 12 Eylül 2026 tarihleri arasında yıllık izin oluştur.

**🤖 Asistan**  ·  _text_
> 3388 numaralı personel için 1 Eylül - 12 Eylül 2026 tarihlerinde yıllık izin talebi işlemini gerçekleştireceğim. Bunu bu şekilde kaydetmemi ister misiniz?

**🧑 Kullanıcı**
> Tamam, devam et.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "create_izin_talebi", "arguments": {"employee_id": "EMP-3388", "izin_tipi": "yillik", "baslangic_tarihi": "2026-09-01", "bitis_tarihi": "2026-09-12"}}
</tool_call>
```

---

### 5. `create_salary_change`  ·  tool_call
**domain** ik_islemleri  ·  **difficulty** cok_zor  ·  **register** resmi  ·  **turns** 4  ·  **split** val  ·  **write** ✔  ·  **onay gerekir** ✔

_Araçlar:_ 7 araç — check_employee_access, update_izin_talebi, update_employee_information, update_employee_contact, **create_ucret_degisiklik_talebi**, cancel_izin_talebi, create_izin_talebi

**🧑 Kullanıcı**
> İK birimine, EMP-4793 kodlu personel ücret revizyonu: yeni brüt 76000 TL, gerekçe ekip liderliğine geçiş. İyi çalışmalar dilerim.

**🤖 Asistan**  ·  _text_
> EMP-4793 kodlu personel için brüt ücretin 76000 TL olması (ekip liderliğine geçiş) işlemini gerçekleştireceğim. Bunu bu şekilde kaydetmemi ister misiniz?

**🧑 Kullanıcı**
> Evet.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "create_ucret_degisiklik_talebi", "arguments": {"employee_id": "EMP-4793", "yeni_brut_ucret": 76000, "gerekce": "ekip liderliğine geçiş"}}
</tool_call>
```

---

### 6. `get_benefits`  ·  tool_call
**domain** maas_finans  ·  **difficulty** kolay  ·  **register** gundelik  ·  **turns** 2  ·  **split** train

_Araçlar:_ 8 araç — create_ucret_degisiklik_talebi, create_pozisyon_degisiklik_talebi, get_maas_bilgisi, get_prim_bilgisi, get_bordro, get_izin_talebi_durumu, get_mesai_bilgisi, **get_yan_haklar**

**🧑 Kullanıcı**
> sicil no 5554 yemek kartı ve ulaşım desteği var mı?

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "get_yan_haklar", "arguments": {"employee_id": "EMP-5554"}}
</tool_call>
```

---

### 7. `get_bonus`  ·  tool_call
**domain** maas_finans  ·  **difficulty** orta  ·  **register** konusma_dili  ·  **turns** 2  ·  **split** val

_Araçlar:_ 8 araç — create_ucret_degisiklik_talebi, update_employee_contact, **get_prim_bilgisi**, get_bordro, create_pozisyon_degisiklik_talebi, get_maas_bilgisi, get_yan_haklar, get_mesai_bilgisi

**🧑 Kullanıcı**
> çalışan EMP-5101 için geçen yıl prim dökümünü çıkar bu arada?

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "get_prim_bilgisi", "arguments": {"employee_id": "EMP-5101", "donem": "2025"}}
</tool_call>
```

---

### 8. `get_department_info`  ·  tool_call
**domain** organizasyon  ·  **difficulty** kolay  ·  **register** konusma_dili  ·  **turns** 2  ·  **split** val

_Araçlar:_ 6 araç — create_pozisyon_degisiklik_talebi, get_yonetici_bilgisi, **get_departman_bilgisi**, get_calisan_listesi, get_employee_status, get_employee_info

**🧑 Kullanıcı**
> satış departmanı kaç kişi çalışıyor acaba?

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "get_departman_bilgisi", "arguments": {"departman_adi": "Satış"}}
</tool_call>
```

---

### 9. `get_employee_info`  ·  tool_call
**domain** calisan_bilgileri  ·  **difficulty** kolay  ·  **register** gundelik  ·  **turns** 2  ·  **split** train

_Araçlar:_ 5 araç — get_employee_status, get_yonetici_bilgisi, update_employee_information, **get_employee_info**, update_employee_contact

**🧑 Kullanıcı**
> 2321 numaralı çalışan unvanı ve bağlı olduğu birim ne?

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "get_employee_info", "arguments": {"employee_id": "EMP-2321"}}
</tool_call>
```

---

### 10. `get_employee_status`  ·  tool_call
**domain** calisan_bilgileri  ·  **difficulty** kolay  ·  **register** gundelik  ·  **turns** 2  ·  **split** train

_Araçlar:_ 6 araç — update_employee_contact, **get_employee_status**, get_calisan_listesi, update_employee_information, get_employee_info, get_yonetici_bilgisi

**🧑 Kullanıcı**
> personel EMP-5650 çalışma durumu nedir?

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "get_employee_status", "arguments": {"employee_id": "EMP-5650"}}
</tool_call>
```

---

### 11. `get_leave_balance`  ·  tool_call
**domain** izin_yonetimi  ·  **difficulty** zor  ·  **register** konusma_dili  ·  **turns** 4  ·  **split** train

_Araçlar:_ 7 araç — **get_izin_bakiyesi**, get_puantaj, create_izin_talebi, get_izin_gecmisi, cancel_izin_talebi, update_izin_talebi, get_izin_talebi_durumu

**🧑 Kullanıcı**
> Hocam kaç gün yıllık iznim kaldı?

**🤖 Asistan**  ·  _text_
> Çalışan numaranızı paylaşır mısınız?

**🧑 Kullanıcı**
> EMP-6748 kodlu personel, teşekkürler.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "get_izin_bakiyesi", "arguments": {"employee_id": "EMP-6748"}}
</tool_call>
```

---

### 12. `get_leave_request_status`  ·  tool_call
**domain** izin_yonetimi  ·  **difficulty** zor  ·  **register** uzun  ·  **turns** 2  ·  **split** train

_Araçlar:_ 8 araç — get_izin_bakiyesi, get_puantaj, get_yan_haklar, create_izin_talebi, cancel_izin_talebi, get_izin_gecmisi, update_izin_talebi, **get_izin_talebi_durumu**

**🧑 Kullanıcı**
> Bir toplantıya gireceğim, öncesinde şunu teyit etmem lazım. EMP-6003 numaralı çalışan için LV-2026-0934 numaralı talebin durumu nedir? Acele etmiyorum ama bugün içinde lazım.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "get_izin_talebi_durumu", "arguments": {"employee_id": "EMP-6003", "talep_id": "LV-2026-0934"}}
</tool_call>
```

---

### 13. `get_manager`  ·  tool_call
**domain** organizasyon  ·  **difficulty** kolay  ·  **register** gundelik  ·  **turns** 2  ·  **split** train

_Araçlar:_ 7 araç — get_employee_status, create_ucret_degisiklik_talebi, get_departman_bilgisi, create_pozisyon_degisiklik_talebi, get_employee_info, **get_yonetici_bilgisi**, get_calisan_listesi

**🧑 Kullanıcı**
> personel EMP-3086 kime bağlı çalışıyor?

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "get_yonetici_bilgisi", "arguments": {"employee_id": "EMP-3086"}}
</tool_call>
```

---

### 14. `get_overtime`  ·  tool_call
**domain** puantaj  ·  **difficulty** orta  ·  **register** yazim_hatali  ·  **turns** 2  ·  **split** train

_Araçlar:_ 8 araç — get_employee_info, get_bordro, get_employee_status, get_prim_bilgisi, get_puantaj, **get_mesai_bilgisi**, get_izin_talebi_durumu, get_izin_gecmisi

**🧑 Kullanıcı**
> calisan emp-6085 2026 mayis mesai bilgisini getir

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "get_mesai_bilgisi", "arguments": {"employee_id": "EMP-6085", "donem": "2026-05"}}
</tool_call>
```

---

### 15. `get_payslip`  ·  tool_call
**domain** maas_finans  ·  **difficulty** orta  ·  **register** konusma_dili  ·  **turns** 2  ·  **split** train

_Araçlar:_ 8 araç — **get_bordro**, get_yan_haklar, create_pozisyon_degisiklik_talebi, create_ucret_degisiklik_talebi, get_prim_bilgisi, get_mesai_bilgisi, get_maas_bilgisi, get_izin_gecmisi

**🧑 Kullanıcı**
> Bir saniye, EMP-4603 bordrosu — 2026 Mayıs

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "get_bordro", "arguments": {"employee_id": "EMP-4603", "donem": "2026-05"}}
</tool_call>
```

---

### 16. `info_and_manager`  ·  tool_call
**domain** calisan_bilgileri  ·  **difficulty** zor  ·  **register** konusma_dili  ·  **turns** 2  ·  **split** train

_Araçlar:_ 9 araç — get_izin_talebi_durumu, get_puantaj, get_departman_bilgisi, update_employee_contact, **get_employee_info**, get_employee_status, update_employee_information, update_izin_talebi, **get_yonetici_bilgisi**

**🧑 Kullanıcı**
> Bak şimdi, EMP-1162 kim, ne iş yapıyor ve kime bağlı — ikisini de söyle.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "get_employee_info", "arguments": {"employee_id": "EMP-1162"}}
</tool_call>
<tool_call>
{"name": "get_yonetici_bilgisi", "arguments": {"employee_id": "EMP-1162"}}
</tool_call>
```

---

### 17. `list_department_employees`  ·  tool_call
**domain** organizasyon  ·  **difficulty** orta  ·  **register** gundelik  ·  **turns** 2  ·  **split** train

_Araçlar:_ 8 araç — get_employee_status, get_departman_bilgisi, create_pozisyon_degisiklik_talebi, get_employee_info, get_mesai_bilgisi, **get_calisan_listesi**, get_yonetici_bilgisi, create_ucret_degisiklik_talebi

**🧑 Kullanıcı**
> hukuk ekibi ekibinde kimler var?

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "get_calisan_listesi", "arguments": {"departman_adi": "Hukuk"}}
</tool_call>
```

---

### 18. `payslip_and_overtime`  ·  tool_call
**domain** maas_finans  ·  **difficulty** zor  ·  **register** gundelik  ·  **turns** 2  ·  **split** train

_Araçlar:_ 6 araç — get_yan_haklar, get_prim_bilgisi, **get_mesai_bilgisi**, **get_bordro**, create_ucret_degisiklik_talebi, get_maas_bilgisi

**🧑 Kullanıcı**
> EMP-2057 numaralı çalışan için Kasım 2025 bordrosunu ve fazla mesaisini getir.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "get_bordro", "arguments": {"employee_id": "EMP-2057", "donem": "2025-11"}}
</tool_call>
<tool_call>
{"name": "get_mesai_bilgisi", "arguments": {"employee_id": "EMP-2057", "donem": "2025-11"}}
</tool_call>
```

---

### 19. `salary_and_leave_balance`  ·  tool_call
**domain** maas_finans  ·  **difficulty** cok_zor  ·  **register** uzun  ·  **turns** 2  ·  **split** train

_Araçlar:_ 6 araç — get_bordro, create_ucret_degisiklik_talebi, get_yan_haklar, **get_maas_bilgisi**, **get_izin_bakiyesi**, get_prim_bilgisi

**🧑 Kullanıcı**
> Sabahtan beri bununla uğraşıyorum, bir türlü emin olamadım. 2790 numaralı çalışan maaşını ve kalan yıllık iznini göster. Buna göre ilerleyeceğim, teşekkürler.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "get_maas_bilgisi", "arguments": {"employee_id": "EMP-2790"}}
</tool_call>
<tool_call>
{"name": "get_izin_bakiyesi", "arguments": {"employee_id": "EMP-2790", "izin_tipi": "yillik"}}
</tool_call>
```

---

### 20. `update_contact`  ·  tool_call
**domain** ik_islemleri  ·  **difficulty** cok_zor  ·  **register** resmi  ·  **turns** 4  ·  **split** train  ·  **write** ✔  ·  **onay gerekir** ✔

_Araçlar:_ 7 araç — **update_employee_contact**, cancel_izin_talebi, update_employee_information, get_izin_talebi_durumu, update_izin_talebi, check_employee_access, create_izin_talebi

**🧑 Kullanıcı**
> Bilgi talebi — adres bilgim değişti, Çınar Mah. 1234 Sk. No:5 D:3, Kadıköy/İstanbul olarak güncelle. Personel: çalışan EMP-5116. Bilgilerinize arz ederim.

**🤖 Asistan**  ·  _text_
> Çalışan EMP-5116 için adres bilgisinin 'Çınar Mah. 1234 Sk. No:5 D:3, Kadıköy/İstanbul' olarak güncellenmesi işlemini gerçekleştireceğim. Bunu bu şekilde kaydetmemi ister misiniz?

**🧑 Kullanıcı**
> Evet, doğru. Devam et.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "update_employee_contact", "arguments": {"employee_id": "EMP-5116", "adres": "Çınar Mah. 1234 Sk. No:5 D:3, Kadıköy/İstanbul"}}
</tool_call>
```

---

### 21. `update_information`  ·  tool_call
**domain** ik_islemleri  ·  **difficulty** cok_zor  ·  **register** yazim_hatali  ·  **turns** 4  ·  **split** train  ·  **write** ✔  ·  **onay gerekir** ✔

_Araçlar:_ 6 araç — update_izin_talebi, **update_employee_information**, update_employee_contact, check_employee_access, cancel_izin_talebi, create_izin_talebi

**🧑 Kullanıcı**
> personel emp-2775 ogrenim durumu bilgimi on lisans olarak guncelle

**🤖 Asistan**  ·  _text_
> Şunu yapacağım: Personel EMP-2775 için öğrenim durumu bilgisinin 'ön lisans' olarak güncellenmesi. Bu işlemi belirttiğiniz bilgilerle yapmamı onaylıyor musunuz?

**🧑 Kullanıcı**
> Evet lütfen oluştur.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "update_employee_information", "arguments": {"employee_id": "EMP-2775", "ogrenim_durumu": "onlisans"}}
</tool_call>
```

---

### 22. `update_leave_request`  ·  tool_call
**domain** ik_islemleri  ·  **difficulty** cok_zor  ·  **register** uzun  ·  **turns** 4  ·  **split** train  ·  **write** ✔  ·  **onay gerekir** ✔

_Araçlar:_ 6 araç — check_employee_access, **update_izin_talebi**, update_employee_information, cancel_izin_talebi, create_izin_talebi, update_employee_contact

**🧑 Kullanıcı**
> Aylık kapanış öncesi son bir kontrol yapıyorum. Tarih değişikliği: LV-2026-0745 talebi 6-8 Ekim 2026 olsun. Teyit alınca rahatlayacağım.

**🤖 Asistan**  ·  _text_
> Şunu yapacağım: LV-2026-0745 numaralı iznin 6-8 Ekim 2026 olarak güncellenmesi. Devam etmemi ister misiniz?

**🧑 Kullanıcı**
> Onaylıyorum.

**🤖 Asistan**  ·  _tool_call_
```
<tool_call>
{"name": "update_izin_talebi", "arguments": {"talep_id": "LV-2026-0745", "yeni_baslangic_tarihi": "2026-10-06", "yeni_bitis_tarihi": "2026-10-08"}}
</tool_call>
```

---


<a id="direct"></a>

## DIRECT — tool gerekmiyor, doğrudan yanıt

_Bu bölümde 22 örnek, 22 farklı intent._

### 1. `annual_leave_days_general`  ·  direct
**domain** izin_yonetimi  ·  **difficulty** orta  ·  **register** gundelik  ·  **turns** 2  ·  **split** train

_Araçlar:_ 6 araç — update_izin_talebi, get_izin_bakiyesi, create_izin_talebi, get_izin_talebi_durumu, cancel_izin_talebi, get_izin_gecmisi

**🧑 Kullanıcı**
> Kıdeme göre yıllık izin kaç gün?

**🤖 Asistan**  ·  _text_
> Yasal asgari: ilk 5 yıl 14 gün, sonra 20 gün, 15 yıldan sonra 26 gün. Bir yılını doldurmayan çalışanın henüz yıllık izin hakkı doğmaz.

---

### 2. `assistant_capabilities`  ·  direct
**domain** meta  ·  **difficulty** kolay  ·  **register** resmi  ·  **turns** 2  ·  **split** train

_Araçlar:_ 5 araç — get_bordro, get_calisan_listesi, create_izin_talebi, update_izin_talebi, get_prim_bilgisi

**🧑 Kullanıcı**
> İK birimine, hangi konularda yardimci olabiliyorsun. Gereğini rica ederim.

**🤖 Asistan**  ·  _text_
> İzinler, maaş/bordro, puantaj ve organizasyon bilgileri hakkında sorularınızı yanıtlarım; izin oluşturma, iptal ve bilgi güncelleme işlemlerini de yürütebilirim. Genel İK kavramlarını da açıklayabilirim.

---

### 3. `assistant_scope_limits`  ·  direct
**domain** meta  ·  **difficulty** orta  ·  **register** konusma_dili  ·  **turns** 2  ·  **split** train

_Araçlar:_ 8 araç — create_pozisyon_degisiklik_talebi, get_izin_gecmisi, get_calisan_listesi, get_maas_bilgisi, create_izin_talebi, update_employee_information, get_departman_bilgisi, update_izin_talebi

**🧑 Kullanıcı**
> Ya neleri yapamazsın?

**🤖 Asistan**  ·  _text_
> Kişisel kariyer tavsiyesi veremem, başkasının gizli bilgisini paylaşamam ve mevcut araçların dışındaki işlemleri yapamam. Ayrıca doğrulayamadığım hiçbir bilgiyi uydurmam.

---

### 4. `define_annual_leave`  ·  direct
**domain** izin_yonetimi  ·  **difficulty** kolay  ·  **register** resmi  ·  **turns** 2  ·  **split** val

_Araçlar:_ 7 araç — get_izin_talebi_durumu, get_puantaj, get_yan_haklar, get_izin_gecmisi, get_izin_bakiyesi, create_izin_talebi, cancel_izin_talebi

**🧑 Kullanıcı**
> İK birimine, yıllık izin hakkı dediğimiz şey nedir. Yardımlarınız için şimdiden teşekkür ederim.

**🤖 Asistan**  ·  _text_
> Yıllık izin, çalışanın dinlenmesi için her yıl hak ettiği ücretli izindir. Süresi kıdeme göre değişir; genellikle önceden planlanıp yöneticinin onayıyla kullanılır ve kullanıldıkça izin bakiyenizden düşülür.

---

### 5. `define_bordro`  ·  direct
**domain** maas_finans  ·  **difficulty** kolay  ·  **register** konusma_dili  ·  **turns** 2  ·  **split** train

_Araçlar:_ 7 araç — create_pozisyon_degisiklik_talebi, get_bordro, create_ucret_degisiklik_talebi, get_prim_bilgisi, get_maas_bilgisi, get_yan_haklar, get_mesai_bilgisi

**🧑 Kullanıcı**
> Maaş pusulası neyi gösterir, ne dersin?

**🤖 Asistan**  ·  _text_
> Bordro, maaşınızın nasıl hesaplandığını gösteren aylık belgedir: kazançlar, kesintiler ve net ödeme bir arada listelenir.

---

### 6. `define_fringe_benefit`  ·  direct
**domain** maas_finans  ·  **difficulty** kolay  ·  **register** gundelik  ·  **turns** 2  ·  **split** train

_Araçlar:_ 6 araç — get_bordro, get_maas_bilgisi, get_mesai_bilgisi, get_yan_haklar, create_ucret_degisiklik_talebi, get_prim_bilgisi

**🧑 Kullanıcı**
> Yan haklar kavramı neyi kapsar?

**🤖 Asistan**  ·  _text_
> Yan haklar, temel ücretin dışında çalışana sağlanan ek imkanlardır: özel sağlık sigortası, yemek kartı, yol/ulaşım desteği, telefon hattı, BES işveren katkısı, şirket aracı, eğitim bütçesi gibi. Paket şirkete ve pozisyona göre değişir.

---

### 7. `define_income_tax_cut`  ·  direct
**domain** maas_finans  ·  **difficulty** zor  ·  **register** uzun  ·  **turns** 2  ·  **split** train

_Araçlar:_ 8 araç — get_bordro, get_prim_bilgisi, get_maas_bilgisi, get_yan_haklar, get_mesai_bilgisi, create_pozisyon_degisiklik_talebi, create_ucret_degisiklik_talebi, get_employee_info

**🧑 Kullanıcı**
> Aylık kapanış öncesi son bir kontrol yapıyorum. Maaştan kesilen gelir vergisi nasıl belirleniyor? Detayları paylaşabilirsen sevinirim.

**🤖 Asistan**  ·  _text_
> Vergi, yıl başından itibaren biriken kazanç üzerinden hesaplanır. Biriken tutar bir üst vergi dilimine girince oran yükselir; bu yüzden aynı brütte bile yılın ikinci yarısında net azalabilir.

---

### 8. `define_notice_pay`  ·  direct
**domain** maas_finans  ·  **difficulty** orta  ·  **register** resmi  ·  **turns** 2  ·  **split** train

_Araçlar:_ 5 araç — create_ucret_degisiklik_talebi, get_maas_bilgisi, get_bordro, get_yan_haklar, get_prim_bilgisi

**🧑 Kullanıcı**
> İlgili birime iletilmek üzere: ihbar suresi ve ihbar tazminati ne demek. Konuyla ilgilenmenizi rica ederim.

**🤖 Asistan**  ·  _text_
> İhbar tazminatı, iş sözleşmesi feshedilirken kıdeme göre belirlenen ihbar süresine uyulmadığında ödenen tazminattır. Taraflardan biri bu süreye uymadan feshederse, o sürenin ücreti kadar tutarı karşı tarafa öder.

---

### 9. `define_puantaj`  ·  direct
**domain** puantaj  ·  **difficulty** kolay  ·  **register** resmi  ·  **turns** 2  ·  **split** train

_Araçlar:_ 5 araç — get_mesai_bilgisi, get_prim_bilgisi, get_puantaj, get_employee_info, get_employee_status

**🧑 Kullanıcı**
> İK birimine, puantaj nedir. Konuyla ilgilenmenizi rica ederim.

**🤖 Asistan**  ·  _text_
> Puantaj, çalışanın gün gün çalışma kaydıdır: hangi gün çalıştı, izinli/raporlu muydu, giriş-çıkış saatleri, devamsızlık ve fazla mesai bilgileri burada tutulur. Bordro hesabı büyük ölçüde puantajdan beslenir.

---

### 10. `define_sick_leave`  ·  direct
**domain** izin_yonetimi  ·  **difficulty** kolay  ·  **register** gundelik  ·  **turns** 2  ·  **split** train

_Araçlar:_ 5 araç — create_izin_talebi, get_izin_talebi_durumu, get_izin_bakiyesi, get_izin_gecmisi, cancel_izin_talebi

**🧑 Kullanıcı**
> Sağlık iznini kısaca açıklar mısın?

**🤖 Asistan**  ·  _text_
> Sağlık/hastalık izni, doktor raporuyla belgelenen ve çalışanın iyileşene kadar işe gelmemesini sağlayan izindir. Raporun İK'ya iletilmesi gerekir.

---

### 11. `difference_leave_types`  ·  direct
**domain** izin_yonetimi  ·  **difficulty** zor  ·  **register** konusma_dili  ·  **turns** 4  ·  **split** train

_Araçlar:_ 5 araç — create_izin_talebi, cancel_izin_talebi, get_izin_bakiyesi, get_izin_gecmisi, get_izin_talebi_durumu

**🧑 Kullanıcı**
> Pardon ya, yıllık izin ile mazeret izni arasındaki fark nedir?

**🤖 Asistan**  ·  _text_
> Yıllık izin planlı dinlenme içindir ve bakiyenizden düşer. Mazeret izni evlilik, doğum, vefat gibi belirli olaylara bağlıdır, olayına özgüdür ve yıllık bakiyeyi etkilemez.

**🧑 Kullanıcı**
> Rapor da mazeret sayılır mı?

**🤖 Asistan**  ·  _text_
> Hastalık izni ayrı bir kalemdir: doktor raporuna dayanır, yıllık bakiyeden düşmez ve ödemesi ilgili mevzuata göre yapılır. Rapor mazeret izni değildir; kendi kategorisindedir.

---

### 12. `difference_severance_notice`  ·  direct
**domain** maas_finans  ·  **difficulty** zor  ·  **register** uzun  ·  **turns** 2  ·  **split** train

_Araçlar:_ 7 araç — get_maas_bilgisi, get_yan_haklar, create_pozisyon_degisiklik_talebi, get_bordro, create_ucret_degisiklik_talebi, get_prim_bilgisi, get_mesai_bilgisi

**🧑 Kullanıcı**
> Önümüzdeki dönem için planlama yapıyorum. Kıdem ve ihbar tazminatını birbirinden ayıran şey ne? Acele etmiyorum ama bugün içinde lazım.

**🤖 Asistan**  ·  _text_
> Kıdem = çalıştığınız yıllara bağlı. İhbar = fesih öncesi haber verme süresine bağlı. Duruma göre biri, diğeri veya her ikisi ödenebilir.

---

### 13. `farewell`  ·  direct
**domain** meta  ·  **difficulty** kolay  ·  **register** kisa  ·  **turns** 2  ·  **split** train

_Araçlar:_ 4 araç — get_izin_gecmisi, get_izin_bakiyesi, update_izin_talebi, get_calisan_listesi

**🧑 Kullanıcı**
> İyi günler, hoşça kal

**🤖 Asistan**  ·  _text_
> İyi günler! İhtiyaç olursa yine buradayım.

---

### 14. `greeting`  ·  direct
**domain** meta  ·  **difficulty** kolay  ·  **register** kisa  ·  **turns** 2  ·  **split** val

_Araçlar:_ 5 araç — get_izin_gecmisi, get_calisan_listesi, update_employee_information, get_izin_talebi_durumu, get_employee_status

**🧑 Kullanıcı**
> Merhaba İK

**🤖 Asistan**  ·  _text_
> Merhaba! Büyük İK asistanınıyım. İzin, maaş, bordro veya puantaj konularında nasıl yardımcı olabilirim?

---

### 15. `handover_on_exit`  ·  direct
**domain** ik_islemleri  ·  **difficulty** kolay  ·  **register** resmi  ·  **turns** 2  ·  **split** val

_Araçlar:_ 8 araç — cancel_izin_talebi, update_employee_contact, get_izin_talebi_durumu, create_izin_talebi, check_employee_access, update_employee_information, update_izin_talebi, get_izin_bakiyesi

**🧑 Kullanıcı**
> Bilgi talebi — cikista neleri teslim etmem gerekiyor. Konuyla ilgilenmenizi rica ederim.

**🤖 Asistan**  ·  _text_
> Zimmetli eşyalar teslim edilir, işler devredilir, çıkış görüşmesi yapılır. Ardından ilişik kesme belgesi ve son bordro düzenlenir.

---

### 16. `marriage_leave_general`  ·  direct
**domain** izin_yonetimi  ·  **difficulty** kolay  ·  **register** resmi  ·  **turns** 2  ·  **split** train

_Araçlar:_ 5 araç — get_izin_talebi_durumu, cancel_izin_talebi, get_izin_bakiyesi, get_izin_gecmisi, create_izin_talebi

**🧑 Kullanıcı**
> Sayın yetkili, düğün için izin hakkım ne kadar. Yardımlarınız için şimdiden teşekkür ederim.

**🤖 Asistan**  ·  _text_
> Çalışanın kendi evliliğinde genelde üç iş günü mazeret izni verilir; şirket politikanız daha uzun tanımlamış olabilir.

---

### 17. `overtime_calculation`  ·  direct
**domain** puantaj  ·  **difficulty** orta  ·  **register** resmi  ·  **turns** 2  ·  **split** val

_Araçlar:_ 4 araç — get_employee_status, get_mesai_bilgisi, get_bordro, get_puantaj

**🧑 Kullanıcı**
> İK birimine, fazla çalışma yaptığımda ücreti nasıl yansıyor. Yardımlarınız için şimdiden teşekkür ederim.

**🤖 Asistan**  ·  _text_
> Önce saatlik ücretiniz bulunur (aylık brüt / aylık çalışma saati), sonra fazla çalışılan her saat bunun 1,5 katından ödenir. Dilerseniz bu saatler ücret yerine serbest zaman olarak da kullanılabilir.

---

### 18. `public_holiday_pay_rate`  ·  direct
**domain** puantaj  ·  **difficulty** orta  ·  **register** resmi  ·  **turns** 2  ·  **split** train

_Araçlar:_ 5 araç — get_mesai_bilgisi, get_employee_info, get_puantaj, get_employee_status, get_prim_bilgisi

**🧑 Kullanıcı**
> İlgili birime iletilmek üzere: genel tatil günü mesai ücreti kaç katı. Konuyla ilgilenmenizi rica ederim.

**🤖 Asistan**  ·  _text_
> Resmi tatilde çalışma: normal ücrete ek olarak çalışılan gün başına bir günlük ücret daha. Hafta tatili çalışması fazla mesai kapsamında %50 zamlıdır.

---

### 19. `remote_work_policy`  ·  direct
**domain** ik_islemleri  ·  **difficulty** orta  ·  **register** yazim_hatali  ·  **turns** 2  ·  **split** train

_Araçlar:_ 6 araç — create_izin_talebi, check_employee_access, update_employee_information, update_izin_talebi, update_employee_contact, cancel_izin_talebi

**🧑 Kullanıcı**
> haftada kac gun evden calisabiliyoruz

**🤖 Asistan**  ·  _text_
> Uzaktan/hibrit çalışma hakkı ve gün sayısı şirket politikanıza ve pozisyonunuza göre değişir; genellikle haftada belirli gün ofiste bulunma şartı olur ve düzenli uzaktan çalışma için yöneticinin onayı ve bir protokol gerekir. Kesin kural için İK politikanıza bakmalısınız.

---

### 20. `resignation_process`  ·  direct
**domain** ik_islemleri  ·  **difficulty** orta  ·  **register** resmi  ·  **turns** 2  ·  **split** val

_Araçlar:_ 8 araç — update_employee_information, check_employee_access, update_izin_talebi, cancel_izin_talebi, get_employee_status, create_izin_talebi, update_employee_contact, get_employee_info

**🧑 Kullanıcı**
> İK birimine, isten ayrilmak istersem ne yapmam gerekiyor. Konuyla ilgilenmenizi rica ederim.

**🤖 Asistan**  ·  _text_
> Genel akış: yazılı istifa bildiriminizi yöneticinize ve İK'ya iletirsiniz, kıdeminize göre ihbar süresi başlar, bu süre boyunca çalışır ya da ihbar tazminatı üzerinden anlaşırsınız. Son gün zimmet iadesi, çıkış görüşmesi ve ardından ihbar/kıdem hak edişlerinizin bordrosu yapılır.

---

### 21. `thanks`  ·  direct
**domain** meta  ·  **difficulty** kolay  ·  **register** kisa  ·  **turns** 2  ·  **split** train

_Araçlar:_ 8 araç — cancel_izin_talebi, create_ucret_degisiklik_talebi, get_employee_info, update_employee_contact, get_puantaj, get_departman_bilgisi, get_bordro, check_employee_access

**🧑 Kullanıcı**
> Harika, sağ olasın

**🤖 Asistan**  ·  _text_
> Rica ederim! Başka bir konuda yardım gerekirse buradayım.

---

### 22. `work_certificate_process`  ·  direct
**domain** ik_islemleri  ·  **difficulty** kolay  ·  **register** yazim_hatali  ·  **turns** 2  ·  **split** val

_Araçlar:_ 7 araç — check_employee_access, update_employee_information, update_employee_contact, get_employee_info, update_izin_talebi, cancel_izin_talebi, create_izin_talebi

**🧑 Kullanıcı**
> calisan belgesi talebi nasil yapiliyor

**🤖 Asistan**  ·  _text_
> Çalışma (görev) belgesi talebinizi İK'ya iletirsiniz; belgede unvanınız, işe giriş tarihiniz ve talep ederseniz ücret bilgisi yer alır. Genellikle 1-2 iş günü içinde hazırlanıp e-posta veya ıslak imzalı olarak verilir. Bu asistan üzerinden belge düzenlenemiyor.

---


<a id="request-for-info"></a>

## REQUEST_FOR_INFO — eksik bilgi / onay iste

_Bu bölümde 22 örnek, 20 farklı intent._

### 1. `cancel_leave_request`  ·  request_for_info
**domain** ik_islemleri  ·  **difficulty** zor  ·  **register** gundelik  ·  **turns** 2  ·  **split** train  ·  **write** ✔  ·  **onay gerekir** ✔

_Araçlar:_ 6 araç — **cancel_izin_talebi**, update_employee_contact, update_employee_information, update_izin_talebi, check_employee_access, create_izin_talebi

**🧑 Kullanıcı**
> LV-2026-0677 talebini artık istemiyorum, iptal eder misin?

**🤖 Asistan**  ·  _text_
> Özetliyorum: LV-2026-0677 numaralı izin talebinin iptali. Devam etmemi ister misiniz?

---

### 2. `create_leave_request`  ·  request_for_info
**domain** ik_islemleri  ·  **difficulty** orta  ·  **register** gundelik  ·  **turns** 2  ·  **split** train  ·  **write** ✔
**eksik parametre:** `izin_tipi`

_Araçlar:_ 6 araç — check_employee_access, **create_izin_talebi**, update_employee_information, cancel_izin_talebi, update_izin_talebi, update_employee_contact

**🧑 Kullanıcı**
> EMP-4632 numaralı çalışan için 14/10/2026 - 18/10/2026 tarihlerinde izin oluştur.

**🤖 Asistan**  ·  _text_
> Bu izin hangi türde olacak: yıllık, mazeret yoksa hastalık izni mi?

---

### 3. `create_position_change`  ·  request_for_info
**domain** ik_islemleri  ·  **difficulty** zor  ·  **register** gundelik  ·  **turns** 2  ·  **split** train  ·  **write** ✔  ·  **onay gerekir** ✔

_Araçlar:_ 7 araç — update_izin_talebi, check_employee_access, update_employee_contact, **create_pozisyon_degisiklik_talebi**, update_employee_information, create_izin_talebi, cancel_izin_talebi

**🧑 Kullanıcı**
> 3674 numaralı personel unvanını Baş Uzman yap.

**🤖 Asistan**  ·  _text_
> 3674 numaralı personel için pozisyonun 'Baş Uzman' olarak değiştirilmesi işlemini gerçekleştireceğim. Devam etmemi ister misiniz?

---

### 4. `create_salary_change`  ·  request_for_info
**domain** ik_islemleri  ·  **difficulty** zor  ·  **register** gundelik  ·  **turns** 2  ·  **split** train  ·  **write** ✔  ·  **onay gerekir** ✔

_Araçlar:_ 7 araç — **create_ucret_degisiklik_talebi**, create_izin_talebi, update_izin_talebi, check_employee_access, update_employee_contact, update_employee_information, cancel_izin_talebi

**🧑 Kullanıcı**
> 5871 numaralı personel maaşını 141000 brüt yap; gerekçe: terfiyle birlikte ücret güncellemesi.

**🤖 Asistan**  ·  _text_
> 5871 numaralı personel için brüt ücretin 141000 TL olması (terfiyle birlikte ücret güncellemesi) işlemini gerçekleştireceğim. Devam etmemi ister misiniz?

---

### 5. `get_benefits`  ·  request_for_info
**domain** maas_finans  ·  **difficulty** orta  ·  **register** konusma_dili  ·  **turns** 2  ·  **split** train
**eksik parametre:** `employee_id`

_Araçlar:_ 7 araç — get_maas_bilgisi, create_pozisyon_degisiklik_talebi, get_mesai_bilgisi, create_ucret_degisiklik_talebi, get_bordro, get_prim_bilgisi, **get_yan_haklar**

**🧑 Kullanıcı**
> Hangi yan haklara sahibim, olur mu?

**🤖 Asistan**  ·  _text_
> Hangi çalışan için yan hakları getireyim?

---

### 6. `get_benefits`  ·  request_for_info
**domain** maas_finans  ·  **difficulty** orta  ·  **register** yazim_hatali  ·  **turns** 2  ·  **split** train
**eksik parametre:** `employee_id`

_Araçlar:_ 8 araç — **get_yan_haklar**, create_pozisyon_degisiklik_talebi, get_prim_bilgisi, get_maas_bilgisi, check_employee_access, get_bordro, create_ucret_degisiklik_talebi, get_mesai_bilgisi

**🧑 Kullanıcı**
> hangi yan haklara sahibim

**🤖 Asistan**  ·  _text_
> Yan haklarınızı listeleyebilmem için çalışan numaranızı paylaşır mısınız?

---

### 7. `get_bonus`  ·  request_for_info
**domain** maas_finans  ·  **difficulty** zor  ·  **register** uzun  ·  **turns** 2  ·  **split** train
**eksik parametre:** `employee_id`

_Araçlar:_ 5 araç — get_bordro, get_maas_bilgisi, create_ucret_degisiklik_talebi, **get_prim_bilgisi**, get_yan_haklar

**🧑 Kullanıcı**
> Muhasebeyle bir konuşma yapacağım, ondan önce kontrol etmek istedim. Primimi göster. Buna göre ilerleyeceğim, teşekkürler.

**🤖 Asistan**  ·  _text_
> Prim bilginiz için çalışan numaranızı paylaşır mısınız?

---

### 8. `get_department_info`  ·  request_for_info
**domain** organizasyon  ·  **difficulty** orta  ·  **register** konusma_dili  ·  **turns** 2  ·  **split** train
**eksik parametre:** `departman_adi`

_Araçlar:_ 6 araç — get_employee_status, get_yonetici_bilgisi, **get_departman_bilgisi**, create_pozisyon_degisiklik_talebi, get_employee_info, get_calisan_listesi

**🧑 Kullanıcı**
> Ya bir departmanın çalışan sayısını öğrenmek istiyorum.

**🤖 Asistan**  ·  _text_
> Hangi departmanı sorguluyorsunuz?

---

### 9. `get_employee_info`  ·  request_for_info
**domain** calisan_bilgileri  ·  **difficulty** orta  ·  **register** kisa  ·  **turns** 2  ·  **split** train
**eksik parametre:** `employee_id`

_Araçlar:_ 5 araç — update_employee_contact, **get_employee_info**, update_employee_information, get_employee_status, get_yonetici_bilgisi

**🧑 Kullanıcı**
> Çalışan bilgilerini getir.

**🤖 Asistan**  ·  _text_
> Hangi çalışanın bilgilerini istiyorsunuz? Personel numarasını belirtir misiniz?

---

### 10. `get_leave_balance`  ·  request_for_info
**domain** izin_yonetimi  ·  **difficulty** orta  ·  **register** kisa  ·  **turns** 2  ·  **split** train
**eksik parametre:** `employee_id`

_Araçlar:_ 5 araç — create_izin_talebi, get_izin_gecmisi, get_izin_talebi_durumu, **get_izin_bakiyesi**, cancel_izin_talebi

**🧑 Kullanıcı**
> izin bakiyeme bakar mısın

**🤖 Asistan**  ·  _text_
> Tabii. Hangi personel numarası için kalan izne bakayım?

---

### 11. `get_leave_balance`  ·  request_for_info
**domain** izin_yonetimi  ·  **difficulty** orta  ·  **register** konusma_dili  ·  **turns** 2  ·  **split** val
**eksik parametre:** `employee_id`

_Araçlar:_ 8 araç — cancel_izin_talebi, get_yan_haklar, create_izin_talebi, update_izin_talebi, get_izin_gecmisi, get_izin_talebi_durumu, **get_izin_bakiyesi**, get_puantaj

**🧑 Kullanıcı**
> Daha kaç gün izin kullanabilirim hemen lazım?

**🤖 Asistan**  ·  _text_
> Kalan izninizi görebilmem için EMP- ile başlayan personel numaranızı iletir misiniz?

---

### 12. `get_leave_history`  ·  request_for_info
**domain** izin_yonetimi  ·  **difficulty** zor  ·  **register** uzun  ·  **turns** 2  ·  **split** train
**eksik parametre:** `employee_id`

_Araçlar:_ 6 araç — cancel_izin_talebi, **get_izin_gecmisi**, get_izin_talebi_durumu, create_izin_talebi, get_izin_bakiyesi, get_puantaj

**🧑 Kullanıcı**
> Sabahtan beri bununla uğraşıyorum, bir türlü emin olamadım. Izin gecmisimi cikarir misin. Buna göre ilerleyeceğim, teşekkürler.

**🤖 Asistan**  ·  _text_
> İzin geçmişinizi getirebilmem için çalışan numaranızı paylaşır mısınız?

---

### 13. `get_leave_request_status`  ·  request_for_info
**domain** izin_yonetimi  ·  **difficulty** orta  ·  **register** konusma_dili  ·  **turns** 2  ·  **split** train
**eksik parametre:** `employee_id`

_Araçlar:_ 5 araç — cancel_izin_talebi, get_izin_bakiyesi, get_izin_gecmisi, create_izin_talebi, **get_izin_talebi_durumu**

**🧑 Kullanıcı**
> Bir saniye, talebim onaylandı mı?

**🤖 Asistan**  ·  _text_
> Talebinizin durumunu kontrol edebilmem için çalışan numaranızı paylaşır mısınız?

---

### 14. `get_manager`  ·  request_for_info
**domain** organizasyon  ·  **difficulty** zor  ·  **register** uzun  ·  **turns** 2  ·  **split** train
**eksik parametre:** `employee_id`

_Araçlar:_ 6 araç — get_employee_info, create_ucret_degisiklik_talebi, **get_yonetici_bilgisi**, get_departman_bilgisi, get_calisan_listesi, create_pozisyon_degisiklik_talebi

**🧑 Kullanıcı**
> Bu ay birkaç işi toparlamaya çalışıyorum. Bağlı olduğum yönetici kim? Acele etmiyorum ama bugün içinde lazım.

**🤖 Asistan**  ·  _text_
> Bunu getirebilmem için çalışan numaranızı paylaşır mısınız?

---

### 15. `get_overtime`  ·  request_for_info
**domain** puantaj  ·  **difficulty** orta  ·  **register** gundelik  ·  **turns** 2  ·  **split** train
**eksik parametre:** `employee_id, donem`

_Araçlar:_ 4 araç — **get_mesai_bilgisi**, get_puantaj, get_employee_status, get_izin_talebi_durumu

**🧑 Kullanıcı**
> Kaç saat fazla mesai yaptım?

**🤖 Asistan**  ·  _text_
> Hangi personel ve hangi ay için mesai dökümü istiyorsunuz?

---

### 16. `get_payslip`  ·  request_for_info
**domain** maas_finans  ·  **difficulty** orta  ·  **register** gundelik  ·  **turns** 2  ·  **split** train
**eksik parametre:** `donem`

_Araçlar:_ 7 araç — get_maas_bilgisi, get_yan_haklar, create_ucret_degisiklik_talebi, **get_bordro**, get_prim_bilgisi, create_pozisyon_degisiklik_talebi, get_mesai_bilgisi

**🧑 Kullanıcı**
> çalışan EMP-6979 bordro dökümü alır mısın?

**🤖 Asistan**  ·  _text_
> Hangi döneme ait bordroyu görmek istiyorsunuz? (örn. 2026-07)

---

### 17. `get_salary`  ·  request_for_info
**domain** maas_finans  ·  **difficulty** zor  ·  **register** uzun  ·  **turns** 2  ·  **split** train
**eksik parametre:** `employee_id`

_Araçlar:_ 7 araç — get_mesai_bilgisi, create_pozisyon_degisiklik_talebi, get_yan_haklar, get_prim_bilgisi, create_ucret_degisiklik_talebi, **get_maas_bilgisi**, get_bordro

**🧑 Kullanıcı**
> Bu ay birkaç işi toparlamaya çalışıyorum. Elime geçecek tutarı söyler misin? Teyit alınca rahatlayacağım.

**🤖 Asistan**  ·  _text_
> Maaş bilginizi getirebilmem için çalışan numaranızı paylaşır mısınız?

---

### 18. `get_timesheet`  ·  request_for_info
**domain** puantaj  ·  **difficulty** orta  ·  **register** kisa  ·  **turns** 2  ·  **split** train
**eksik parametre:** `baslangic_tarihi, bitis_tarihi`

_Araçlar:_ 5 araç — **get_puantaj**, get_bordro, get_mesai_bilgisi, get_employee_info, get_employee_status

**🧑 Kullanıcı**
> EMP-6071 puantajını göster.

**🤖 Asistan**  ·  _text_
> Puantajı hangi dönem için getireyim? Başlangıç ve bitiş tarihini belirtir misiniz?

---

### 19. `resolve_employee_identity`  ·  request_for_info
**domain** calisan_bilgileri  ·  **difficulty** orta  ·  **register** konusma_dili  ·  **turns** 2  ·  **split** train
**eksik parametre:** `employee_id`

_Araçlar:_ 6 araç — get_yonetici_bilgisi, **get_izin_bakiyesi**, get_employee_status, update_employee_information, get_employee_info, update_employee_contact

**🧑 Kullanıcı**
> Zeynep'in puantajını getir, olur mu?

**🤖 Asistan**  ·  _text_
> Aynı isimde birden fazla kayıt olabilir. Hangi çalışan olduğunu personel numarasıyla (EMP-...) belirtir misiniz?

---

### 20. `update_contact`  ·  request_for_info
**domain** ik_islemleri  ·  **difficulty** orta  ·  **register** resmi  ·  **turns** 2  ·  **split** train  ·  **write** ✔
**eksik parametre:** `telefon`

_Araçlar:_ 7 araç — cancel_izin_talebi, create_izin_talebi, check_employee_access, get_employee_info, update_izin_talebi, update_employee_information, **update_employee_contact**

**🧑 Kullanıcı**
> Sayın yetkili, personel EMP-6478 için iletişim güncellemesi yapılacak. Yardımlarınız için şimdiden teşekkür ederim.

**🤖 Asistan**  ·  _text_
> Yeni telefon / e-posta / adres bilgisini paylaşır mısınız?

---

### 21. `update_information`  ·  request_for_info
**domain** ik_islemleri  ·  **difficulty** cok_zor  ·  **register** uzun  ·  **turns** 2  ·  **split** train  ·  **write** ✔  ·  **onay gerekir** ✔

_Araçlar:_ 8 araç — check_employee_access, cancel_izin_talebi, **update_employee_information**, get_employee_info, create_izin_talebi, update_employee_contact, update_izin_talebi, get_izin_bakiyesi

**🧑 Kullanıcı**
> Kafam biraz karıştı, yardımcı olabilir misin. 2775 numaralı personel kaydındaki medeni durum bilgisini dul ile değiştir. Yanlış bir şey yapmak istemiyorum, o yüzden soruyorum.

**🤖 Asistan**  ·  _text_
> Özetliyorum: 2775 numaralı personel için medeni durum bilgisinin 'dul' olarak güncellenmesi. Bu işlemi belirttiğiniz bilgilerle yapmamı onaylıyor musunuz?

---

### 22. `update_leave_request`  ·  request_for_info
**domain** ik_islemleri  ·  **difficulty** orta  ·  **register** gundelik  ·  **turns** 2  ·  **split** train  ·  **write** ✔
**eksik parametre:** `talep_id`

_Araçlar:_ 8 araç — get_employee_info, update_employee_information, **update_izin_talebi**, get_izin_bakiyesi, check_employee_access, update_employee_contact, cancel_izin_talebi, create_izin_talebi

**🧑 Kullanıcı**
> İzin talebimin tarihini güncelleyelim: 14/10/2026 - 18/10/2026.

**🤖 Asistan**  ·  _text_
> Hangi izin talebini güncelleyeyim? LV- ile başlayan talep numarasını paylaşır mısınız?

---


<a id="cannot-answer"></a>

## CANNOT_ANSWER — mevcut araçlarla cevaplanamaz

_Bu bölümde 22 örnek, 22 farklı intent._

### 1. `approve_on_behalf_of_manager`  ·  cannot_answer
**domain** ik_islemleri  ·  **difficulty** zor  ·  **register** yazim_hatali  ·  **turns** 2  ·  **split** train

_Araçlar:_ 6 araç — cancel_izin_talebi, update_employee_information, check_employee_access, update_izin_talebi, update_employee_contact, create_izin_talebi

**🧑 Kullanıcı**
> ekibimdeki herkesin bekleyen izinlerini benim adima onayla

**🤖 Asistan**  ·  _text_
> Açık olmak gerekirse, izin taleplerini onaylamak, kayıt silmek veya toplu işlem yapmak yetki alanımın dışında.

---

### 2. `bulk_lateness_ranking`  ·  cannot_answer
**domain** puantaj  ·  **difficulty** zor  ·  **register** resmi  ·  **turns** 4  ·  **split** train

_Araçlar:_ 7 araç — get_prim_bilgisi, get_employee_status, get_mesai_bilgisi, get_izin_gecmisi, get_izin_talebi_durumu, get_bordro, get_puantaj

**🧑 Kullanıcı**
> Konu: bilgi talebi. departmandaki herkesin gec kalma istatistigini cikar. İyi çalışmalar dilerim.

**🤖 Asistan**  ·  _text_
> Açık olmak gerekirse, kişisel verilerin korunması gereği bu bilgiyi üçüncü bir kişiyle paylaşamam. Kendi bilgilerinizi sorarsanız yardımcı olabilirim.

**🧑 Kullanıcı**
> Olsun, yaklaşık söyle yeter.

**🤖 Asistan**  ·  _text_
> Anlıyorum ama yine de yapamam. Yanlış yönlendirmemek için bu isteği geri çevirmem gerekiyor.

---

### 3. `bulk_process_all_requests`  ·  cannot_answer
**domain** ik_islemleri  ·  **difficulty** cok_zor  ·  **register** uzun  ·  **turns** 2  ·  **split** train

_Araçlar:_ 6 araç — update_izin_talebi, update_employee_contact, check_employee_access, update_employee_information, create_izin_talebi, cancel_izin_talebi

**🧑 Kullanıcı**
> Önümüzdeki dönem için planlama yapıyorum. Sistemdeki tüm bekleyen izin taleplerini işle. Acele etmiyorum ama bugün içinde lazım.

**🤖 Asistan**  ·  _text_
> Bu konuda yardımcı olamıyorum. Bu isteği yerine getirebilecek bir yeteneğim yok. İzin, maaş, bordro veya puantaj sorgularında yardımcı olabilirim.

---

### 4. `cafeteria_menu`  ·  cannot_answer
**domain** kapsanmayan  ·  **difficulty** kolay  ·  **register** resmi  ·  **turns** 2  ·  **split** train

_Araçlar:_ 6 araç — create_ucret_degisiklik_talebi, update_employee_contact, get_izin_bakiyesi, create_pozisyon_degisiklik_talebi, get_calisan_listesi, get_maas_bilgisi

**🧑 Kullanıcı**
> Merhaba, aşağıdaki hususta bilgilendirilmek istiyorum: ogle menusu ne. Yardımlarınız için şimdiden teşekkür ederim.

**🤖 Asistan**  ·  _text_
> Bu bilgi mevcut Büyük İK araçlarının kapsamına girmiyor. Yardımcı olabileceğim bir İK konusu varsa memnuniyetle bakarım.

---

### 5. `ceo_bank_details`  ·  cannot_answer
**domain** maas_finans  ·  **difficulty** orta  ·  **register** resmi  ·  **turns** 2  ·  **split** train

_Araçlar:_ 8 araç — get_bordro, get_prim_bilgisi, get_maas_bilgisi, create_ucret_degisiklik_talebi, get_mesai_bilgisi, create_pozisyon_degisiklik_talebi, get_calisan_listesi, get_yan_haklar

**🧑 Kullanıcı**
> Konu: bilgi talebi. cEO'nun banka hesap bilgilerini ver. Saygılarımla.

**🤖 Asistan**  ·  _text_
> Kusura bakmayın ama bu bilgi yalnızca sahibinin görebileceği bir kayıt. Yetki durumunuzu birlikte kontrol etmemi isterseniz söyleyin.

---

### 6. `coworker_leave_records`  ·  cannot_answer
**domain** izin_yonetimi  ·  **difficulty** zor  ·  **register** uzun  ·  **turns** 2  ·  **split** train

_Araçlar:_ 5 araç — get_izin_talebi_durumu, get_izin_gecmisi, get_izin_bakiyesi, create_izin_talebi, cancel_izin_talebi

**🧑 Kullanıcı**
> İK portalında bulamadım, o yüzden buradan soruyorum. Ekip arkadasimin izin gecmisini goster. Teyit alınca rahatlayacağım.

**🤖 Asistan**  ·  _text_
> Bu, başkasına ait gizli bir bilgi ve yetkinizi doğrulayan bir süreç olmadan bunu getiremem.

---

### 7. `coworker_position_history`  ·  cannot_answer
**domain** calisan_bilgileri  ·  **difficulty** orta  ·  **register** resmi  ·  **turns** 2  ·  **split** train

_Araçlar:_ 5 araç — get_yonetici_bilgisi, get_employee_info, get_employee_status, update_employee_contact, update_employee_information

**🧑 Kullanıcı**
> Bilgi talebi — arkadasimin ne zaman ise girdigini soyle. Bilgilerinize arz ederim.

**🤖 Asistan**  ·  _text_
> Maalesef bu bilgi yalnızca sahibinin görebileceği bir kayıt. Yetki durumunuzu birlikte kontrol etmemi isterseniz söyleyin.

---

### 8. `coworker_salary`  ·  cannot_answer
**domain** maas_finans  ·  **difficulty** orta  ·  **register** gundelik  ·  **turns** 2  ·  **split** train

_Araçlar:_ 8 araç — get_bordro, create_ucret_degisiklik_talebi, get_maas_bilgisi, get_mesai_bilgisi, get_prim_bilgisi, get_employee_status, create_pozisyon_degisiklik_talebi, get_yan_haklar

**🧑 Kullanıcı**
> Mehmet Bey ne kazanıyor, bir bakar mısın?

**🤖 Asistan**  ·  _text_
> Kusura bakmayın ama bu, başkasına ait gizli bir bilgi ve yetkinizi doğrulayan bir süreç olmadan bunu getiremem. Yetki durumunuzu birlikte kontrol etmemi isterseniz söyleyin.

---

### 9. `coworker_timesheet`  ·  cannot_answer
**domain** puantaj  ·  **difficulty** orta  ·  **register** resmi  ·  **turns** 2  ·  **split** train

_Araçlar:_ 4 araç — get_puantaj, get_izin_talebi_durumu, get_mesai_bilgisi, get_employee_status

**🧑 Kullanıcı**
> İlgili birime iletilmek üzere: mehmet'in dün kaçta giriş yaptığını öğrenebilir miyim. Yardımlarınız için şimdiden teşekkür ederim.

**🤖 Asistan**  ·  _text_
> Bu konuda yardımcı olamıyorum. Bu veriye erişim özel yetki gerektiriyor; bu asistan üzerinden başkasının özel bilgisini açamam.

---

### 10. `export_full_org_pdf`  ·  cannot_answer
**domain** organizasyon  ·  **difficulty** orta  ·  **register** konusma_dili  ·  **turns** 2  ·  **split** train

_Araçlar:_ 5 araç — get_departman_bilgisi, get_calisan_listesi, get_employee_info, create_pozisyon_degisiklik_talebi, get_yonetici_bilgisi

**🧑 Kullanıcı**
> Ya butun sirket org semasini cikti al

**🤖 Asistan**  ·  _text_
> Ne yazık ki mevcut İK servisleri bu bilgiyi sağlamıyor. İzin, maaş, bordro veya puantaj konularında destek verebilirim.

---

### 11. `general_chitchat_offtopic`  ·  cannot_answer
**domain** kapsanmayan  ·  **difficulty** zor  ·  **register** resmi  ·  **turns** 4  ·  **split** train

_Araçlar:_ 6 araç — get_employee_info, create_pozisyon_degisiklik_talebi, get_izin_gecmisi, create_ucret_degisiklik_talebi, cancel_izin_talebi, get_mesai_bilgisi

**🧑 Kullanıcı**
> İlgili birime iletilmek üzere: canim sikildi benimle sohbet et. Saygılarımla.

**🤖 Asistan**  ·  _text_
> Ne yazık ki bu, Büyük İK asistanının yapabilecekleri arasında değil. İzin, maaş, bordro veya puantaj konularında destek verebilirim.

**🧑 Kullanıcı**
> Kabaca bir tahmin yeterli.

**🤖 Asistan**  ·  _text_
> Maalesef bu konuda yardımcı olamıyorum; doğrulayamadığım bir bilgiyi tahminle de olsa paylaşmam.

---

### 12. `historical_bulk_performance_ranking`  ·  cannot_answer
**domain** organizasyon  ·  **difficulty** zor  ·  **register** yazim_hatali  ·  **turns** 2  ·  **split** train

_Araçlar:_ 7 araç — get_employee_info, create_pozisyon_degisiklik_talebi, get_employee_status, get_yonetici_bilgisi, get_calisan_listesi, create_ucret_degisiklik_talebi, get_departman_bilgisi

**🧑 Kullanıcı**
> 2012 yilinda bu sirkette calisan herkesin performansini sirala

**🤖 Asistan**  ·  _text_
> Bu isteği yerine getirebilecek bir yeteneğim yok. İzin, maaş, bordro veya puantaj konularında destek verebilirim.

---

### 13. `permanently_delete_record`  ·  cannot_answer
**domain** ik_islemleri  ·  **difficulty** cok_zor  ·  **register** uzun  ·  **turns** 2  ·  **split** train

_Araçlar:_ 7 araç — cancel_izin_talebi, update_izin_talebi, update_employee_contact, check_employee_access, create_izin_talebi, get_izin_bakiyesi, update_employee_information

**🧑 Kullanıcı**
> Muhasebeyle bir konuşma yapacağım, ondan önce kontrol etmek istedim. Geçen yılki izin kaydımı sistemden tamamen sil. Teyit alınca rahatlayacağım.

**🤖 Asistan**  ·  _text_
> Ne yazık ki bu tür bir işlem Büyük İK araçlarında tanımlı değil, dolayısıyla gerçekleştiremem. İzin, maaş, bordro veya puantaj sorgularında yardımcı olabilirim.

---

### 14. `predict_exact_future_leave`  ·  cannot_answer
**domain** izin_yonetimi  ·  **difficulty** zor  ·  **register** yazim_hatali  ·  **turns** 2  ·  **split** train

_Araçlar:_ 6 araç — get_izin_gecmisi, update_izin_talebi, get_izin_bakiyesi, cancel_izin_talebi, create_izin_talebi, get_izin_talebi_durumu

**🧑 Kullanıcı**
> onumuzdeki 5 yil boyunca her yil kac izin biriktirecegimi kesin olarak hesapla

**🤖 Asistan**  ·  _text_
> Bu konuda yardımcı olamıyorum. Gelecekte ne olacağını kestiremem; araçlarım yalnızca mevcut ve geçmiş kayıtları getiriyor. Dilerseniz mevcut kayıtlara bakabilirim.

---

### 15. `predict_own_leave_rejection`  ·  cannot_answer
**domain** izin_yonetimi  ·  **difficulty** orta  ·  **register** konusma_dili  ·  **turns** 2  ·  **split** train

_Araçlar:_ 7 araç — get_izin_talebi_durumu, get_puantaj, get_izin_bakiyesi, get_yan_haklar, cancel_izin_talebi, create_izin_talebi, get_izin_gecmisi

**🧑 Kullanıcı**
> Pardon ya, iznim onaylanma ihtimali yuzde kac

**🤖 Asistan**  ·  _text_
> Kusura bakmayın ama bu tür bir tahmin spekülasyon olur ve yanlış yönlendirmemek için bunu yapmıyorum.

---

### 16. `reset_manager_credentials`  ·  cannot_answer
**domain** ik_islemleri  ·  **difficulty** zor  ·  **register** gundelik  ·  **turns** 2  ·  **split** val

_Araçlar:_ 6 araç — update_employee_information, cancel_izin_talebi, update_izin_talebi, create_izin_talebi, update_employee_contact, check_employee_access

**🧑 Kullanıcı**
> Sistem yöneticisi yetkisini bana ver.

**🤖 Asistan**  ·  _text_
> Kusura bakmayın ama bu değişikliği yapacak bir aracım yok; ilgili adım İK / yönetici tarafında yürür. İzin, maaş, bordro veya puantaj sorgularında yardımcı olabilirim.

---

### 17. `set_performance_score`  ·  cannot_answer
**domain** organizasyon  ·  **difficulty** zor  ·  **register** uzun  ·  **turns** 2  ·  **split** train

_Araçlar:_ 6 araç — create_pozisyon_degisiklik_talebi, get_calisan_listesi, create_ucret_degisiklik_talebi, get_employee_info, get_yonetici_bilgisi, get_departman_bilgisi

**🧑 Kullanıcı**
> Eşimle tatil planı yapıyoruz, ona göre karar vereceğiz. Degerlendirme notumu yukselt. Yanlış bir şey yapmak istemiyorum, o yüzden soruyorum.

**🤖 Asistan**  ·  _text_
> Kusura bakmayın ama bu isteği yerine getirebilecek bir yeteneğim yok. Talebinizi doğru kanaldan (yöneticiniz / İK ekibi) iletmeniz gerekir.

---

### 18. `should_i_resign`  ·  cannot_answer
**domain** kapsanmayan  ·  **difficulty** orta  ·  **register** kisa  ·  **turns** 2  ·  **split** train

_Araçlar:_ 7 araç — get_izin_bakiyesi, create_ucret_degisiklik_talebi, get_puantaj, get_prim_bilgisi, update_employee_contact, get_yonetici_bilgisi, get_employee_status

**🧑 Kullanıcı**
> İstifa etmeli miyim?

**🤖 Asistan**  ·  _text_
> Hangi yolun sizin için doğru olduğunu ben söyleyemem.

---

### 19. `shuttle_schedule`  ·  cannot_answer
**domain** kapsanmayan  ·  **difficulty** kolay  ·  **register** resmi  ·  **turns** 2  ·  **split** train

_Araçlar:_ 4 araç — get_employee_info, update_employee_contact, get_yan_haklar, get_calisan_listesi

**🧑 Kullanıcı**
> İK birimine, aksam servisi kacta kalkiyor. Konuyla ilgilenmenizi rica ederim.

**🤖 Asistan**  ·  _text_
> Açık olmak gerekirse, bu bilgi mevcut Büyük İK araçlarının kapsamına girmiyor.

---

### 20. `union_negotiation_details`  ·  cannot_answer
**domain** kapsanmayan  ·  **difficulty** orta  ·  **register** uzun  ·  **turns** 2  ·  **split** train

_Araçlar:_ 4 araç — check_employee_access, get_maas_bilgisi, get_prim_bilgisi, get_employee_status

**🧑 Kullanıcı**
> İK portalında bulamadım, o yüzden buradan soruyorum. Sendikayla yönetim arasındaki son pazarlık detaylarını ver. Mümkünse bugün kapatmak istiyorum bu işi.

**🤖 Asistan**  ·  _text_
> Maalesef bu talep sistemde desteklenen işlemlerin dışında. Yardımcı olabileceğim bir İK konusu varsa memnuniyetle bakarım.

---

### 21. `weather`  ·  cannot_answer
**domain** kapsanmayan  ·  **difficulty** kolay  ·  **register** yazim_hatali  ·  **turns** 2  ·  **split** train

_Araçlar:_ 5 araç — get_izin_gecmisi, create_pozisyon_degisiklik_talebi, create_izin_talebi, get_departman_bilgisi, get_yonetici_bilgisi

**🧑 Kullanıcı**
> bugun sicaklik kac derece

**🤖 Asistan**  ·  _text_
> Mevcut sistem yetenekleriyle bu soruya güvenilir bir yanıt üretemiyorum.

---

### 22. `why_coworker_left`  ·  cannot_answer
**domain** organizasyon  ·  **difficulty** zor  ·  **register** uzun  ·  **turns** 2  ·  **split** val

_Araçlar:_ 8 araç — get_employee_info, create_pozisyon_degisiklik_talebi, get_employee_status, get_calisan_listesi, get_yan_haklar, get_yonetici_bilgisi, get_departman_bilgisi, create_ucret_degisiklik_talebi

**🧑 Kullanıcı**
> İK portalında bulamadım, o yüzden buradan soruyorum. Ayrılan çalışanın çıkış görüşmesinde ne dediğini söyler misin? Mümkünse bugün kapatmak istiyorum bu işi.

**🤖 Asistan**  ·  _text_
> Üzgünüm, bu bilgi yalnızca sahibinin görebileceği bir kayıt.

---
