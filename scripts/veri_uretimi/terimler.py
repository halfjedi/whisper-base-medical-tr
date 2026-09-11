"""
Sentetik hasta cümlesi üretimi için hedef terim listeleri.

Bu liste tam/kapsayıcı değildir — Whisper fine-tune veri setinin ÇEKİRDEĞİ
olarak tasarlanmıştır. Zamanla genişletilebilir (özellikle ilaç isimleri
kategorisi, kullanıcı tarafından ayrıca vurgulanan ÖNCELİKLİ kategoridir).

Her terim, üretim script'inde bir hasta cümlesi içine doğal şekilde
yerleştirilecek "hedef kelime/ifade"dir. Whisper bu terimleri doğru
tanımayı öğrenmeli (örn. "Parol" -> yanlışlıkla "Paroldu" değil).
"""

# --------------------------------------------------------------------------
# SOCRATES aşamaları (hastanın anlatım tarzını yönlendirmek için)
# --------------------------------------------------------------------------

SOCRATES_STAGES = {
    "site": "Şikayetin vücutta NEREDE olduğunu anlatan bir cümle (yer/bölge belirterek)",
    "onset": "Şikayetin NE ZAMAN ve NASIL başladığını anlatan bir cümle (ani mi, yavaş yavaş mı)",
    "character": "Şikayetin NASIL BİR HİS olduğunu anlatan bir cümle (batıcı, zonklayıcı, yanıcı, künt, vb.)",
    "radiation": "Şikayetin başka bir yere YAYILIP yayılmadığını anlatan bir cümle",
    "associations": "Şikayetle BİRLİKTE görülen başka belirtileri anlatan bir cümle (bulantı, ateş, terleme vb.)",
    "time_course": "Şikayetin ZAMAN İÇİNDE nasıl değiştiğini anlatan bir cümle (sürekli mi, aralıklı mı, kötüleşiyor mu)",
    "exacerbating_relieving": "Şikayeti ARTTIRAN ya da AZALTAN şeyleri anlatan bir cümle (hareket, yemek, ilaç, istirahat vb.)",
    "severity": "Şikayetin ŞİDDETİNİ anlatan bir cümle (günlük hayatı ne kadar etkilediği, 1-10 arası şiddet gibi)",
}

# --------------------------------------------------------------------------
# 18 aktif poliklinik dalı -> o dala özgü semptom/muayene terimleri
# --------------------------------------------------------------------------

BRANCHES = {
    "acil_tip": [
        "göğüs ağrısı", "nefes darlığı", "bilinç kaybı", "yüksek ateş",
        "şiddetli kanama", "travma", "kırık şüphesi", "zehirlenme",
        "alerjik reaksiyon", "boğulma hissi", "çarpıntı", "bayılma",
        "şiddetli karın ağrısı", "kafa travması", "yanık", "kesik yarası",
    ],
    "dahiliye": [
        "halsizlik", "kilo kaybı", "iştahsızlık", "gece terlemesi",
        "sürekli yorgunluk", "baş dönmesi", "şeker yüksekliği",
        "tansiyon yüksekliği", "ödem", "genel vücut ağrısı",
        "uykusuzluk", "kabızlık", "ishal",
    ],
    "kardiyoloji": [
        "göğüs ağrısı", "çarpıntı", "nefes darlığı", "bacaklarda şişlik",
        "efor sonrası yorulma", "kalp ritim bozukluğu", "göğüste baskı hissi",
        "tansiyon düşmesi", "senkop", "kalp çarpıntısı",
    ],
    "gogus_hastaliklari": [
        "öksürük", "balgam", "nefes darlığı", "hırıltılı solunum",
        "göğüs ağrısı", "kan tükürme", "gece öksürüğü", "hışıltı",
        "sigara öksürüğü", "uykuda solunum durması",
    ],
    "gastroenteroloji": [
        "karın ağrısı", "mide bulantısı", "kusma", "reflü", "şişkinlik",
        "hazımsızlık", "kanlı dışkı", "sarılık", "iştah kaybı",
        "yutma güçlüğü", "mide yanması", "karında şişlik",
    ],
    "noroloji": [
        "baş ağrısı", "baş dönmesi", "uyuşma", "karıncalanma",
        "güçsüzlük", "denge kaybı", "titreme", "nöbet geçirme",
        "bellek kaybı", "konuşma bozukluğu", "görme bulanıklığı",
        "migren", "yüz felci",
    ],
    "ortopedi": [
        "diz ağrısı", "bel ağrısı", "boyun ağrısı", "eklem ağrısı",
        "kırık şüphesi", "burkulma", "sırt ağrısı", "omuz ağrısı",
        "hareket kısıtlılığı", "topallama", "kas ağrısı", "kireçlenme",
    ],
    "uroloji": [
        "idrar yaparken yanma", "sık idrara çıkma", "kanlı idrar",
        "böbrek ağrısı", "idrar tutamama", "böğür ağrısı",
        "idrar yapmakta zorluk", "prostat şikayeti",
    ],
    "kadin_dogum": [
        "adet düzensizliği", "karın ağrısı", "gebelik takibi",
        "vajinal akıntı", "adet kanamasında artış", "gebelik bulantısı",
        "kasık ağrısı", "regl ağrısı", "menopoz şikayetleri",
    ],
    "kbb": [
        "boğaz ağrısı", "kulak ağrısı", "işitme kaybı", "burun tıkanıklığı",
        "baş dönmesi", "ses kısıklığı", "kulak çınlaması",
        "koku alamama", "sinüzit şikayeti", "burun kanaması",
    ],
    "goz": [
        "göz kızarıklığı", "görme bulanıklığı", "göz ağrısı",
        "göz kaşıntısı", "çift görme", "göz sulanması",
        "ışığa hassasiyet", "göz kuruluğu", "gözde çapaklanma",
    ],
    "dermatoloji": [
        "cilt döküntüsü", "kaşıntı", "egzama", "sivilce",
        "ben değişikliği", "cilt kuruluğu", "kızarıklık",
        "alerjik döküntü", "saç dökülmesi", "tırnak mantarı",
    ],
    "pediatri": [
        "ateş", "öksürük", "kusma", "ishal", "iştahsızlık",
        "huzursuzluk", "döküntü", "gelişim geriliği", "büyüme takibi",
        "aşı takibi", "karın ağrısı",
    ],
    "enfeksiyon": [
        "yüksek ateş", "titreme", "gece terlemesi", "boğaz ağrısı",
        "lenf bezi şişmesi", "genel halsizlik", "kas ağrısı",
        "yara yeri enfeksiyonu",
    ],
    "romatoloji": [
        "eklem ağrısı", "eklem şişliği", "sabah tutukluğu",
        "kas ağrısı", "yorgunluk", "cilt döküntüsü", "eklem kızarıklığı",
    ],
    "endokrinoloji": [
        "kilo alma", "kilo kaybı", "aşırı susama", "sık idrara çıkma",
        "çarpıntı", "tiroid şişliği", "el titremesi", "terleme artışı",
        "saç dökülmesi",
    ],
    "psikiyatri": [
        "uykusuzluk", "kaygı", "yoğun stres", "isteksizlik",
        "konsantrasyon güçlüğü", "panik atak", "huzursuzluk",
        "içe kapanma", "sürekli üzüntü hissi",
    ],
    "dis": [
        "diş ağrısı", "diş eti kanaması", "diş hassasiyeti",
        "çene ağrısı", "diş çürüğü", "diş eti şişliği",
    ],
}

# --------------------------------------------------------------------------
# Ortak kelime dağarcığı — tüm dallarda geçebilecek genel ifadeler
# --------------------------------------------------------------------------

# Zaman ifadelerinde SADECE TEK bir sayı olursa (örn. hep "iki gündür"), LoRA
# o kalıbı dar bir bağlama bağlayıp başka sayılarla (örn. "beş gündür")
# karşılaştığında hafif şaşırabilir (context-binding overfitting riski).
# Bu yüzden sayı çeşitliliği bilinçli olarak genişletildi - "gündür/haftadır/
# aydır/saattir" hepsi birden fazla farklı sayıyla temsil ediliyor.
ZAMAN_IFADELERI = [
    "bir gündür", "iki gündür", "üç gündür", "dört gündür", "beş gündür",
    "yedi gündür", "on gündür", "on beş gündür",
    "yarım saattir", "bir saattir", "üç saattir", "altı saattir",
    "bir haftadır", "iki haftadır", "üç haftadır",
    "bir aydır", "iki aydır", "üç aydır", "altı aydır",
    "bir yıldır", "iki yıldır",
]

# Şiddet skalasında da aynı risk - "10 üzerinden 8" tek örnekse LoRA o sayıyı
# ezberleyebilir. Farklı şiddet seviyeleri (hafif->şiddetli aralığı) eklendi.
SIDDET_IFADELERI = [
    "10 üzerinden 2 şiddetinde", "10 üzerinden 3 şiddetinde",
    "10 üzerinden 5 şiddetinde", "10 üzerinden 6 şiddetinde",
    "10 üzerinden 7 şiddetinde", "10 üzerinden 8 şiddetinde",
    "10 üzerinden 9 şiddetinde",
]

COMMON_VOCAB = ZAMAN_IFADELERI + SIDDET_IFADELERI + [
    "ne zamandır", "dün akşamdan beri",
    "aniden başladı", "yavaş yavaş arttı", "geçmiyor", "gittikçe kötüleşiyor",
    "ara ara oluyor", "sürekli devam ediyor", "dayanılmaz bir ağrı",
    "hafif bir rahatsızlık", "gece uyandırıyor",
    "yemekten sonra artıyor", "hareket edince artıyor", "istirahatle geçiyor",
    "ilaç kullanınca hafifliyor", "ailede de var", "daha önce hiç olmamıştı",
    "acile ilk kez geliyorum", "geçen sene de olmuştu",
]

# --------------------------------------------------------------------------
# İLAÇ İSİMLERİ — öncelikli kategori (kullanıcı özellikle vurguladı)
# HALK AĞZINDAKİ FONETİK YAZIMLA (marka adının resmi yazımıyla DEĞİL):
# "Xanax" değil "Zanaks", "Augmentin" değil "Ogmentin" gibi. Amaç: TTS'in
# doğru okuyacağı VE Whisper'ın gerçekte duyacağı sesle birebir eşleşen bir
# etiket kullanmak (bkz. proje notu: TTS resmi yazımı bazen yanlış/İngilizce
# aksanla okuyordu, testte doğrulandı).
#
# ⚠️ GÜVENİLİRLİK NOTU: Bu dönüşümler Türkçe okuma kuralına (Türkçe "c" HER
# ZAMAN /c/ (as in "can") değil "coğrafya" gibi okunur, İngilizce/Latin
# kökenli "c" ise a/o/u önünde /k/, e/i önünde /s/ okunur - bu yüzden
# "Coraspin"->"Koraspin", "Cipro"->"Sipro" gibi) dayanarak elle çevrildi,
# gerçek ses kaydıyla doğrulanmadı. Bazıları (özellikle az bilinen/karmaşık
# olanlar) yanlış olabilir - TTS ile dinleyip gözden geçirmek gerekir.
# --------------------------------------------------------------------------

DRUGS = [
    # Ağrı kesici / ateş düşürücü
    "Parol", "Aferin", "Majezik", "Nurofen", "Arveles", "Koraspin",
    "Apranaks", "Vermidon", "Minoset", "Novalgin", "Kataflam", "Brufen",
    "Panadol", "Dolven", "Voltaren", "Doloreks", "Deksalgin", "Kontramal",
    "Muskoril", "Aspirin", "Ekopirin", "Junior Parol", "Kalpol",
    "Dolgit", "Flanaks", "Olin", "Perfalgan",

    # Antibiyotik
    "Ogmentin", "Sipro", "Zinnat", "Klimisin", "Tarcefoksim", "Sefaks",
    "Klasid", "Zitromaks", "Amoklavin", "Biofloks", "Doksi", "Rulid",
    "Duosid", "Seflor", "Velosef",

    # Mide / gastrointestinal
    "Neksiyum", "Reni", "Talsid", "Motilyum", "Metpamid", "Gaviskon",
    "Buskopan", "Dufalak", "Enterol", "Reflor", "Lansor", "Pantpas",
    "Debridat", "Nospa", "Ulkuran",

    # Kalp / tansiyon / kolesterol
    "Konkor", "Koversil", "Norvask", "Belok", "Diovan", "Aprovel",
    "Kozar", "Kumadin", "Kordaron", "Lipitor", "Krestor",

    # Şeker / diyabet
    "Glukofaj", "Diamikron", "Kanuvia", "Forksiga", "Lantus", "Novomiks",
    "Humalog", "İnsulatard", "Aktos", "İnsülin",

    # Tiroid
    "Ötiroks", "Levotiron",

    # Solunum / astım
    "Ventolin", "Pulmikort", "Seretayd", "Simbikort", "Singuler",

    # Alerji / soğuk algınlığı
    "Deloday", "Aeriyus", "Zirtek", "Klaritin", "Zizal", "Rinaz",
    "Otrivin", "Koldreks", "Teraflu", "Grippin", "Sinerjin",

    # Psikiyatrik
    "Zanaks", "Prozak", "Sipraleks", "Leksapro", "Zoloft", "Simbalta",
    "Rivotril", "Lustıral", "Serokel",

    # Vitamin / takviye
    "Vitamin D3", "Kalsiyum tablet", "Magnezyum", "Ferro Sanol",
    "Vitamin B12", "Multivit", "Beroka", "Redokson",

    # Cilt / topikal
    "Bepanten", "Fusidin", "Terramisin", "Battikon", "Fusikort",
    "Zoviraks",

    # Kadın doğum / doğum kontrol
    "Yasmin", "Mikroginon", "Dufaston",

    # Sık kullanılan ama marka belirsiz genel ifadeler (marka riski yok,
    # gerçek hastalar da sıklıkla böyle söylüyor)
    "ağrı kesici hapım", "ateş düşürücü şurup", "antibiyotik", "mide koruyucu ilaç",
    "göz damlası", "burun spreyi", "doğum kontrol hapı", "uyku hapı",
    "sakinleştirici", "kolesterol ilacı", "kas gevşetici", "öksürük şurubu",
    "alerji hapı", "ağrı bandı", "doktorumun yazdığı tansiyon ilacı",
    "kalp ilacım", "şeker ilacım", "reçetesiz aldığım bir hap",
]

# --------------------------------------------------------------------------
# İLAÇ İSİMLERİ — fonetik yazım -> orijinal/resmi marka adı eşlemesi.
# DRUGS listesindeki her marka için izlenebilirlik amacıyla tutuluyor (veri
# seti raporunda/dokümantasyonunda "bu fonetik isim hangi gerçek ilaca denk
# geliyor" sorusuna cevap vermek için). Genel ifadeler (marka olmayanlar,
# örn. "ağrı kesici hapım") burada YOK çünkü zaten marka adı değiller.
# --------------------------------------------------------------------------

DRUGS_ORIGINAL_ADLAR = {
    "Parol": "Parol", "Aferin": "Aferin", "Majezik": "Majezik", "Nurofen": "Nurofen",
    "Arveles": "Arveles", "Koraspin": "Coraspin", "Apranaks": "Apranax",
    "Vermidon": "Vermidon", "Minoset": "Minoset", "Novalgin": "Novalgin",
    "Kataflam": "Cataflam", "Brufen": "Brufen", "Panadol": "Panadol",
    "Dolven": "Dolven", "Voltaren": "Voltaren", "Doloreks": "Dolorex",
    "Deksalgin": "Deksalgin", "Kontramal": "Contramal", "Muskoril": "Muscoril",
    "Aspirin": "Aspirin", "Ekopirin": "Ecopirin", "Junior Parol": "Junior Parol",
    "Kalpol": "Calpol", "Dolgit": "Dolgit", "Flanaks": "Flanax", "Olin": "Aulin",
    "Perfalgan": "Perfalgan",

    "Ogmentin": "Augmentin", "Sipro": "Cipro", "Zinnat": "Zinnat",
    "Klimisin": "Klimicin", "Tarcefoksim": "Tarcefoksim", "Sefaks": "Sefaks",
    "Klasid": "Klacid", "Zitromaks": "Zitromax", "Amoklavin": "Amoklavin",
    "Biofloks": "Bioflox", "Doksi": "Doxi", "Rulid": "Rulid",
    "Duosid": "Duocid", "Seflor": "Ceflor", "Velosef": "Velosef",

    "Neksiyum": "Nexium", "Reni": "Rennie", "Talsid": "Talcid",
    "Motilyum": "Motilium", "Metpamid": "Metpamid", "Gaviskon": "Gaviscon",
    "Buskopan": "Buscopan", "Dufalak": "Duphalac", "Enterol": "Enterol",
    "Reflor": "Reflor", "Lansor": "Lansor", "Pantpas": "Pantpas",
    "Debridat": "Debridat", "Nospa": "Nospa", "Ulkuran": "Ulcuran",

    "Konkor": "Concor", "Koversil": "Coversyl", "Norvask": "Norvasc",
    "Belok": "Beloc", "Diovan": "Diovan", "Aprovel": "Aprovel",
    "Kozar": "Cozaar", "Kumadin": "Coumadin", "Kordaron": "Cordarone",
    "Lipitor": "Lipitor", "Krestor": "Crestor",

    "Glukofaj": "Glucophage", "Diamikron": "Diamicron", "Kanuvia": "Januvia",
    "Forksiga": "Forxiga", "Lantus": "Lantus", "Novomiks": "Novomix",
    "Humalog": "Humalog", "İnsulatard": "Insulatard", "Aktos": "Actos",
    "İnsülin": "İnsülin",

    "Ötiroks": "Euthyrox", "Levotiron": "Levotiron",

    "Ventolin": "Ventolin", "Pulmikort": "Pulmicort", "Seretayd": "Seretide",
    "Simbikort": "Symbicort", "Singuler": "Singulair",

    "Deloday": "Deloday", "Aeriyus": "Aerius", "Zirtek": "Zyrtec",
    "Klaritin": "Claritine", "Zizal": "Xyzal", "Rinaz": "Rinaz",
    "Otrivin": "Otrivin", "Koldreks": "Coldrex", "Teraflu": "Theraflu",
    "Grippin": "Grippin", "Sinerjin": "Sinerjin",

    "Zanaks": "Xanax", "Prozak": "Prozac", "Sipraleks": "Cipralex",
    "Leksapro": "Lexapro", "Zoloft": "Zoloft", "Simbalta": "Cymbalta",
    "Rivotril": "Rivotril", "Lustıral": "Lustral", "Serokel": "Seroquel",

    "Vitamin D3": "Vitamin D3", "Kalsiyum tablet": "Kalsiyum tablet",
    "Magnezyum": "Magnezyum", "Ferro Sanol": "Ferro Sanol",
    "Vitamin B12": "Vitamin B12", "Multivit": "Multivit",
    "Beroka": "Berocca", "Redokson": "Redoxon",

    "Bepanten": "Bepanthene", "Fusidin": "Fucidin", "Terramisin": "Terramycin",
    "Battikon": "Batticon", "Fusikort": "Fucicort", "Zoviraks": "Zovirax",

    "Yasmin": "Yasmin", "Mikroginon": "Microgynon", "Dufaston": "Duphaston",
}

# --------------------------------------------------------------------------
# Yardımcı: tüm terimleri (dal, terim) çiftleri halinde düz liste olarak ver
# --------------------------------------------------------------------------

def tum_dal_terimleri():
    for dal, terimler in BRANCHES.items():
        for terim in terimler:
            yield dal, terim


def dal_listesi():
    return list(BRANCHES.keys()) + ["ortak", "ilac"]


# --------------------------------------------------------------------------
# Hasta personaları — her "tekrar" farklı bir profile bağlanır, böylece
# aynı (dal, aşama, terim) için üretilen 5 cümle sadece sampling şansına
# bırakılmaz, gerçek bir farklılaştırma zorlanmış olur (overfitting riskini
# azaltır). Yaş/cinsiyet kategorileri, ileride Common Voice ses klonlama
# adımındaki demografik dağılımla (kadın/erkek %50-%50, gerçekçi yaş
# dağılımı) hizalanacak şekilde seçildi.
# --------------------------------------------------------------------------

PERSONAS = [
    {"yas": "24 yaşında", "cinsiyet": "kadın", "uslup": "endişeli, hızlı ve biraz dağınık konuşan"},
    {"yas": "42 yaşında", "cinsiyet": "erkek", "uslup": "sakin, net ve kısa cümlelerle anlatan"},
    {"yas": "67 yaşında", "cinsiyet": "kadın", "uslup": "ağır ağır, detaylı ve şikayetçi bir üslupla anlatan"},
    {"yas": "31 yaşında", "cinsiyet": "erkek", "uslup": "öz ve pratik, gereksiz detay vermeyen"},
    {"yas": "55 yaşında", "cinsiyet": "kadın", "uslup": "çok bilgi veren, geçmiş şikayetleriyle kıyaslayan"},
]


def persona_sec(tekrar_no: int) -> dict:
    return PERSONAS[tekrar_no % len(PERSONAS)]
