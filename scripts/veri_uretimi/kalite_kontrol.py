"""
metin_uret.py'nin ürettiği ~11.000 cümleyi TEK TEK OKUMADAN doğrulamak için
üç katmanlı kalite kontrol.

Katman 1 — "sert filtreler" (--mod hard): Ücretsiz, deterministik, TÜM
    satırlarda çalışır. Kontrol eder:
      - hedef terim cümlede gerçekten geçiyor mu (geçmiyorsa cümle işe yaramaz)
      - persona/meta sızıntısı var mı ("24 yaşında kadın hasta:" gibi LLM'in
        promptu cümleye sızdırması, "İşte cümle:", tırnak kalıntısı, vb.)
      - cümle makul uzunlukta mı (çok kısa/çok uzun anomalisi)
      - İngilizce/başka dil sızıntısı var mı (kaba heuristic)
      - tam birebir tekrar (exact duplicate) var mı
    Sorunlu satırlar cikti/hard_filtre_basarisiz.jsonl'a yazılır, geri kalanı
    "temiz" kabul edilir.

Katman 2 — "LLM hakem" (--mod llm-hakem): Latency önemli değil (offline),
    bu yüzden LM Studio'daki modele HER satırı (ya da --sample ile bir alt
    kümeyi) ayrı ayrı gösterip "bu cümle doğal, bağlamı tutarlı, hedef terimi
    doğru kullanmış bir hasta cümlesi mi?" diye sorulur, JSON değerlendirme
    alınır. Kendi ürettiği cümleyi kendisi kontrol etse de, üretim ve
    değerlendirme farklı promptlar/görevler olduğu için tamamen anlamsız
    değildir; yine de KESİN doğrulama değil, geniş taramada anomali
    yakalamak için kullanılır.

Katman 3 — "insan örneklemi" (--mod ornek): Her (dal x persona) hücresinden
    birkaç örnek çekip okunabilir bir .md dosyasına yazar. 11.000 cümleyi
    değil, istatistiksel olarak temsil eden ~150-250 cümleyi hızlıca
    (30-40 dakikada) gözden geçirip genel hata oranını tahmin etmeye yeter
    (bkz. dosya sonundaki "nasıl kullanılır" notu).

Kullanım:
    source venv/bin/activate
    python veri_uretimi/kalite_kontrol.py --mod hard
    python veri_uretimi/kalite_kontrol.py --mod llm-hakem --sample 300
    python veri_uretimi/kalite_kontrol.py --mod ornek --sample 200
"""

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import logging

import requests
import zeyrek

logging.getLogger("zeyrek").setLevel(logging.ERROR)
logging.getLogger("zeyrek.rulebasedanalyzer").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))

IN_FILE = Path(__file__).parent / "cikti" / "hasta_cumleleri.jsonl"
OUT_DIR = Path(__file__).parent / "cikti"
HARD_FAIL_FILE = OUT_DIR / "hard_filtre_basarisiz.jsonl"
JUDGE_FILE = OUT_DIR / "llm_degerlendirme.jsonl"
SAMPLE_FILE = OUT_DIR / "insan_ornek.md"

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LLM_MODEL = "qwen2.5-7b-instruct"

# --------------------------------------------------------------------------
# Katman 1: sert filtreler
# --------------------------------------------------------------------------

META_SIZINTI_KALIPLARI = [
    r"\byaşında\b.*\b(kadın|erkek)\b",
    r"\bhasta profili\b", r"^poliklinik\s*:",
    r"^işte\b", r"^tabii\b", r"^elbette\b", r"^cevap\s*:",
    r"^cümle\s*:", r"^\d+\.\s",
    r"<[^>]+>",  # olası html/etiket sızıntısı
]
META_REGEX = re.compile("|".join(META_SIZINTI_KALIPLARI), re.IGNORECASE)

INGILIZCE_YAYGIN_KELIME = re.compile(
    r"\b(the|and|is|are|patient|doctor|pain|sorry|however|please)\b", re.IGNORECASE
)

# Qwen bazen ChatML stop token'ı doğru işlemeyince Çince'ye kayıyor (gözlemlendi).
YABANCI_ALFABE_REGEX = re.compile(
    r"[一-鿿぀-ヿ가-힯Ѐ-ӿ]"  # CJK, Japonca, Korece, Kiril
)

MIN_KELIME = 3
MAX_KELIME = 40

# zeyrek: gerçek bir Türkçe morfolojik analizör - LLM'in aksine "kadırım",
# "kaseşim", "attackleri" gibi UYDURMA/GEÇERSİZ kelimeleri objektif olarak
# yakalıyor (LLM hakem kendi ürettiği bu tür hataları göremiyordu, bkz. test).
# Tıbbi Latince kökenli kelimeleri (sinüzit, reflü, kireçlenme) doğru tanıyor;
# sadece ilaç MARKA isimleri (Parol, Augmentin) "bilinmiyor" çıkıyor - bu
# YANLIŞ POZİTİF değil, o kelimeler zaten hedef terim olduğu için hariç tutulur.
_ZEYREK_ANALYZER = None
_KELIME_REGEX = re.compile(r"[a-zA-ZçÇğĞıİöÖşŞüÜ]+")


def _zeyrek():
    global _ZEYREK_ANALYZER
    if _ZEYREK_ANALYZER is None:
        _ZEYREK_ANALYZER = zeyrek.MorphAnalyzer()
    return _ZEYREK_ANALYZER


def _kelime_lemmalari(kelime: str) -> set[str]:
    """zeyrek ile bir kelimenin olası köklerini (lemma) döner. Analiz
    başarısız olursa kelimenin kendisini döner (marka isimleri için güvenli)."""
    sonuc = _zeyrek().analyze(kelime.lower())
    lemmalar = {kelime.lower()}
    if sonuc and sonuc[0]:
        for parse in sonuc[0]:
            if parse.lemma and parse.lemma != "Unk":
                lemmalar.add(parse.lemma.lower())
    return lemmalar


def terim_gecerli_kullanilmis_mi(cumle: str, terim: str) -> bool:
    """Hedef terimin cümlede -çekimli haliyle bile olsa- geçip geçmediğini
    kontrol eder. Basit alt-metin (substring) araması Türkçe'nin eklemeli
    yapısında yanlış pozitif üretiyordu (örn. "bayılınca"/"bayılmışım",
    "bayılma" teriminin birebir alt metni değil ama AYNI KÖK - zeyrek'in
    lemma analiziyle bunu doğru eşleştiriyoruz)."""
    terim_kelimeler = [k.lower() for k in _KELIME_REGEX.findall(terim) if len(k) >= 2]
    if not terim_kelimeler:
        return terim.lower() in cumle.lower()

    cumle_kelimeler = [k.lower() for k in _KELIME_REGEX.findall(cumle)]
    cumle_lemma_havuzu: set[str] = set()
    for k in cumle_kelimeler:
        cumle_lemma_havuzu |= _kelime_lemmalari(k)

    for tk in terim_kelimeler:
        # SADECE tk in ck yönü kontrol edilir (terim kökü, çekimli/uzun bir
        # cümle kelimesinin İÇİNDE mi diye). Tersi (ck in tk) yanlış pozitif
        # üretiyordu: "da", "ki" gibi kısa yaygın kelimeler tesadüfen "darlığı"
        # gibi uzun terim kelimelerinin içinde geçiyor diye yanlışlıkla eşleşiyordu.
        dogrudan_eslesme = any(tk in ck for ck in cumle_kelimeler if len(ck) >= len(tk))
        lemma_eslesme = bool(_kelime_lemmalari(tk) & cumle_lemma_havuzu)
        if not (dogrudan_eslesme or lemma_eslesme):
            return False
    return True


def bilinmeyen_kelimeler(cumle: str, terim: str) -> list[str]:
    """Cümledeki, zeyrek'in Türkçe olarak çözemediği kelimeleri döner.
    Hedef terimin kendi kelimeleri (marka isimleri dahil) hariç tutulur."""
    haric = {k.lower() for k in _KELIME_REGEX.findall(terim)}
    bilinmeyenler = []
    for kelime in _KELIME_REGEX.findall(cumle):
        if len(kelime) < 3 or kelime.lower() in haric:
            continue
        sonuc = _zeyrek().analyze(kelime)
        parse = sonuc[0][0] if sonuc and sonuc[0] else None
        if parse is None or parse.pos == "Unk":
            bilinmeyenler.append(kelime)
    return bilinmeyenler


def hard_filtre_uygula(row: dict) -> list[str]:
    """Satırdaki sorunları liste olarak döner; boş liste = temiz."""
    sorunlar = []
    cumle = row.get("hasta_cumlesi", "")
    terim = row.get("hedef_terim", "")

    if not cumle.strip():
        sorunlar.append("bos_cumle")
        return sorunlar

    # Terimin BİREBİR/lemma eşleşmesi sadece "ilac" kategorisinde zorunlu:
    # marka isimleri patients tarafından zaten literal söyleniyor (Whisper'a
    # tam olarak bunu öğretmek istiyoruz). Diğer kategorilerde (18 dal + ortak)
    # bu kural doğallıkla çelişiyordu - gerçek hastalar "nefes darlığı" gibi
    # klinik terimi değil "nefesim daralıyor" gibi günlük tabir kullanıyor.
    # Onlarda artık sadece dal/konuyla ilgili doğal bir cümle olması yeterli.
    if row.get("dal") == "ilac" and not terim_gecerli_kullanilmis_mi(cumle, terim):
        sorunlar.append("hedef_terim_eksik")

    if META_REGEX.search(cumle):
        sorunlar.append("meta_sizinti")

    kelime_sayisi = len(cumle.split())
    if kelime_sayisi < MIN_KELIME:
        sorunlar.append("cok_kisa")
    elif kelime_sayisi > MAX_KELIME:
        sorunlar.append("cok_uzun")

    if INGILIZCE_YAYGIN_KELIME.search(cumle):
        sorunlar.append("olasi_ingilizce_sizinti")

    if YABANCI_ALFABE_REGEX.search(cumle):
        sorunlar.append("yabanci_alfabe_sizinti")

    if cumle.count('"') >= 2 or cumle.startswith('"'):
        sorunlar.append("tirnak_kalintisi")

    bilinmeyenler = bilinmeyen_kelimeler(cumle, terim)
    if bilinmeyenler:
        sorunlar.append("bilinmeyen_kelime")
        row["_bilinmeyen_kelimeler"] = bilinmeyenler

    return sorunlar


def mod_hard():
    if not IN_FILE.exists():
        raise SystemExit(f"Bulunamadı: {IN_FILE}. Önce metin_uret.py çalıştırılmalı.")

    toplam = 0
    temiz = 0
    sorun_sayaci = defaultdict(int)
    gorulmus_cumleler = set()
    tam_tekrar = 0
    basarisizlar = []

    with open(IN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            toplam += 1

            sorunlar = hard_filtre_uygula(row)

            norm = row.get("hasta_cumlesi", "").strip().lower()
            if norm in gorulmus_cumleler:
                sorunlar.append("tam_tekrar")
                tam_tekrar += 1
            else:
                gorulmus_cumleler.add(norm)

            if row.get("dusuk_cesitlilik"):
                sorunlar.append("dusuk_cesitlilik_flagli")

            if sorunlar:
                for s in sorunlar:
                    sorun_sayaci[s] += 1
                basarisizlar.append({**row, "sorunlar": sorunlar})
            else:
                temiz += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(HARD_FAIL_FILE, "w", encoding="utf-8") as f:
        for row in basarisizlar:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Toplam satır: {toplam}")
    print(f"Temiz (sorunsuz): {temiz} ({100 * temiz / toplam:.1f}%)")
    print(f"Sorunlu: {len(basarisizlar)} ({100 * len(basarisizlar) / toplam:.1f}%)")
    print("\nSorun dağılımı:")
    for sorun, adet in sorted(sorun_sayaci.items(), key=lambda x: -x[1]):
        print(f"  {sorun}: {adet}")
    print(f"\nSorunlu satırlar: {HARD_FAIL_FILE}")


# --------------------------------------------------------------------------
# Katman 2: LLM hakem
# --------------------------------------------------------------------------

HAKEM_SYSTEM_PROMPT = (
    "Sen çok titiz bir Türkçe dilbilgisi ve tıbbi veri kalite kontrolcüsüsün. "
    "Sana bir hasta cümlesi ve hedeflenen tıbbi terim verilecek. Cümleyi "
    "SIKI şekilde denetle:\n"
    "1. dogal: gerçek bir hastanın ağzından çıkmış gibi doğal mı?\n"
    "2. baglam_tutarli: cümle anlamlı, mantıklı, dağınık/kopuk değil mi?\n"
    "3. terim_dogru_kullanilmis: hedef terim doğru ve anlamlı kullanılmış mı?\n"
    "4. gramer_dogru: TÜM kelimeler gerçek, doğru çekimli Türkçe kelimeler mi? "
    "(uydurma kelime, yanlış çekim, İngilizce kelime sızıntısı varsa FALSE)\n"
    "5. uc_sahis_sizinti: cümlede \"24 yaşında bir kadınım/kadırım/erkeğim\" gibi "
    "yaş/cinsiyet/kimlik bilgisi açıkça geçiyor mu? (geçiyorsa TRUE - bu kötü)\n\n"
    "SADECE şu JSON formatında yanıt ver, başka hiçbir şey yazma:\n"
    '{"dogal": true/false, "baglam_tutarli": true/false, '
    '"terim_dogru_kullanilmis": true/false, "gramer_dogru": true/false, '
    '"uc_sahis_sizinti": true/false, "sorun": "kısa açıklama veya boş string"}'
)


def hakem_gecti_mi(degerlendirme: dict) -> bool:
    return (
        degerlendirme.get("dogal") is True
        and degerlendirme.get("baglam_tutarli") is True
        and degerlendirme.get("terim_dogru_kullanilmis") is True
        and degerlendirme.get("gramer_dogru") is True
        and degerlendirme.get("uc_sahis_sizinti") is False
    )


def llm_hakem_degerlendir(cumle: str, terim: str, dal: str) -> dict | None:
    user_prompt = (
        f"Poliklinik: {dal}\nHedef terim: \"{terim}\"\nHasta cümlesi: \"{cumle}\"\n\n"
        "Yukarıdaki JSON formatında değerlendir."
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": HAKEM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 150,
    }
    try:
        resp = requests.post(f"{LM_STUDIO_BASE_URL}/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        return json.loads(text)
    except Exception as e:
        print(f"  [uyarı] hakem hatası: {e}")
        return None


def mod_llm_hakem(sample_size: int):
    if not IN_FILE.exists():
        raise SystemExit(f"Bulunamadı: {IN_FILE}.")

    rows = [json.loads(l) for l in open(IN_FILE, encoding="utf-8") if l.strip()]
    random.seed(7)
    secilenler = random.sample(rows, min(sample_size, len(rows)))

    print(f"{len(secilenler)} satır LLM hakeme gönderiliyor (toplam {len(rows)} satırdan stratified olmayan rastgele örnek)...")

    sonuclar = []
    basari = defaultdict(int)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(JUDGE_FILE, "w", encoding="utf-8") as f:
        for i, row in enumerate(secilenler, 1):
            degerlendirme = llm_hakem_degerlendir(row["hasta_cumlesi"], row["hedef_terim"], row["dal"])
            if degerlendirme is None:
                continue
            sonuc = {**row, "hakem_degerlendirme": degerlendirme}
            f.write(json.dumps(sonuc, ensure_ascii=False) + "\n")
            sonuclar.append(sonuc)

            for k in ("dogal", "baglam_tutarli", "terim_dogru_kullanilmis"):
                if degerlendirme.get(k):
                    basari[k] += 1

            if i % 25 == 0:
                print(f"  [{i}/{len(secilenler)}]")

    n = len(sonuclar)
    if n == 0:
        print("Hiç değerlendirme alınamadı (LM Studio çalışıyor mu?).")
        return

    print(f"\n{n} satır değerlendirildi.")
    for k, v in basari.items():
        print(f"  {k}: {v}/{n} ({100*v/n:.1f}%)")
    sorunlu = [s for s in sonuclar if s["hakem_degerlendirme"].get("sorun")]
    print(f"\nSorun belirtilen satır sayısı: {len(sorunlu)}")
    for s in sorunlu[:10]:
        print(f"  - \"{s['hasta_cumlesi']}\" -> {s['hakem_degerlendirme']['sorun']}")
    print(f"\nTüm değerlendirmeler: {JUDGE_FILE}")


def mod_temizle():
    """Kapalı döngü kalite kontrolü, İKİ aşamalı:
    1) zeyrek (objektif Türkçe morfolojik analiz) - LLM çağrısı YOK, hızlı/ücretsiz.
       Uydurma/geçersiz kelime içeren satırlar direkt elenir (LLM hakem bu tür
       hataları kendi ürettiği için göremiyordu, bkz. proje notları).
    2) zeyrek'i geçen satırlar için LLM hakem - doğallık/bağlam/gramer kontrolü.
    Başarısız olanlar hasta_cumleleri.jsonl'dan SİLİNİR. Silinen
    (dal,asama,terim,tekrar_no) kombinasyonları metin_uret.py'nin resume
    mekanizması sayesinde bir sonraki çalıştırmada otomatik yeniden üretilir.
    Bu döngüyü (temizle -> metin_uret.py -> temizle -> ...) birkaç kez
    tekrarlayarak hata oranını aşamalı olarak düşürebilirsin."""
    if not IN_FILE.exists():
        raise SystemExit(f"Bulunamadı: {IN_FILE}.")

    rows = [json.loads(l) for l in open(IN_FILE, encoding="utf-8") if l.strip()]
    print(f"{len(rows)} satır işleniyor: önce zeyrek (yerel, hızlı), sonra LLM hakem (yerel LM Studio)...")

    gecenler = []
    kalanlar = []
    sorun_sayaci = defaultdict(int)
    zeyrek_elenen = 0

    for i, row in enumerate(rows, 1):
        bilinmeyenler = bilinmeyen_kelimeler(row["hasta_cumlesi"], row["hedef_terim"])
        if bilinmeyenler:
            kalanlar.append({**row, "hakem_degerlendirme": None, "bilinmeyen_kelimeler": bilinmeyenler})
            sorun_sayaci["zeyrek_bilinmeyen_kelime"] += 1
            zeyrek_elenen += 1
            if i % 20 == 0:
                print(f"  [{i}/{len(rows)}] gecen={len(gecenler)} silinen={len(kalanlar)}")
            continue

        degerlendirme = llm_hakem_degerlendir(row["hasta_cumlesi"], row["hedef_terim"], row["dal"])
        if degerlendirme is None:
            gecenler.append(row)  # hakem yanıt vermezse veriyi kaybetmemek için tut
            continue

        if hakem_gecti_mi(degerlendirme):
            gecenler.append(row)
        else:
            kalanlar.append({**row, "hakem_degerlendirme": degerlendirme})
            for k in ("dogal", "baglam_tutarli", "terim_dogru_kullanilmis", "gramer_dogru"):
                if degerlendirme.get(k) is False:
                    sorun_sayaci[k] += 1
            if degerlendirme.get("uc_sahis_sizinti") is True:
                sorun_sayaci["uc_sahis_sizinti"] += 1

        if i % 20 == 0:
            print(f"  [{i}/{len(rows)}] gecen={len(gecenler)} silinen={len(kalanlar)}")

    # Temiz satırları geri yaz - silinenler dosyada kalmaz, resume onları yeniden üretir
    with open(IN_FILE, "w", encoding="utf-8") as f:
        for row in gecenler:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(HARD_FAIL_FILE, "w", encoding="utf-8") as f:
        for row in kalanlar:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSonuç: {len(gecenler)}/{len(rows)} geçti ({100*len(gecenler)/len(rows):.1f}%), "
          f"{len(kalanlar)} satır silindi.")
    print("Sorun dağılımı (silinenler arasında):")
    for sorun, adet in sorted(sorun_sayaci.items(), key=lambda x: -x[1]):
        print(f"  {sorun}: {adet}")
    print(f"\n{IN_FILE} güncellendi (sadece geçenler kaldı).")
    print(f"Silinen satırlar (referans için): {HARD_FAIL_FILE}")
    if kalanlar:
        print(f"\nŞimdi 'python veri_uretimi/metin_uret.py --dal <dal>' çalıştırarak "
              f"silinen {len(kalanlar)} satırı yeniden ürettirebilirsin.")


# --------------------------------------------------------------------------
# Katman 3: insan örneklemi (stratified) - okunabilir .md çıktı
# --------------------------------------------------------------------------

def mod_ornek(sample_size: int):
    if not IN_FILE.exists():
        raise SystemExit(f"Bulunamadı: {IN_FILE}.")

    rows = [json.loads(l) for l in open(IN_FILE, encoding="utf-8") if l.strip()]

    # (dal, persona) hücrelerine göre grupla, her hücreden orantılı örnek çek
    hucreler = defaultdict(list)
    for row in rows:
        hucreler[(row.get("dal"), row.get("persona"))].append(row)

    random.seed(3)
    hedef_hucre_basi = max(1, sample_size // max(1, len(hucreler)))
    secilenler = []
    for key, grup in hucreler.items():
        secilenler.extend(random.sample(grup, min(hedef_hucre_basi, len(grup))))
    random.shuffle(secilenler)
    secilenler = secilenler[:sample_size]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
        f.write(f"# İnsan gözden geçirme örneklemi ({len(secilenler)} / {len(rows)} satır)\n\n")
        f.write(
            "Her satır için: cümle bağlamdan kopmuş mu, hedef terimi doğal "
            "kullanıyor mu, persona/dal ile tutarlı mı diye bak. Sorunlu "
            "olanları not al, oranı toplam örneklem sayısına bölüp genel "
            "hata oranını tahmin et.\n\n---\n\n"
        )
        for i, row in enumerate(secilenler, 1):
            f.write(
                f"**{i}.** [{row.get('dal')} / {row.get('socrates_asama')} / "
                f"{row.get('persona')}] hedef: *{row.get('hedef_terim')}*\n"
                f"> {row.get('hasta_cumlesi')}\n\n"
            )

    print(f"{len(secilenler)} satırlık stratified örneklem yazıldı: {SAMPLE_FILE}")
    print(f"({len(hucreler)} (dal, persona) hücresinden, hücre başı ~{hedef_hucre_basi} örnek)")
    print("Bu dosyayı okuyup sorunlu satırları işaretleyerek genel hata oranını tahmin edebilirsin.")


def main():
    global LLM_MODEL

    parser = argparse.ArgumentParser(description="Sentetik hasta cümleleri için 3 katmanlı kalite kontrolü")
    parser.add_argument("--mod", choices=["hard", "llm-hakem", "ornek", "temizle"], required=True)
    parser.add_argument("--sample", type=int, default=200, help="llm-hakem/ornek modları için örneklem büyüklüğü")
    parser.add_argument("--model", type=str, default=LLM_MODEL, help="llm-hakem için LM Studio model adı")
    args = parser.parse_args()

    LLM_MODEL = args.model

    if args.mod == "hard":
        mod_hard()
    elif args.mod == "llm-hakem":
        mod_llm_hakem(args.sample)
    elif args.mod == "ornek":
        mod_ornek(args.sample)
    elif args.mod == "temizle":
        mod_temizle()


if __name__ == "__main__":
    main()
