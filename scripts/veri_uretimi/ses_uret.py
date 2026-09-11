"""
hasta_cumleleri.jsonl'daki metinleri Chatterbox Multilingual TTS ile seslendirip
Whisper fine-tune için (ses, metin) eğitim manifestosu üretir.

Ses çeşitliliği: her cümle için TÜİK'in gerçek 20+ yaş dağılımına ve %50/%50
cinsiyet oranına göre ağırlıklı rastgele seçilmiş bir Common Voice referans
sesi kullanılır (bkz. ses_profilleri.py, ses_profil_sec.py). Chatterbox bu
referansı klonlayarak konuşuyor - yani üretilen ses tek tip değil, gerçek
nüfus dağılımını yansıtan farklı yaş/cinsiyetten "hasta sesleri" oluyor.

Kullanılan üretim parametreleri (Mac'te canlı test edilip onaylandı):
    repetition_penalty=1.2, temperature=0.8, cfg_weight=0.5
    (varsayılan repetition_penalty=2.0 kısa cümlelerde token tekrarı döngüsüne
    girip sonda "drone" artefaktı üretiyordu, bkz. proje notları)
Her üretimden sonra Silero VAD ile baş/son sessizlik ve tutarsız pay kırpılır
(ses_kirp.py), sonra Whisper'ın beklediği 16kHz mono'ya resample edilir.

Devam ettirilebilirlik: her satır için ayrı bir .wav dosyası üretilir
(dosya adı satırın içerik hash'ine göre - aynı script tekrar çalıştırılırsa
zaten üretilmiş dosyalar atlanır). Manifest de append modunda yazılır ve
başta var olan manifest okunup zaten işlenmiş dosyalar atlanır.

Kullanım (Mac'te test, RTX'te tam üretim - ikisinde de aynı script):
    source venv/bin/activate
    python veri_uretimi/ses_uret.py --dal acil_tip
    python veri_uretimi/ses_uret.py --dal acil_tip --limit 20   # küçük test
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch
import torchaudio as ta

sys.path.insert(0, str(Path(__file__).parent.parent))  # ses_kirp.py proje kökünde
from ses_kirp import sesi_kirp
from ses_profilleri import profil_sec
import terimler

METIN_DOSYASI = Path(__file__).parent / "cikti" / "hasta_cumleleri.jsonl"
SES_DIZINI = Path(__file__).parent / "cikti" / "sesler"
MANIFEST_DOSYASI = Path(__file__).parent / "cikti" / "whisper_manifest.jsonl"

HEDEF_SR = 16000  # Whisper'ın beklediği örnekleme hızı

# Kelime başına en az bu kadar saniye beklenir (çok gevşek bir alt sınır -
# doğal konuşma çok daha yavaş, ama bozuk/neredeyse-sessiz üretimi eleyecek
# kadar sıkı). Altında kalan üretim "muhtemelen bozuk" sayılıp yeniden denenir.
MIN_SANIYE_KELIME_BASI = 0.12
MAX_SES_DENEME = 3

_MODEL = None
_WHISPER = None

# Üretilen sesi Whisper'la geri okutup hedef metinle kabaca örtüşüyor mu diye
# kontrol eder. Süre eşiği TEK BAŞINA yetersizdi: bir cümle "yeterince uzun"
# görünse bile içeriği yarıda kesilmiş olabiliyordu (gözlemlendi: 8 kelimelik
# bir cümle 2.26sn'de "yeterince uzun" sayıldı ama sesin sadece ilk 3 kelimesi
# vardı - VAD/süre kontrolü bunu yakalayamadı). Whisper transkriptinin kelime
# sayısı, hedef metnin kelime sayısına kıyasla bu oranın altındaysa "muhtemelen
# eksik/kesik" sayılır.
MIN_TRANSKRIPT_ORANI = 0.7


def resmi_isme_cevir(cumle: str, hedef_terim: str) -> str:
    """TTS'e verilen metin (cumle) fonetik yazımı (örn. "Zanaks") kullanır,
    çünkü Türkçe TTS "Xanax" gibi orijinal yazımları hatalı okuyor. Ama
    Whisper'ın eğitim etiketi (whisper_manifest.jsonl'daki "text") resmi
    ilaç ismini içermeli. Bu fonksiyon sadece etiket metninde fonetik ismi
    resmi isimle değiştirir; TTS'e giden ses hâlâ fonetik metinden üretilir."""
    resmi_isim = terimler.DRUGS_ORIGINAL_ADLAR.get(hedef_terim)
    if resmi_isim is None or resmi_isim == hedef_terim:
        return cumle
    desen = re.compile(r"\b" + re.escape(hedef_terim) + r"('\w*)?\b")
    return desen.sub(lambda m: resmi_isim + (m.group(1) or ""), cumle)


def _cihaz():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _model():
    global _MODEL
    if _MODEL is None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        cihaz = _cihaz()
        print(f"Cihaz: {cihaz}")
        print("Chatterbox modeli yükleniyor...")
        t0 = time.monotonic()
        _MODEL = ChatterboxMultilingualTTS.from_pretrained(device=cihaz)
        print(f"Yüklendi: {time.monotonic() - t0:.1f}s")
    return _MODEL


def _whisper():
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel
        # base DEĞİL small: base'in Türkçe'de zayıf olduğunu (temel kelimelerde
        # bile hatalı) kendi testlerimizde görmüştük - doğrulama/hakem rolünde
        # güvenilmez bir ölçüt olurdu (hem iyi sesi yanlışlıkla eler hem kötüyü
        # geçirebilirdi). Burada hız değil doğruluk önemli, latency kısıtı yok.
        print("Doğrulama için Whisper (small) yükleniyor...")
        _WHISPER = WhisperModel("small", device="cpu", compute_type="int8")
    return _WHISPER


def icerik_tam_mi(wav_tensor: torch.Tensor, sr: int, hedef_metin: str) -> bool:
    """Üretilen sesi Whisper'la geri okutup, hedef metnin kabaca tamamının
    söylenip söylenmediğini kontrol eder (kesik/yarım üretimleri yakalamak için)."""
    import numpy as np
    ses = wav_tensor[0].numpy().astype(np.float32)
    segments, _ = _whisper().transcribe(ses, language="tr", beam_size=1, temperature=0.0)
    transkript = " ".join(s.text for s in segments)
    transkript_kelime = len(transkript.split())
    hedef_kelime = len(hedef_metin.split())
    if hedef_kelime == 0:
        return True
    return (transkript_kelime / hedef_kelime) >= MIN_TRANSKRIPT_ORANI


def dosya_adi_uret(row: dict) -> str:
    anahtar = f"{row['dal']}|{row['socrates_asama']}|{row['hedef_terim']}|{row['tekrar_no']}|{row.get('kaynak', 'lmstudio')}"
    kisa_hash = hashlib.sha1(anahtar.encode("utf-8")).hexdigest()[:12]
    return f"{row['dal']}_{kisa_hash}.wav"


def load_islenmis_dosyalar() -> set[str]:
    islenmis = set()
    if not MANIFEST_DOSYASI.exists():
        return islenmis
    with open(MANIFEST_DOSYASI, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            islenmis.add(row.get("audio_path"))
    return islenmis


def main():
    parser = argparse.ArgumentParser(description="Hasta cümlelerini seslendirip Whisper manifestosu üretir")
    parser.add_argument("--dal", type=str, default=None, help="sadece belirli bir dalı seslendir")
    parser.add_argument("--limit", type=int, default=None, help="işlenecek satır sayısını sınırla (test için)")
    args = parser.parse_args()

    if not METIN_DOSYASI.exists():
        raise SystemExit(f"Bulunamadı: {METIN_DOSYASI}")

    rows = [json.loads(l) for l in open(METIN_DOSYASI, encoding="utf-8") if l.strip()]
    if args.dal:
        rows = [r for r in rows if r.get("dal") == args.dal]

    SES_DIZINI.mkdir(parents=True, exist_ok=True)
    islenmis = load_islenmis_dosyalar()

    islenecekler = []
    for row in rows:
        dosya_adi = dosya_adi_uret(row)
        yol = str(SES_DIZINI / dosya_adi)
        if yol in islenmis:
            continue
        islenecekler.append((row, yol))

    if args.limit is not None:
        islenecekler = islenecekler[:args.limit]

    print(f"Toplam metin satırı: {len(rows)}")
    print(f"Zaten seslendirilmiş: {len(islenmis)}")
    print(f"Bu çalıştırmada işlenecek: {len(islenecekler)}")

    if not islenecekler:
        print("İşlenecek yeni satır yok.")
        return

    model = _model()
    uretilen = 0
    hatali = 0
    t_baslangic = time.monotonic()

    with open(MANIFEST_DOSYASI, "a", encoding="utf-8") as manifest_f:
        for i, (row, yol) in enumerate(islenecekler, 1):
            cumle = row["hasta_cumlesi"]
            kelime_sayisi = len(cumle.split())
            min_sure = kelime_sayisi * MIN_SANIYE_KELIME_BASI

            profil_anahtari, ses_referans_yolu = profil_sec()

            try:
                wav16 = None
                for deneme in range(1, MAX_SES_DENEME + 1):
                    wav = model.generate(
                        cumle, language_id="tr", audio_prompt_path=ses_referans_yolu,
                        repetition_penalty=1.2, temperature=0.8, cfg_weight=0.5,
                    )
                    wav = sesi_kirp(wav, model.sr)
                    aday_wav16 = ta.functional.resample(wav, model.sr, HEDEF_SR)
                    aday_sure = aday_wav16.shape[-1] / HEDEF_SR
                    if aday_sure < min_sure:
                        print(f"  [deneme {deneme}/{MAX_SES_DENEME}] süre çok kısa ({aday_sure:.2f}s < {min_sure:.2f}s), "
                              f"muhtemelen bozuk üretim, yeniden deneniyor: \"{cumle[:40]}...\"")
                        continue
                    if not icerik_tam_mi(aday_wav16, HEDEF_SR, cumle):
                        print(f"  [deneme {deneme}/{MAX_SES_DENEME}] içerik eksik/kesik görünüyor (Whisper "
                              f"doğrulaması geçemedi), yeniden deneniyor: \"{cumle[:40]}...\"")
                        continue
                    wav16 = aday_wav16
                    break

                if wav16 is None:
                    print(f"  [hata] {MAX_SES_DENEME} denemede de geçerli ses üretilemedi, atlanıyor: \"{cumle[:50]}...\"")
                    hatali += 1
                    continue

                # 16-bit PCM (standart) olarak kaydet - torchaudio varsayılanı 32-bit
                # float PCM veriyor, çoğu eğitim/veri yükleme aracı 16-bit integer bekler
                ta.save(yol, wav16, HEDEF_SR, encoding="PCM_S", bits_per_sample=16)

                sure = wav16.shape[-1] / HEDEF_SR
                etiket_metni = (
                    resmi_isme_cevir(cumle, row["hedef_terim"])
                    if row["dal"] == "ilac" else cumle
                )
                manifest_satiri = {
                    "audio_path": yol,
                    "text": etiket_metni,
                    "sample_rate": HEDEF_SR,
                    "duration_s": round(sure, 3),
                    "dal": row["dal"],
                    "socrates_asama": row["socrates_asama"],
                    "hedef_terim": row["hedef_terim"],
                    "persona": row.get("persona"),
                    "kaynak": row.get("kaynak", "lmstudio"),
                    "ses_profili": profil_anahtari,
                }
                manifest_f.write(json.dumps(manifest_satiri, ensure_ascii=False) + "\n")
                manifest_f.flush()
                uretilen += 1
            except Exception as e:
                print(f"  [hata] \"{cumle[:50]}...\": {e}")
                hatali += 1

            if i % 20 == 0 or i == len(islenecekler):
                gecen = time.monotonic() - t_baslangic
                hiz = gecen / i
                kalan = (len(islenecekler) - i) * hiz
                print(f"  [{i}/{len(islenecekler)}] {hiz:.1f}s/cümle, tahmini kalan süre: {kalan/60:.1f} dakika")

    print(f"\nTamamlandı. Üretilen: {uretilen}, hatalı: {hatali}")
    print(f"Ses dosyaları: {SES_DIZINI}")
    print(f"Manifest: {MANIFEST_DOSYASI}")


if __name__ == "__main__":
    main()
