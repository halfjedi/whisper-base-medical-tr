"""
ses_profil_sec.py'nin ürettiği ses_referans/profil_havuzu.json'dan, TÜİK'in
gerçek 20+ yaş dağılımına ve %50/%50 cinsiyet oranına göre AĞIRLIKLI RASTGELE
bir referans ses seçer. ses_uret.py bunu her cümle için çağırıp Chatterbox'a
audio_prompt_path olarak verir.

Kaynak: TÜİK Adrese Dayalı Nüfus Kayıt Sistemi Sonuçları, 2025
(https://data.tuik.gov.tr/Bulten/Index?p=Adrese-Dayali-Nufus-Kayit-Sistemi-Sonuclari-2025-53899)
"""

import json
import random
from pathlib import Path

HAVUZ_DOSYASI = Path(__file__).parent / "ses_referans" / "profil_havuzu.json"

# TÜİK 15+ yaş dağılımı (teens eklenince taban nüfus 20+'tan 15+'a genişledi,
# oranlar yeniden hesaplandı - kaynak: TÜİK ADNKS 2025/2026, bkz. ses_profil_sec.py)
TUIK_YAS_DAGILIMI = {
    "teens": 0.093,
    "twenties": 0.187,
    "thirties": 0.183,
    "fourties": 0.184,
    "fifties": 0.145,
    "sixties": 0.115,
    "seventies": 0.067,
    "eighties": 0.026,
}
CINSIYETLER = ["female_feminine", "male_masculine"]

_HAVUZ = None
_AGIRLIKLI_ANAHTARLAR = None
_AGIRLIKLAR = None
_SON_KULLANILAN_INDEKS: dict[str, int] = {}


def _yukle():
    global _HAVUZ, _AGIRLIKLI_ANAHTARLAR, _AGIRLIKLAR
    if _HAVUZ is not None:
        return
    if not HAVUZ_DOSYASI.exists():
        raise FileNotFoundError(
            f"{HAVUZ_DOSYASI} bulunamadı. Önce 'python veri_uretimi/ses_profil_sec.py' çalıştırılmalı."
        )
    with open(HAVUZ_DOSYASI, encoding="utf-8") as f:
        _HAVUZ = json.load(f)

    anahtarlar = []
    agirliklar = []
    for age, oran in TUIK_YAS_DAGILIMI.items():
        for gender in CINSIYETLER:
            anahtar = f"{age}_{gender}"
            if _HAVUZ.get(anahtar):
                anahtarlar.append(anahtar)
                agirliklar.append(oran * 0.5)  # yas orani x cinsiyet orani (%50)
    if not anahtarlar:
        raise RuntimeError("profil_havuzu.json boş görünüyor.")
    _AGIRLIKLI_ANAHTARLAR = anahtarlar
    _AGIRLIKLAR = agirliklar


def profil_sec() -> tuple[str, str]:
    """TÜİK dağılımına göre ağırlıklı rastgele bir (bucket_anahtari, dosya_yolu) döner.
    Aynı bucket içindeki klipler sırayla (round-robin) döndürülür - böylece art
    arda aynı ses tekrarlanmaz, havuzdaki tüm konuşmacılar eşit kullanılır."""
    _yukle()
    anahtar = random.choices(_AGIRLIKLI_ANAHTARLAR, weights=_AGIRLIKLAR, k=1)[0]
    klipler = _HAVUZ[anahtar]
    indeks = _SON_KULLANILAN_INDEKS.get(anahtar, -1) + 1
    indeks %= len(klipler)
    _SON_KULLANILAN_INDEKS[anahtar] = indeks
    # profil_havuzu.json uretildigi makinenin MUTLAK yolunu iceriyor (orn.
    # /Users/ozanpatlar/...) - baska bir makinede (Colab) bu yol yok. Sadece
    # dosya adini alip, bu modulun yaninda duran ses_referans/ dizinine gore
    # yeniden kuruyoruz - hem Mac'te hem Colab'da dogru sonuc verir.
    dosya_adi = Path(klipler[indeks]).name
    return anahtar, str(HAVUZ_DOSYASI.parent / dosya_adi)


def dagilim_ozeti() -> str:
    _yukle()
    satirlar = []
    for anahtar, agirlik in zip(_AGIRLIKLI_ANAHTARLAR, _AGIRLIKLAR):
        satirlar.append(f"  {anahtar:28s} hedef oran: %{agirlik*100:.1f}  (havuzda {len(_HAVUZ[anahtar])} klip)")
    return "\n".join(satirlar)


if __name__ == "__main__":
    print("Ses profili dağılımı (TÜİK 20+ yaş, %50/%50 cinsiyet):")
    print(dagilim_ozeti())

    print("\n20 örnek seçim (dağılımı doğrulamak için):")
    from collections import Counter
    sayac = Counter()
    for _ in range(2000):
        anahtar, _ = profil_sec()
        sayac[anahtar] += 1
    for anahtar, adet in sorted(sayac.items(), key=lambda x: -x[1]):
        print(f"  {anahtar:28s} {adet/2000*100:5.1f}%")
