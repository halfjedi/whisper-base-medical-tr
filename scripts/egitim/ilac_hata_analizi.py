"""
ilac_test.jsonl uzerindeki hatalari IKI kategoriye ayirir:
  1) Ilac adinin kendisi yanlis mi tanindi (hedef_terim ile eslesen kelime)
  2) Cumlenin GERI KALANI ne kadar yanlis (ilac adi cikarilarak hesaplanan WER)

Bu ayrim onemli: eger kalan hatalar ilac adlarindaysa daha fazla GENEL Turkce
verisi eklemek ise yaramaz; eger cumlenin geri kalanindaysa genel Turkce verisi
dogrudan yardim eder.

Tahminler <checkpoint>/../ilac_tahminleri_<ckptadi>.json olarak onbellege alinir,
ayni checkpoint tekrar analiz edilirse GPU/MPS tekrar calistirilmaz.

Kullanim:
    source venv/bin/activate
    python egitim/ilac_hata_analizi.py <checkpoint_yolu> [etiket] [<checkpoint2> [etiket2] ...]
"""

import difflib
import json
import re
import sys
from pathlib import Path

import torch
import evaluate

KOK = Path(__file__).parent.parent
ILAC_TEST = KOK / "veri_uretimi/cikti/ilac_test.jsonl"
ONBELLEK_DIZIN = Path("/tmp/catpi_ilac_tahminleri")

# Ilac adi olarak kabul etmek icin hedef_terim'e minimum benzerlik
ADAY_ESIK = 0.40
# Tahmindeki kelimeyi "ayni ilac" saymak icin benzerlik (ek/sonek farki tolere edilir)
GOVDE_ESIK = 0.80

wer_metrigi = evaluate.load("wer")


def normalize_et(m: str) -> str:
    m = m.lower()
    m = re.sub(r"[.,!?;:\"'()]", "", m)
    return re.sub(r"\s+", " ", m).strip()


def benzerlik(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def cihaz_sec():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def veriyi_yukle():
    with open(ILAC_TEST, encoding="utf-8") as f:
        return [json.loads(s) for s in f]


def tahminleri_al(checkpoint_yolu, satirlar: list) -> list:
    """Tahminleri uretir ya da onbellekten okur.
    checkpoint_yolu None ise LoRA'siz referans whisper-base olculur."""
    ONBELLEK_DIZIN.mkdir(parents=True, exist_ok=True)
    if checkpoint_yolu is None:
        anahtar = "whisper_base__referans"
    else:
        ckpt = Path(checkpoint_yolu)
        # ust klasor adi + checkpoint adi -> ayni epoch farkli deneylerde cakismasin
        anahtar = f"{ckpt.parent.name}__{ckpt.name}"
    onbellek = ONBELLEK_DIZIN / f"{anahtar}.json"

    if onbellek.exists():
        with open(onbellek, encoding="utf-8") as f:
            tahminler = json.load(f)
        if len(tahminler) == len(satirlar):
            print(f"  (onbellekten okundu: {onbellek})")
            return tahminler

    import soundfile as sf
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    cihaz = cihaz_sec()
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="turkish", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    base_model.generation_config.language = "turkish"
    base_model.generation_config.task = "transcribe"
    if checkpoint_yolu is None:
        model = base_model.to(cihaz).eval()
    else:
        model = PeftModel.from_pretrained(base_model, checkpoint_yolu).merge_and_unload().to(cihaz).eval()

    tahminler = []
    for i, satir in enumerate(satirlar):
        dalga, sr = sf.read(satir["audio_path"], dtype="float32")
        ozellikler = processor.feature_extractor(dalga, sampling_rate=sr).input_features
        girdi = torch.tensor(ozellikler).to(cihaz)
        with torch.no_grad():
            tahmin_ids = model.generate(girdi, language="turkish", task="transcribe", max_length=128)
        tahminler.append(processor.tokenizer.decode(tahmin_ids[0], skip_special_tokens=True))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(satirlar)}")

    with open(onbellek, "w", encoding="utf-8") as f:
        json.dump(tahminler, f, ensure_ascii=False, indent=2)
    del model
    return tahminler


def ilac_tokenini_bul(ref_kelimeler: list, hedef_norm: str):
    """Referans cumlesinde hedef_terim'e en cok benzeyen kelimeyi dondurur."""
    en_iyi, en_iyi_skor = None, 0.0
    for k in ref_kelimeler:
        s = benzerlik(k, hedef_norm)
        if s > en_iyi_skor:
            en_iyi, en_iyi_skor = k, s
    if en_iyi_skor >= ADAY_ESIK:
        return en_iyi, en_iyi_skor
    return None, en_iyi_skor


def analiz_et(satirlar: list, tahminler: list, etiket: str) -> dict:
    toplam_ilac = 0
    govde_dogru = 0
    tam_dogru = 0
    kalan_ref, kalan_pred = [], []
    tum_ref, tum_pred = [], []
    hatali_ilaclar = []

    for satir, tahmin in zip(satirlar, tahminler):
        ref_n = normalize_et(satir["text"])
        pred_n = normalize_et(tahmin)
        tum_ref.append(ref_n)
        tum_pred.append(pred_n)

        ref_kelimeler = ref_n.split()
        pred_kelimeler = pred_n.split()
        hedef_norm = normalize_et(satir.get("hedef_terim", ""))

        ilac_token, _ = ilac_tokenini_bul(ref_kelimeler, hedef_norm) if hedef_norm else (None, 0)

        if ilac_token is not None:
            toplam_ilac += 1
            tam = ilac_token in pred_kelimeler
            govde = tam or any(benzerlik(pk, ilac_token) >= GOVDE_ESIK for pk in pred_kelimeler)
            if tam:
                tam_dogru += 1
            if govde:
                govde_dogru += 1
            else:
                # tahminde ilac adinin yerine ne gecmis - en yakin adayi goster
                en_yakin = max(pred_kelimeler, key=lambda pk: benzerlik(pk, ilac_token), default="")
                hatali_ilaclar.append((satir.get("hedef_terim", ""), ilac_token, en_yakin))

            # ilac adini cikarip kalan cumlenin WER'ini olc
            kalan_ref.append(" ".join(k for k in ref_kelimeler if k != ilac_token))
            kalan_pred.append(" ".join(k for k in pred_kelimeler if not benzerlik(k, ilac_token) >= GOVDE_ESIK))
        else:
            kalan_ref.append(ref_n)
            kalan_pred.append(pred_n)

    genel_wer = 100 * wer_metrigi.compute(predictions=tum_pred, references=tum_ref)
    kalan_wer = 100 * wer_metrigi.compute(predictions=kalan_pred, references=kalan_ref)

    print(f"\n=== {etiket} ===")
    print(f"Genel normalize WER (tum cumle)      : {genel_wer:.2f}%")
    print(f"Ilac adi CIKARILMIS normalize WER    : {kalan_wer:.2f}%   <- cumlenin geri kalani")
    print(f"Ilac adi bulunabilen ornek           : {toplam_ilac}/{len(satirlar)}")
    if toplam_ilac:
        print(f"Ilac adi govde dogrulugu (ek serbest): {100 * govde_dogru / toplam_ilac:.1f}%  ({govde_dogru}/{toplam_ilac})")
        print(f"Ilac adi TAM dogruluk (ek dahil)     : {100 * tam_dogru / toplam_ilac:.1f}%  ({tam_dogru}/{toplam_ilac})")
        print(f"Ilac adi tamamen kacirilan           : {toplam_ilac - govde_dogru}")

    if hatali_ilaclar:
        print(f"\n  Kacirilan ilac adlari (ilk 20) — soylenen | referans | tahmin:")
        for hedef, ref_tok, pred_tok in hatali_ilaclar[:20]:
            print(f"    {hedef:<18} | {ref_tok:<20} | {pred_tok}")

    return {
        "etiket": etiket,
        "genel_wer": genel_wer,
        "kalan_wer": kalan_wer,
        "ilac_govde_dogruluk": 100 * govde_dogru / toplam_ilac if toplam_ilac else 0,
        "ilac_tam_dogruluk": 100 * tam_dogru / toplam_ilac if toplam_ilac else 0,
        "kacirilan": toplam_ilac - govde_dogru,
        "toplam_ilac": toplam_ilac,
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Kullanim: python egitim/ilac_hata_analizi.py <checkpoint> [etiket] [<checkpoint2> [etiket2]]")

    # argumanlari (yol, etiket) ciftlerine ayir - etiket opsiyonel
    # "base" ozel degeri: LoRA'siz referans whisper-base
    args = sys.argv[1:]
    isler = []
    i = 0
    while i < len(args):
        yol = None if args[i] == "base" else args[i]
        varsayilan_etiket = "whisper-base (LoRA'siz)" if yol is None else Path(yol).name
        etiket = None
        if i + 1 < len(args) and args[i + 1] != "base" and not Path(args[i + 1]).exists():
            etiket = args[i + 1]
            i += 2
        else:
            i += 1
        isler.append((yol, etiket or varsayilan_etiket))

    satirlar = veriyi_yukle()
    print(f"ilac_test.jsonl: {len(satirlar)} satir (ILAC held-out test)")

    ozetler = []
    for yol, etiket in isler:
        print(f"\nTahminler hazirlaniyor: {etiket}")
        tahminler = tahminleri_al(yol, satirlar)
        ozetler.append(analiz_et(satirlar, tahminler, etiket))

    if len(ozetler) > 1:
        print("\n\n=== KARSILASTIRMA ===")
        print(f"{'model':<28} {'genel':>8} {'kalan':>8} {'ilac_govde':>11} {'ilac_tam':>9} {'kacan':>6}")
        for o in ozetler:
            print(f"{o['etiket']:<28} {o['genel_wer']:>7.2f}% {o['kalan_wer']:>7.2f}% "
                  f"{o['ilac_govde_dogruluk']:>10.1f}% {o['ilac_tam_dogruluk']:>8.1f}% {o['kacirilan']:>6}")


if __name__ == "__main__":
    main()
