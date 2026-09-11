"""
9 (ya da o an hazir olan) epoch checkpoint'ini FLEURS (genel Turkce, gercek
insan sesi) test setinde olcer - LoRA fine-tuning'in genel Turkce kapasitesini
ne kadar bozdugunu epoch bazinda gormek icin.

Kullanim:
    source venv/bin/activate
    python egitim/fleurs_epoch_testi.py <tumepoch_checkpoint_dizini>

Ornek:
    python egitim/fleurs_epoch_testi.py /path/to/checkpoint_dizini
"""

import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate

# google/fleurs'un eski (script tabanli) yukleme yolu artik datasets>=3'te
# calismiyor / donuyor (denendi, dogrulandi) - HF'nin otomatik parquet
# donusumunu dogrudan URL ile cekmek calisiyor
FLEURS_TR_PARQUET = "hf://datasets/google/fleurs@refs%2Fconvert%2Fparquet/tr_tr/test/0000.parquet"

ADIM_BASINA_EPOCH = 245
N_ORNEK = 80

wer_metrigi = evaluate.load("wer")


def normalize_et(m: str) -> str:
    m = m.lower()
    m = re.sub(r"[.,!?;:\"'()]", "", m)
    return re.sub(r"\s+", " ", m).strip()


def cihaz_sec():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def referans_metin(ornek):
    return ornek.get("raw_transcription") or ornek.get("transcription")


def modeli_hazirla(cihaz, checkpoint_yolu=None):
    base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    base_model.generation_config.language = "turkish"
    base_model.generation_config.task = "transcribe"
    if checkpoint_yolu is None:
        m = base_model
    else:
        peft_model = PeftModel.from_pretrained(base_model, checkpoint_yolu)
        m = peft_model.merge_and_unload()
    return m.to(cihaz).eval()


def degerlendir(model, cihaz, processor, ornekler, etiket):
    pred_strs, label_strs = [], []
    for ornek in ornekler:
        dalga = ornek["audio"]["array"]
        sr = ornek["audio"]["sampling_rate"]
        ozellikler = processor.feature_extractor(dalga, sampling_rate=sr).input_features
        girdi = torch.tensor(ozellikler).to(cihaz)
        with torch.no_grad():
            tahmin_ids = model.generate(girdi, language="turkish", task="transcribe", max_length=128)
        pred_strs.append(processor.tokenizer.decode(tahmin_ids[0], skip_special_tokens=True))
        label_strs.append(referans_metin(ornek))

    ham_wer = 100 * wer_metrigi.compute(predictions=pred_strs, references=label_strs)
    norm_wer = 100 * wer_metrigi.compute(
        predictions=[normalize_et(p) for p in pred_strs],
        references=[normalize_et(l) for l in label_strs],
    )
    print(f"[{etiket}] ham_wer={ham_wer:.2f} normalize_wer={norm_wer:.2f}")
    return ham_wer, norm_wer


def main():
    global ADIM_BASINA_EPOCH
    if len(sys.argv) < 2:
        raise SystemExit("Kullanim: python egitim/fleurs_epoch_testi.py <tumepoch_checkpoint_dizini> [adim_basina_epoch]")
    tumepoch_dizin = Path(sys.argv[1])
    if len(sys.argv) > 2:
        ADIM_BASINA_EPOCH = int(sys.argv[2])

    cihaz = cihaz_sec()
    print(f"Cihaz: {cihaz}")

    print("FLEURS (tr_tr) test seti cekiliyor (parquet mirror)...")
    fleurs = load_dataset("parquet", data_files={"test": FLEURS_TR_PARQUET})["test"]
    ornekler = [fleurs[i] for i in range(min(N_ORNEK, len(fleurs)))]
    print(f"{len(ornekler)} gercek konusma ornegi hazir (toplam {len(fleurs)} mevcuttu).")

    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="turkish", task="transcribe")

    mevcut_checkpointler = sorted(
        tumepoch_dizin.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    print(f"Bulunan checkpoint sayisi: {len(mevcut_checkpointler)}")

    MODELLER = [(None, "whisper-base (LoRA'siz, referans)")]
    for ckpt_dir in mevcut_checkpointler:
        adim = int(ckpt_dir.name.split("-")[1])
        epoch = round(adim / ADIM_BASINA_EPOCH)
        MODELLER.append((str(ckpt_dir), f"epoch{epoch}"))

    sonuclar = []
    for ckpt, etiket in MODELLER:
        try:
            m = modeli_hazirla(cihaz, ckpt)
        except Exception as e:
            print(f"[{etiket}] atlandi: {e}")
            continue
        ham, norm = degerlendir(m, cihaz, processor, ornekler, etiket)
        sonuclar.append((etiket, ham, norm))
        del m

    print("\n=== OZET: Epoch basina genel Turkce (FLEURS) WER ===")
    for etiket, ham, norm in sonuclar:
        print(f"{etiket:<15} ham={ham:.2f}%  normalize={norm:.2f}%")


if __name__ == "__main__":
    main()
