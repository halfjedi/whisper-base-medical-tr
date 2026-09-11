"""
Bir deney klasorundeki TUM checkpoint-* klasorlerini ilac_test.jsonl (233 satir,
held-out) uzerinde olcer - ham + normalize WER ayri ayri.

Kullanim:
    source venv/bin/activate
    python egitim/ilac_epoch_testi.py <checkpoint_dizini> [adim_basina_epoch]

Ornek:
    python egitim/ilac_epoch_testi.py /Users/ozanpatlar/Downloads/drive-download-20260909T101502Z-1-001 557
"""

import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate

KOK = Path(__file__).parent.parent
ILAC_TEST = KOK / "veri_uretimi/cikti/ilac_test.jsonl"
ADIM_BASINA_EPOCH = 557

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


def veriyi_yukle():
    satirlar = []
    with open(ILAC_TEST, encoding="utf-8") as f:
        for satir in f:
            satirlar.append(json.loads(satir))
    return satirlar


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


def degerlendir(model, cihaz, processor, satirlar, etiket):
    import soundfile as sf

    pred_strs, label_strs = [], []
    for satir in satirlar:
        dalga, sr = sf.read(satir["audio_path"], dtype="float32")
        ozellikler = processor.feature_extractor(dalga, sampling_rate=sr).input_features
        girdi = torch.tensor(ozellikler).to(cihaz)
        with torch.no_grad():
            tahmin_ids = model.generate(girdi, language="turkish", task="transcribe", max_length=128)
        pred_strs.append(processor.tokenizer.decode(tahmin_ids[0], skip_special_tokens=True))
        label_strs.append(satir["text"])

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
        raise SystemExit("Kullanim: python egitim/ilac_epoch_testi.py <checkpoint_dizini> [adim_basina_epoch]")
    dizin = Path(sys.argv[1])
    if len(sys.argv) > 2:
        ADIM_BASINA_EPOCH = int(sys.argv[2])

    cihaz = cihaz_sec()
    print(f"Cihaz: {cihaz}")

    satirlar = veriyi_yukle()
    print(f"ilac_test.jsonl: {len(satirlar)} satir (ILAC - held-out test seti)")

    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="turkish", task="transcribe")

    mevcut_checkpointler = sorted(
        dizin.glob("checkpoint-*"),
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
        ham, norm = degerlendir(m, cihaz, processor, satirlar, etiket)
        sonuclar.append((etiket, ham, norm))
        del m

    print("\n=== OZET: Epoch basina ILAC (held-out test) WER ===")
    for etiket, ham, norm in sonuclar:
        print(f"{etiket:<15} ham={ham:.2f}%  normalize={norm:.2f}%")


if __name__ == "__main__":
    main()
