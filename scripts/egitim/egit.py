"""
Tum LoRA egitim parcalarini (model+LoRA, processor, dataset+augmentation,
collator, egitim ayarlari, WER metrigi) birlestirip Seq2SeqTrainer'i kurar.

ONEMLI: Bu dosya calistirildiginda SADECE Trainer nesnesini kurar ve
dogrular (agirliklara hic dokunmaz, .train() COAGIRMAZ). Gercek egitimi
baslatmak icin ayri, acik bir onay gerekiyor - bkz. dosyanin sonundaki not.

Kullanim:
    source venv/bin/activate
    python egitim/egit.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import EarlyStoppingCallback, Seq2SeqTrainer, WhisperForConditionalGeneration, WhisperProcessor

from egitim_ayarlari import egitim_ayarlari
from metrik import compute_metrics_olustur
from veri_hazirlama import (
    DataCollatorSpeechSeq2SeqWithPadding,
    MPSBellekTemizleCallback,
    degerlendirme_donusumu_olustur,
    egitim_donusumu_olustur,
    esleme_havuzu_olustur,
)

KOK = Path(__file__).parent.parent
CHECKPOINT = "openai/whisper-base"


def trainer_olustur(ayarlar=None, train_ornek_sayisi=None, val_ornek_sayisi=None,
                     lora_dropout=0.05, erken_durdurma_sabri=None,
                     train_dosyasi=None, val_dosyasi=None, gurultu_p=0.45, reverb_p=0.25,
                     birlestir_p=0.0, kazanc_p=0.5, veri_aciklamasi=None,
                     lora_r=64, lora_alpha=128):
    if ayarlar is None:
        ayarlar = egitim_ayarlari
    if train_dosyasi is None:
        train_dosyasi = KOK / "veri_uretimi/cikti/acil_tip_train.jsonl"
    if val_dosyasi is None:
        val_dosyasi = KOK / "veri_uretimi/cikti/acil_tip_val.jsonl"

    # 1) Processor: ses -> log-mel ozellik, metin -> token ID (Turkce/transcribe sabit)
    processor = WhisperProcessor.from_pretrained(CHECKPOINT, language="turkish", task="transcribe")

    # 2) Veri seti: train (ozellestirilebilir kaynak) / val (hep ayni, karsilastirilabilirlik icin)
    veri = load_dataset("json", data_files={
        "train": str(train_dosyasi),
        "val": str(val_dosyasi),
    })

    # Kucuk olcekli smoke test icin alt kume secimi (varsayilan: hepsi)
    if train_ornek_sayisi is not None:
        veri["train"] = veri["train"].select(range(train_ornek_sayisi))
    if val_ornek_sayisi is not None:
        veri["val"] = veri["val"].select(range(val_ornek_sayisi))

    # 3) Birlestirme icin ayni (dal, hedef_terim, ses_profili) esleme havuzu (sadece train)
    ham_train = [json.loads(l) for l in open(train_dosyasi, encoding="utf-8")]
    havuz = esleme_havuzu_olustur(ham_train)

    # 4) set_transform: train'de augmentation (parametrelerle kontrol edilebilir), val'de hep temiz/sabit
    veri["train"].set_transform(egitim_donusumu_olustur(
        processor, havuz, gurultu_p=gurultu_p, reverb_p=reverb_p,
        birlestir_p=birlestir_p, kazanc_p=kazanc_p,
    ))
    veri["val"].set_transform(degerlendirme_donusumu_olustur(processor))

    # 5) Base model + LoRA (r/alpha parametrik - varsayilan r=64, alpha=128, q_proj+v_proj)
    model = WhisperForConditionalGeneration.from_pretrained(CHECKPOINT)
    # Whisper .generate() dil/gorev token'larini bu ayardan alir
    model.generation_config.language = "turkish"
    model.generation_config.task = "transcribe"

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
    )
    peft_model = get_peft_model(model, lora_config)

    # 6) Batch padding (gercek data collator)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 7) WER metrigi
    compute_metrics = compute_metrics_olustur(processor)

    # 8) Hepsini Seq2SeqTrainer'da birlestir
    callbacks = [MPSBellekTemizleCallback()]
    if erken_durdurma_sabri is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=erken_durdurma_sabri))

    # deney_ozet_uret.py'nin okuyacagi metadata - her egitimin kendi ayarlarini
    # otomatik kaydetmesi icin (ozet dosyalari elle yazilmasin diye)
    Path(ayarlar.output_dir).mkdir(parents=True, exist_ok=True)
    deney_bilgisi = {
        "model_adi": Path(ayarlar.output_dir).name,
        "veri_aciklamasi": veri_aciklamasi or f"{train_dosyasi} ({len(veri['train'])} satir)",
        "train_dosyasi": str(train_dosyasi),
        "train_satir_sayisi": len(veri["train"]),
        "val_satir_sayisi": len(veri["val"]),
        "augmentation": {
            "gurultu_p": gurultu_p, "reverb_p": reverb_p,
            "kazanc_p": kazanc_p, "birlestir_p": birlestir_p,
        },
        "lora": {"r": lora_r, "alpha": lora_alpha, "target_modules": ["q_proj", "v_proj"], "dropout": lora_dropout},
        "egitim_ayarlari": {
            "learning_rate": ayarlar.learning_rate,
            "batch_size": ayarlar.per_device_train_batch_size,
            "grad_accum": ayarlar.gradient_accumulation_steps,
            "epoch_tavani": ayarlar.num_train_epochs,
            "erken_durdurma_sabri": erken_durdurma_sabri,
            "warmup_steps": ayarlar.warmup_steps,
        },
    }
    with open(Path(ayarlar.output_dir) / "deney_bilgisi.json", "w", encoding="utf-8") as f:
        json.dump(deney_bilgisi, f, ensure_ascii=False, indent=2)

    trainer = Seq2SeqTrainer(
        model=peft_model,
        args=ayarlar,
        train_dataset=veri["train"],
        eval_dataset=veri["val"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=callbacks,
    )
    return trainer, peft_model, processor


if __name__ == "__main__":
    trainer, peft_model, processor = trainer_olustur()

    print("Trainer basariyla kuruldu (agirliklara hic dokunulmadi, .train() cagrilmadi).")
    peft_model.print_trainable_parameters()
    print(f"train_dataset boyutu: {len(trainer.train_dataset)}")
    print(f"eval_dataset boyutu: {len(trainer.eval_dataset)}")
    print(f"Cihaz: {trainer.args.device}")

    print("\n=== GERCEK EGITIM BASLIYOR (15 epoch, kullanici onayiyla) ===")
    trainer.train()
