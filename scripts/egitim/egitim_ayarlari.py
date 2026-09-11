"""
Seq2SeqTrainingArguments - acil_tip baseline (augmentation'sız) deneyi.
Değerler ve gerekçeleri konuşmamızda tek tek listelenmişti, burada koda dökülüyor.
"""

from transformers import Seq2SeqTrainingArguments

egitim_ayarlari = Seq2SeqTrainingArguments(
    output_dir="egitim/checkpoints/acil_tip_baseline",

    learning_rate=1e-3,                # LoRA'da az parametre egitildigi icin normal
                                        # fine-tuning'den (~1e-5) cok daha yuksek LR gerekir
    per_device_train_batch_size=8,     # Mac/MPS bellek siniri icin temkinli baslangic
    gradient_accumulation_steps=2,     # Efektif batch=16
    num_train_epochs=15,               # 505 satirlik kucuk veri seti - val WER izlenip
                                        # gerekirse erken durdurulacak (load_best_model_at_end)
    warmup_steps=48,                   # Toplam ~480 optimizer adiminin %10'u (505 satir,
                                        # batch=8, grad_accum=2, 15 epoch ile hesaplandi -
                                        # warmup_ratio 5.2'de deprecated, sabit sayi kullaniyoruz)

    fp16=False,                        # MPS'te karisik hassasiyet destegi olgun degil -
    bf16=False,                        # ilk calistirmada guvenlik onceligi (fp32)
    gradient_checkpointing=False,      # Model kucuk (74M), bellek baskisi dusuk

    eval_strategy="epoch",
    save_strategy="epoch",             # eval ile eslesmeli (load_best_model_at_end icin sart)
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,           # WER'de dusuk = iyi

    predict_with_generate=True,        # Kritik: eval sirasinda gercek .generate() ile
                                        # WER hesaplansin, sadece teacher-forcing loss degil
    generation_max_length=128,

    remove_unused_columns=False,       # Kritik: aksi halde Trainer collator'in ihtiyac
                                        # duydugu dal/hedef_terim/ses_profili sutunlarini siler

    seed=42,                           # Split'te kullanilan seed'le tutarli
    report_to="tensorboard",
    logging_steps=10,
)


if __name__ == "__main__":
    print("Seq2SeqTrainingArguments basariyla olusturuldu.")
    print(f"  output_dir: {egitim_ayarlari.output_dir}")
    print(f"  efektif batch (train_batch_size * grad_accum): "
          f"{egitim_ayarlari.per_device_train_batch_size * egitim_ayarlari.gradient_accumulation_steps}")
    print(f"  fp16={egitim_ayarlari.fp16}, bf16={egitim_ayarlari.bf16}")
    print(f"  remove_unused_columns={egitim_ayarlari.remove_unused_columns}")
