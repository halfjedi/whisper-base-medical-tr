# Deney Özet Tablosu — acil_tip LoRA Fine-tuning

Referans nokta: **LoRA'sız whisper-base** (fine-tune edilmemiş) → test_wer = **%38.41**, test_loss = 1.8997

| Model adı | Veri | Dropout | Epoch (gerçek/tavan) | En iyi val WER | Test WER (top-3 epoch ort.) | Test WER (en iyi tek) |
|---|---|---|---|---|---|---|
| `acil_tip_baseline` | 505 train / 96 val | 0.05 | 15/15 | %12.69 (epoch 12) | **%16.96** (3 ölçüm) | %16.96 |
| `acil_tip_dropout01` | 505 train / 96 val | 0.1 | 8/15 | %15.57 (epoch 5) | **%17.76** (3 ölçüm) | %16.26 |
| `acil_tip_statik` | 1010 train / 96 val | 0.05 | 8/8 | %13.53 (epoch 7) | **%19.15** (3 ölçüm) | %18.69 |

## En iyi model (test setindeki top-3 epoch ortalamasına göre): `acil_tip_baseline` (%16.96)
