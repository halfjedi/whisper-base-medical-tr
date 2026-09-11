# Deney 1: `acil_tip_baseline`

## Veri
- acil_tip_train.jsonl (505 satir), on-the-fly augmentation
- train_dosyasi: `veri_uretimi/cikti/acil_tip_train.jsonl`
- 505 train / 96 val satırı

## Augmentation
- gürültü_p=0.45, reverb_p=0.25, kazanç_p=0.5, birleştir_p=0.0

## LoRA config
- r=64, alpha=128, target_modules=['q_proj', 'v_proj'], dropout=0.05

## Eğitim ayarları
- learning_rate=0.001, batch_size=8, grad_accum=2 (efektif batch=16)
- epoch_tavanı=15, erken_durdurma_sabrı=None, warmup_steps=48

## Epoch bazlı sonuçlar (val seti)

| Epoch | WER | Loss | Train Loss | Fark (train-eval) |
|---|---|---|---|---|
| 1 | 25.63 | 0.3255 | 0.9419 | +0.6164 |
| 2 | 20.84 | 0.2642 | 0.7322 | +0.4680 |
| 3 | 22.63 | 0.2512 | 0.4911 | +0.2398 |
| 4 | 19.28 | 0.2390 | 0.4509 | +0.2120 |
| 5 | 15.45 | 0.2345 | 0.3607 | +0.1261 |
| 6 | 15.45 | 0.2191 | 0.2138 | -0.0053 |
| 7 | 29.10 | 0.2262 | 0.1624 | -0.0638 |
| 8 | 17.01 | 0.2226 | 0.1371 | -0.0855 |
| 9 | 15.57 | 0.2282 | 0.1023 | -0.1259 |
| 10 | 14.73 | 0.2116 | 0.0902 | -0.1214 |
| 11 | 14.85 | 0.2123 | 0.1001 | -0.1122 |
| 12 | 12.69 | 0.2123 | 0.0575 | -0.1548 **(best)** |
| 13 | 14.97 | 0.2159 | 0.0592 | -0.1567 |
| 14 | 13.17 | 0.2146 | 0.0515 | -0.1631 |
| 15 | 12.93 | 0.2136 | 0.0442 | -0.1694 |

**best_metric (val WER)**: 12.69 — `egitim/checkpoints/acil_tip_baseline/checkpoint-384`

## Test seti sonuçları (en iyi val WER'li 3 epoch, nihai/tarafsız ölçüm)

| Epoch | Val WER | Test WER | Test Loss | Checkpoint |
|---|---|---|---|---|
| 12 | 12.69 | **%16.96** | 0.2712 | checkpoint-384 |
| 14 | 13.17 | **%16.96** | 0.2715 | checkpoint-448 |
| 15 | 12.93 | **%16.96** | 0.2720 | checkpoint-480 |

**Ortalama test WER (bu 3 epoch üzerinden): %16.96**
**En iyi tek epoch: %16.96**

## Çıkarım (inference) gecikmesi — MPS

`checkpoint-384` ile, 32 test cümlesi üzerinde `.generate()` süresi ölçüldü:

| Ölçüt | Değer |
|---|---|
| İlk çağrı (ısınma) | 3239 ms (tek seferlik) |
| Ortalama (ısınma hariç) | 361.9 ms |
| Medyan | 344.6 ms |
| Min / Max | 245.5 ms / 551.4 ms |
