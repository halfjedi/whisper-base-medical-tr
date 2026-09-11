# Deney 2: `acil_tip_dropout01`

## Veri
- acil_tip_train.jsonl (505 satir), on-the-fly augmentation (deney1 ile ayni)
- train_dosyasi: `veri_uretimi/cikti/acil_tip_train.jsonl`
- 505 train / 96 val satırı

## Augmentation
- gürültü_p=0.45, reverb_p=0.25, kazanç_p=0.5, birleştir_p=0.0

## LoRA config
- r=64, alpha=128, target_modules=['q_proj', 'v_proj'], dropout=0.1

## Eğitim ayarları
- learning_rate=0.001, batch_size=8, grad_accum=2 (efektif batch=16)
- epoch_tavanı=15, erken_durdurma_sabrı=3, warmup_steps=48

## Epoch bazlı sonuçlar (val seti)

| Epoch | WER | Loss | Train Loss | Fark (train-eval) |
|---|---|---|---|---|
| 1 | 25.63 | 0.3266 | 0.9511 | +0.6245 |
| 2 | 19.16 | 0.2635 | 0.7340 | +0.4705 |
| 3 | 19.64 | 0.2380 | 0.5329 | +0.2949 |
| 4 | 16.65 | 0.2250 | 0.3989 | +0.1739 |
| 5 | 15.57 | 0.2271 | 0.3293 | +0.1022 **(best)** |
| 6 | 17.96 | 0.2235 | 0.2178 | -0.0058 |
| 7 | 16.65 | 0.2231 | 0.1541 | -0.0690 |
| 8 | 15.57 | 0.2237 | 0.1276 | -0.0961 |

**best_metric (val WER)**: 15.57 — `egitim/checkpoints/acil_tip_dropout01/checkpoint-160`

## Test seti sonuçları (en iyi val WER'li 3 epoch, nihai/tarafsız ölçüm)

| Epoch | Val WER | Test WER | Test Loss | Checkpoint |
|---|---|---|---|---|
| 5 | 15.57 | **%18.69** | 0.2741 | checkpoint-160 |
| 7 | 16.65 | **%18.34** | 0.3021 | checkpoint-224 |
| 8 | 15.57 | **%16.26** | 0.2807 | checkpoint-256 |

**Ortalama test WER (bu 3 epoch üzerinden): %17.76**
**En iyi tek epoch: %16.26**

## Çıkarım (inference) gecikmesi — MPS

`checkpoint-160` ile, 32 test cümlesi üzerinde `.generate()` süresi ölçüldü:

| Ölçüt | Değer |
|---|---|
| İlk çağrı (ısınma) | 717 ms (tek seferlik) |
| Ortalama (ısınma hariç) | 367.1 ms |
| Medyan | 338.4 ms |
| Min / Max | 255.7 ms / 572.0 ms |
