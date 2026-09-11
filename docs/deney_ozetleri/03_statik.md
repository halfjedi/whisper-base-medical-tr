# Deney 3: `acil_tip_statik`

## Veri
- train_statik.jsonl (1010 satir = 505 temiz + 505 diske gomulu bozuk kopya), on-the-fly KAPALI
- train_dosyasi: `egitim/statik_veri/train_statik.jsonl`
- 1010 train / 96 val satırı

## Augmentation
- gürültü_p=0.0, reverb_p=0.0, kazanç_p=0.0, birleştir_p=0.0

## LoRA config
- r=64, alpha=128, target_modules=['q_proj', 'v_proj'], dropout=0.05

## Eğitim ayarları
- learning_rate=0.001, batch_size=4, grad_accum=4 (efektif batch=16)
- epoch_tavanı=8, erken_durdurma_sabrı=3, warmup_steps=48

## Epoch bazlı sonuçlar (val seti)

| Epoch | WER | Loss | Train Loss | Fark (train-eval) |
|---|---|---|---|---|
| 1 | 30.66 | 0.2880 | 1.6105 | +1.3225 |
| 2 | 15.93 | 0.2261 | 0.9128 | +0.6867 |
| 3 | 15.81 | 0.2156 | 0.4070 | +0.1914 |
| 4 | 15.57 | 0.2212 | 0.1293 | -0.0919 |
| 5 | 14.25 | 0.2195 | 0.0587 | -0.1608 |
| 6 | 13.77 | 0.2158 | 0.0408 | -0.1750 |
| 7 | 13.53 | 0.2171 | 0.0354 | -0.1818 **(best)** |
| 8 | 13.65 | 0.2177 | 0.0314 | -0.1863 |

**best_metric (val WER)**: 13.53 — `egitim/checkpoints/acil_tip_statik/checkpoint-448`

## Test seti sonuçları (en iyi val WER'li 3 epoch, nihai/tarafsız ölçüm)

| Epoch | Val WER | Test WER | Test Loss | Checkpoint |
|---|---|---|---|---|
| 6 | 13.77 | **%18.69** | 0.2998 | checkpoint-384 |
| 7 | 13.53 | **%19.72** | 0.3023 | checkpoint-448 |
| 8 | 13.65 | **%19.03** | 0.3021 | checkpoint-512 |

**Ortalama test WER (bu 3 epoch üzerinden): %19.15**
**En iyi tek epoch: %18.69**

## Çıkarım (inference) gecikmesi — MPS

`checkpoint-448` ile, 32 test cümlesi üzerinde `.generate()` süresi ölçüldü:

| Ölçüt | Değer |
|---|---|
| İlk çağrı (ısınma) | 631 ms (tek seferlik) |
| Ortalama (ısınma hariç) | 366.6 ms |
| Medyan | 342.9 ms |
| Min / Max | 276.1 ms / 567.9 ms |
