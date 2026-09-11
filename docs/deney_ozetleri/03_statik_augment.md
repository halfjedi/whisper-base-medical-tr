# Deney 3: `acil_tip_statik` — TAMAMLANDI

## Amaç
On-the-fly (anlık, her erişimde farklı) augmentation yerine, bozulmayı diske kalıcı olarak gömüp ("statik çoğaltma") aynı verinin nasıl öğrenildiğini karşılaştırmak.

## Veri
- Yeni, ayrı bir train seti: `egitim/statik_veri/train_statik.jsonl` — **1010 satır**
  - 505 satır "temiz" (orijinal ses dosyasına referans, yeni kopya yok)
  - 505 satır "bozulmus" (gürültü+reverb+kazanç uygulanıp `egitim/statik_veri/sesler/`'e **kalıcı yeni .wav** olarak yazıldı — `egitim/statik_augment_uret.py`, seed=42)
- Val/Test: deney 1-2 ile **aynı** (`acil_tip_val.jsonl` 96 satır, `acil_tip_test.jsonl` 32 satır) — karşılaştırılabilirlik için değiştirilmedi

## Augmentation
- **On-the-fly KAPALI** (gurultu_p=reverb_p=kazanc_p=birlestir_p=0) — bozulma zaten diskte sabit/gömülü, eğitim sırasında tekrar rastgele bozulma uygulanmıyor

## LoRA config
- r=64, lora_alpha=128, target_modules=[q_proj, v_proj], lora_dropout=0,05 (deney 1 ile aynı)

## Eğitim ayarları
- learning_rate=1e-3
- **per_device_train_batch_size=4, gradient_accumulation_steps=4** (efektif batch=16, deney 1-2'deki 8×2'den değişti — bellek güvenlik payı için, MPS'te art arda çok fazla deney çalıştırınca yaşanan swap sorunundan sonra)
- num_train_epochs=8 (tavan — 1010 satırda epoch başı 64 adım, 505'lik sette 15 epoch'un karşılığı ~7-8 epoch'a denk geliyor)
- EarlyStoppingCallback(patience=3), warmup_steps=48, seed=42
- Checkpoint: `egitim/checkpoints/acil_tip_statik/`

## Epoch bazlı sonuçlar (val seti) — ŞU ANA KADAR

| Epoch | WER | Loss |
|---|---|---|
| 1 | 30,66 (diğer denemelerden belirgin kötü başlangıç) | 0,2880 |
| 2 | 15,93 | 0,2261 |
| 3 | 15,81 | 0,2156 |
| 4 | 15,57 | 0,2212 |
| 5 | 14,25 | 0,2195 |
| 6 | 13,77 | 0,2158 |
| **7** | **13,53 (best_model_checkpoint)** | 0,2171 |
| 8 | 13,65 | 0,2177 |

Toplam süre: 57 dakika (3436 saniye). Early stopping tetiklenmedi (epoch 7'ye kadar hâlâ iyileşiyordu, epoch 8'de çok küçük bir gerileme oldu ama patience=3'ü doldurmadı).

## Test seti sonucu (nihai, tarafsız ölçüm)

En iyi checkpoint (epoch 7, checkpoint-448) test edildi:
- **test_wer = %19,72**, test_loss = 0,3023

**Bu, üç deney arasında test setindeki EN KÖTÜ sonuç** — val'de en iyi görünen (%13,53, baseline'ın en iyisinden bile düşük) checkpoint, test'te en kötü çıktı. Bu, statik/kalıcı çoğaltmanın on-the-fly augmentation'a göre generalizasyonu **iyileştirmediğini, muhtemelen hafifçe kötüleştirdiğini** gösteriyor — en başta tartıştığımız teorik endişeyle (statik kopyaların modelin sabit bir bozulma örüntüsüne aşırı uyum sağlamasına yol açabileceği) tutarlı bir sonuç.

## Overfitting analizi (epoch 1-3, egitim devam ederken olculdu)

| Epoch | son_train_loss | eval_loss | fark |
|---|---|---|---|
| 1 | 1,6110 | 0,2880 | +1,3230 |
| 2 | 0,9128 | 0,2261 | +0,6867 |
| 3 | 0,4070 | 0,2156 | +0,1914 |

Fark epoch 3'te hâlâ pozitif (train loss > eval loss) — overfitting sinyali henüz yok. İzlemeye devam ediliyor.

## Sonuç
Val WER'de diğer iki denemeden daha iyi görünmesine rağmen (%13,53 vs deney 1'in %12,69'u aslında biraz daha iyiydi, ama genel eğri daha düşüktü), **test setinde en kötü sonucu verdi (%19,72)**. Bu, val WER'e güvenmemenin ne kadar kritik olduğunu bir kez daha kanıtlıyor — sadece val'e bakılsaydı "statik augmentation en iyisi" sonucuna varılırdı, tam tersi doğru çıktı.

**Öneri**: on-the-fly augmentation (deney 1'in yaklaşımı) tercih edilmeli, statik/kalıcı çoğaltma yaklaşımından vazgeçilmeli.
