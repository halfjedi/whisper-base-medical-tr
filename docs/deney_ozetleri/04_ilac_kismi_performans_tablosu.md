# İlaç Modeli (ilac_kismi_87) — Tam Performans Tablosu

Veri: `ilac_train.jsonl` (3910 satır, RTX'te üretilen %87.1 kısmi veri), aynı kazanan
augmentation/LoRA ayarları (dropout=0.05, birleştir_p=0.0, r=64, alpha=128).
Test seti: `ilac_test.jsonl` (233 satır, eğitimde hiç görülmedi).

`checkpoint-3675` (Colab'ın en iyi epoch'u) için ayrıca 8 farklı makine/hassasiyet
kombinasyonunda gecikme + WER ölçümü yapıldı — bu yüzden o satır 8 kez tekrarlanıyor,
diğer checkpoint'ler sadece kendi orijinal (fp32, quantize'sız) eğitim-sonu ölçümüyle var.

"—" = henüz ölçülmedi (Colab GPU kotası doldu, sonraki oturumda tamamlanacak).
Model adı tabloda kısaltıldı (`ilac_kismi_87` / `ilac_kismi_87_colab` ayrımı "Eğ. Makine"
sütunundan anlaşılıyor — checkpoint klasörleri: `egitim/checkpoints/ilac_kismi_87/` (Mac)
ve `egitim/checkpoints/ilac_kismi_87_colab/` (Colab)).

| Model | Eğ. Makine | Epoch | Test Makine | Hassasiyet | Quantize | Gecikme(ms) | Val WER | Test WER Ham | Test WER Norm | Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| ilac_kismi_87 | Mac/MPS | 8 | Colab T4 | fp32 | Hayır | — | 9.80% | 9.64% | 7.45% | 0.1478 |
| ilac_kismi_87 | Mac/MPS | **9 (ham-iyi)** | Colab T4 | fp32 | Hayır | — | 9.50% | **8.09%** | 6.33% | 0.1363 |
| ilac_kismi_87 | Mac/MPS | **10 (norm-iyi)** | Colab T4 | fp32 | Hayır | — | 9.60% | 8.18% | **6.14%** | 0.1416 |
| ilac_kismi_87 | Colab/T4 | 13 | Colab T4 | fp32 | Hayır | — | 8.54% | 8.77% | 6.87% | 0.1393 |
| ilac_kismi_87 | Colab/T4 | 14 | Colab T4 | fp32 | Hayır | — | 8.25% | 8.72% | 6.72% | 0.1346 |
| ilac_kismi_87 | Colab/T4 | **15 (iyi)** | Colab T4 | fp32 | Hayır | 209.8 | 8.21% | 8.43% | 6.67% | 0.1327 |
| ilac_kismi_87 | Colab/T4 | 15 | **Colab T4** | **fp16** | Hayır | **202.8** | 8.21% | 8.33% | 6.58% | 0.1327 |
| ilac_kismi_87 | Colab/T4 | 15 | Colab T4 | fp32 | **int8(bnb)** | 1007.9 | 8.21% | 8.38% | 6.62% | 0.1327 |
| ilac_kismi_87 | Colab/T4 | 15 | Mac CPU | fp32 | Hayır | 341.6 | 8.21% | 8.43% | 6.67% | 0.1327 |
| ilac_kismi_87 | Colab/T4 | 15 | Mac CPU | fp16 | Hayır | 1711.7 | 8.21% | 8.43% | 6.67% | 0.1327 |
| ilac_kismi_87 | Colab/T4 | 15 | Mac CPU | fp32 | **int8(qnn)** | 833.8 | 8.21% | 9.01% | 7.21% | 0.1327 |
| ilac_kismi_87 | Colab/T4 | 15 | Mac MPS | fp32 | Hayır | 468.6 | 8.21% | 8.43% | 6.67% | 0.1327 |
| ilac_kismi_87 | Colab/T4 | 15 | Mac MPS | fp16 | Hayır | 406.0 | 8.21% | 8.43% | 6.67% | 0.1327 |

## Notlar

- **En iyi eğitim sonucu (test WER, ham)**: `ilac_kismi_87` epoch 9 (Mac) — %8.09.
- **En iyi eğitim sonucu (test WER, normalize)**: `ilac_kismi_87` epoch 10 (Mac) — %6.14, epoch 9'un normalize sonucundan (%6.33) bile düşük. Ham WER'e göre epoch 9 daha iyi görünüyordu ama normalizasyon sırasını değiştirdi — noktalama/büyük-küçük harf farklarının hangi epoch'ta daha çok "yanlış hata" saydığı değişebiliyor, bu yüzden iki metriği de raporlamak önemli.
- **En hızlı ölçülen konfigürasyon**: Colab T4 + fp16, 202.8ms (checkpoint-3675 üzerinde).
- **Quantization iki platformda da kaybetti**: Mac'te (qnnpack) 2.4x yavaş + WER kötüleşti (%8.43→%9.01), Colab'da (bitsandbytes) 4.8x yavaş (WER etkilenmedi ama hız felaketi).
- Gecikme (ms) sütunu sadece checkpoint-3675 için ölçüldü (8 satır) — diğer 5 checkpoint'in gecikmesi hâlâ ölçülmedi, sadece WER ölçüldü.
- Val WER, `trainer_state.json`'dan (eğitim sırasında ölçülen) alınmıştır; test setiyle karıştırılmamalı.
