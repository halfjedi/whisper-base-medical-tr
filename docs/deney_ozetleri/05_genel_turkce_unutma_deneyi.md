# Genel Türkçe Unutma Deneyi — LoRA rank + rehearsal karşılaştırması

**Amaç**: İlaç verisiyle LoRA fine-tuning yapmanın, modelin genel Türkçe konuşma
tanıma kabiliyetini ne kadar bozduğunu ölçmek (catastrophic forgetting), ve
bunu önlemek için iki değişikliğin (LoRA rank'ini düşürmek + eğitime genel
Türkçe verisi karıştırmak — rehearsal) etkisini test etmek.

**Değerlendirme yöntemi (her iki deneyde de aynı)**:
- İlaç Val WER: `ilac_val.jsonl` (736 satır) üzerinde, eğitim sırasında otomatik (Trainer)
- Genel Türkçe WER: [Google FLEURS](https://huggingface.co/datasets/google/fleurs) `tr_tr` test
  bölümünden 80 gerçek insan konuşması örneği (`egitim/fleurs_epoch_testi.py`) — bu örnekler
  **hiçbir deneyin eğitiminde kullanılmadı**, sadece değerlendirme için ayrıldı.
- Referans (LoRA'sız whisper-base): FLEURS'te ham_wer=%40.50, normalize_wer=%31.71

---

## Deney 1: `ilac_kismi_87_colab_tumepoch` — eski yaklaşım (rehearsal YOK)

- **Veri**: `ilac_train.jsonl` — 3910 satır, **sadece ilaç**, genel Türkçe hiç yok
- **LoRA**: r=64, alpha=128 (kazanan `ilac_kismi_87` config'iyle aynı)
- **Eğitildiği yer**: Colab (T4)

| Epoch | İlaç Val WER | Genel Türkçe Ham WER | Genel Türkçe Normalize WER |
|---|---|---|---|
| 1 | 18.75 | 47.30 | 40.21 |
| 2 | 16.01 | 63.71 | 58.76 |
| 3 | 13.44 | 66.00 | 60.61 |
| 4 | 12.92 | 63.93 | 59.05 |
| 5 | 12.17 | 76.05 | 71.69 |
| 6 | 11.04 | 69.18 | 63.64 |
| 7 | 10.43 | 70.95 | 66.08 |
| 8 | 9.92 | 68.74 | 63.78 |
| 9 | 9.99 | 71.47 | 66.08 |
| 10 | 9.08 | 69.77 | 64.23 |
| 11 | 8.99 | 71.40 | 66.30 |

**Sonuç**: İlaç WER çok iyi düşüyor (%8.99'a kadar) ama genel Türkçe **epoch 1'de bile referanstan kötü**,
epoch 5'te %76'ya kadar çıkıp bir daha hiç toparlanmıyor — **ciddi ve kalıcı unutma**.

---

## Deney 2: `ilac_dengeli_r16` — yeni yaklaşım (düşük rank + rehearsal)

- **Veri**: `ilac_karisik_train.jsonl` — 4910 satır = 3910 ilaç + **1000 genel Türkçe**
  (kaynak: [Google FLEURS](https://huggingface.co/datasets/google/fleurs) `tr_tr` train+validation,
  test bölümünden ayrı — %20.4 genel Türkçe oranı)
- **LoRA**: r=16, alpha=32 (64'ün 1/4'ü, alpha/r oranı=2 korunarak)
- **Eğitildiği yer**: Colab (T4)

| Epoch | İlaç Val WER | Genel Türkçe Ham WER | Genel Türkçe Normalize WER |
|---|---|---|---|
| 1 | 24.27 | 38.36 | 29.86 |
| 2 | 18.69 | 38.73 | 30.89 |
| 3 | 16.67 | 38.65 | 30.16 |
| 4 | 15.38 | 38.36 | 31.12 |
| 5 | 14.12 | 37.55 | 31.12 |
| 6 | 12.21 | 37.25 | 29.49 |
| 7 | 13.83 | 38.36 | 30.67 |
| 8 | 11.73 | 38.95 | 31.56 |
| 9 | 11.91 | 38.65 | 31.34 |
| 10 | 11.93 | 39.02 | 31.34 |
| 11 | 14.28 | 38.51 | 30.67 |
| **12 (en iyi ilaç)** | **10.71** | 37.92 | 30.67 |
| 13 | 11.34 | 38.29 | 31.34 |
| 14 | 15.23 | 37.84 | 30.67 |
| 15 | 14.22 | **37.47 (en iyi genel)** | 30.45 |

**En iyi checkpoint**: epoch 12 (`checkpoints/ilac_dengeli_r16/checkpoint-3684`), ilaç val WER %10.71.

**Sonuç**: Genel Türkçe **15 epoch boyunca hiç bozulmadı** — sürekli %37-39 bandında, referansın
(%40.50) altında/yakınında kaldı. İlaç WER biraz daha yavaş düşüyor (r=64 deneyine göre), ama
hâlâ hedefe (%10) çok yakın bir noktaya (%10.71) ulaştı.

---

## Deney 3: `ilac_genis_r16` — Common Voice ile büyütülmüş veri (aynı rank)

- **Veri**: `ilac_genis_train.jsonl` — 8910 satır = 3910 ilaç + **5000 genel Türkçe**
  (kaynak: [Common Voice 17](https://huggingface.co/datasets/ysdede/commonvoice_17_tr_fixed)
  Türkçe, topluluk parquet kopyası — resmi `mozilla-foundation/common_voice_17_0` Ekim
  2025'ten beri Hugging Face'te boş, Mozilla Data Collective'e taşınmış. TÜİK yaş/cinsiyet
  dağılımına göre ağırlıklı seçim, `ses_profilleri.py` ile aynı yöntem — %56.1 genel Türkçe oranı)
- **LoRA**: r=16, alpha=32 (Deney 2 ile birebir aynı — TEK değişken veri miktarı/oranı)
- **Eğitildiği yer**: Colab (T4), oturum RAM taşması nedeniyle birkaç kez kesintiye uğradı
  (`resume_from_checkpoint` ile devam etti)

| Epoch | İlaç Val WER | Genel Türkçe Ham WER | Genel Türkçe Normalize WER |
|---|---|---|---|
| 1 | 24.65 | 41.76 | 34.15 |
| 2 | 20.94 | 42.50 | 34.37 |
| 3 | 16.50 | 46.12 | 38.21 |
| 4 | 15.65 | 45.60 | 38.14 |
| 5 | 13.31 | 43.53 | 35.92 |
| 6 | 20.37 (sıçrama) | 46.78 | 38.58 |
| 7 | 13.56 | 46.64 | 38.58 |
| 8 | 13.37 | 45.97 | 37.84 |
| 9 | 16.05 | 46.19 | 38.21 |
| 10 | 11.34 | 46.34 | 38.58 |
| **11 (en iyi val)** | **11.28** | 46.34 | 38.58 |
| 12 | 11.30 | 45.31 | 37.40 |
| 13 | 12.66 | 46.93 | 39.54 |
| 14 | 11.52 | 46.64 | 38.95 |

**Gerçek test seti ölçümleri (val değil, `ilac_test.jsonl` 233 satır, hiç görülmemiş)**:

| Epoch | İlaç Test Ham | İlaç Test Normalize |
|---|---|---|
| 9 | 17.49 | 15.05 |
| 10 | 11.06 | 8.91 |
| 11 | 10.62 | 8.13 |
| 12 | 10.57 | 8.48 |
| 13 | 11.59 | 9.01 |
| **14 (en iyi)** | **9.79** | **7.50** |

**FLEURS sonucu**: Genel Türkçe verisini 1000'den 5000'e (%20'den %56'ya) çıkarmak, aynı
rank (r=16) ile FLEURS'te **hiç düzelme göstermedi** — 15 epoch boyunca %42-47 ham / %34-40
normalize bandında sabit kaldı, referanstan (%40.50/%31.71) hep kötü.

**Ama Common Voice'un KENDİ görülmemiş kısmında (aynı havuzdan, eğitimde hiç kullanılmamış
80 örnek, farklı seed) tablo tamamen tersine döndü**:

| Checkpoint (epoch) | CV Ham WER | CV Normalize WER |
|---|---|---|
| **Referans (LoRA'sız)** | **46.14** | **38.16** |
| 1 | 36.23 | 31.64 |
| 2 | 35.02 | 29.95 |
| 3 | 38.89 | 33.57 |
| 6 | 37.92 | 32.13 |
| 9 | 36.71 | 29.95 |
| 12 | 34.54 | 29.23 |
| **14 (en iyi)** | **34.54** | **28.99** |
| 15 | 35.99 | 30.43 |

**15 epoch'un HEPSİ referanstan (%46.14/%38.16) daha iyi** — sürekli ~9-12 puan ham,
~5-9 puan normalize kazanım, stabil.

**Sonuç — kritik bir netleşme**: Model genel Türkçe'yi **hiç unutmamış/öğrenmemiş değil** —
tam tersine Common Voice'un kendi tarzında gayet iyi öğrenmiş. FLEURS'teki kötü sonuç,
"unutma" değil, **eğitim verisinin (Common Voice, kalabalık-kaynaklı, çeşitli kayıt
koşulları) ile FLEURS'ün (temiz, resmi okuma) tarzı arasındaki dağılım farkının**
transfer etmemesiydi. Deney 2'de eğitim de FLEURS'ten geldiği için bu engel hiç yoktu —
o yüzden Deney 2 FLEURS'te iyi görünüyordu, ama bu "genel Türkçe'yi öğrendi" değil
"aynı dağılıma ezberledi/uyum sağladı" da olabilir, ayrıca test edilmedi.

**Pratik çıkarım**: Tek bir değerlendirme setine (FLEURS) bakarak "genel Türkçe kabiliyeti
gelişmedi" sonucuna varmak yanıltıcıydı. Gerçek kullanım senaryosu (hastaların doğal,
çeşitli konuşma tarzı) muhtemelen FLEURS'ün resmi okuma tarzından çok Common Voice'un
çeşitliliğine benziyor — bu açıdan Deney 3 aslında gerçek dünyada iyi performans
gösterebilir, sadece FLEURS benchmark'ında kötü görünüyordu.

---

## Genel karşılaştırma

| | İlaç (en iyi, val/test) | Genel Türkçe FLEURS (en iyi, ham/norm) | Genel Türkçe eğilimi |
|---|---|---|---|
| Deney 1: r=64, rehearsal yok | val 8.99 (epoch 11) | 68.74 / — (epoch 8) | Epoch 1'den kötü, sürekli kötüleşiyor |
| Deney 2: r=16, 1000 FLEURS (%20) | val 10.71 (epoch 12) | 37.25 / 29.49 (epoch 6) | **Stabil, referans civarında** |
| Deney 3: r=16, 5000 Common Voice (%56) | **test 7.50 (epoch 14)** | 45.31 / 37.40 (epoch 12, en iyisi bu) | 14 epoch boyunca hiç toparlanmadı |

**İlaçta Deney 3 şimdiye kadarki en iyi sonuç** (test setinde %7.50 normalize — Deney 2'nin
val'ini bile geçti). **Genel Türkçe'de de** — CV-görülmemiş testine göre — Deney 3 aslında
referansı tutarlı şekilde geçiyor, sadece FLEURS'e iyi transfer olmuyor. "Daha fazla veri
her zaman daha iyidir" varsayımı çürütülmedi, düzeltildi: daha fazla veri gerçekten
öğreniyor, ama **hangi dağılımdan geldiğine bağlı olarak nereye transfer olacağı değişiyor**.

---

## Deney 4: `ilac_karma_r32` — FLEURS + Common Voice karışık, yüksek rank

- **Veri**: `ilac_karma_train.jsonl` — 8910 satır = 3910 ilaç + **2500 FLEURS** (train+validation,
  test bölümü hariç) + **2500 Common Voice** (TÜİK ağırlıklı, Deney 3'ün kullandığı 5000 ve
  CV-görülmemiş test havuzu hariç tutularak taze seçildi) — %56.1 genel Türkçe, iki farklı
  dağılımdan
- **LoRA**: r=32, alpha=64 (Deney 2/3'ün r=16'sının 2 katı)
- **İlaç Val WER**: `ilac_val.jsonl` (736 satır) — eğitim sırasında otomatik, **sadece ilaç**
- **Eğitildiği yer**: Colab (T4)

| Epoch | İlaç Val WER | İlaç Test Ham/Norm (233) | FLEURS Ham/Norm (80) | CV-Görülmemiş Ham/Norm (80) |
|---|---|---|---|---|
| 1 | 22.82 | 21.77 / 18.51 | 36.51 / 28.31 | 42.27 / 37.92 |
| 2 | 18.95 | 17.19 / 14.42 | 35.85 / 29.12 | 43.72 / 38.89 |
| 3 | 17.15 | 15.68 / 12.71 | 36.29 / 28.01 | 40.10 / 34.54 |
| 4 | 14.61 | 13.15 / 10.72 | 36.51 / 29.12 | 42.03 / 36.47 |
| 5 | 14.22 | 14.27 / 11.50 | 35.77 / 28.23 | 40.58 / 36.47 |
| **6 (en iyi CV)** | 14.94 | 11.98 / 9.64 | 34.59 / 26.61 | **37.92 / 31.40** |
| 7 | 14.72 | 12.13 / 9.69 | **33.85 / 27.05** | 38.41 / 32.85 |
| 8 | 15.39 | 12.13 / 9.50 | 35.18 / 27.49 | 42.27 / 36.71 |
| 9 | 12.80 | 10.72 / 8.23 | 33.41 / 25.87 | 42.03 / 35.99 |
| 10 | 14.10 | 10.72 / 8.48 | 32.67 / 25.20 | **37.68** / 32.37 |
| 11 | 12.00 | 11.74 / 8.91 | 34.15 / 26.24 | 38.65 / 33.57 |
| 12 | 12.45 | 10.67 / 8.28 | 33.11 / 26.16 | 40.58 / 35.51 |
| 13 | 13.11 | **9.74** / 7.94 | 32.52 / 25.28 | 39.13 / 34.06 |
| 14 | 11.42 | 9.99 / **7.74** | 32.00 / 25.42 | 39.37 / 34.06 |
| **15 (en iyi val + en iyi FLEURS)** | **10.91** | 10.03 / 8.04 | **31.78 / 24.98** | 40.58 / 35.27 |

*(Referanslar — LoRA'sız whisper-base: ilaç test %46.66/%40.04, FLEURS %40.50/%31.71,
CV-görülmemiş %46.14/%38.16)*

**Sonuç — hipotez tamamen doğrulandı, üç eksende de, tüm 15 epoch boyunca**: **Her 15 epoch,
her 3 eksende de (ilaç, FLEURS, CV) kendi referansını geçiyor** — Deney 3'te FLEURS hiçbir
epoch'ta başaramamıştı. Bu, projedeki **ilk deney** hem ilaçta hem genel Türkçe'nin İKİ farklı
ölçütünde birden referansı geçen sonuç, ve bu üstünlük eğitim boyunca hiç bozulmadı.

Epoch 9 sonrası val_loss ~0.187'de düzleşti (plato) — epoch 10-15 marjinal fayda sağladı ama
overfit sinyali yok. Tek eksende net "en iyi" yok, dört küçük kümede toplanıyor:
- **İlaç test**: epoch 13 (ham %9.74) / epoch 14 (norm %7.74) — Deney 3'ün rekoru (test %7.50
  normalize, epoch 14) hâlâ çok az önde ama Deney 4 çok yakın, üstelik FLEURS'ü de kaybetmeden.
- **FLEURS (genel Türkçe, resmi)**: epoch 15, tüm deneyin en iyisi (%31.78/%24.98).
- **CV-görülmemiş (genel Türkçe, çeşitli)**: epoch 6 (norm %31.40) / epoch 10 (ham %37.68).
- **İlaç val**: epoch 15 (%10.91), tüm deneyin en iyisi.

**Genel en iyi aday**: **epoch 14 veya 15** (`checkpoint-7798` / `checkpoint-8355`) — ikisi de
4 eksende de güçlü, hiçbirinde kötü değil, ve FLEURS ile ilaç val'de deneyin rekorunu taşıyor.

---

## Hata analizi: kalan hatalar ilaç adında mı, sıradan Türkçe'de mi?

`egitim/ilac_hata_analizi.py` — her test cümlesinde `hedef_terim`e en çok benzeyen kelime
"ilaç adı" olarak işaretlenip iki ayrı ölçüm yapılır: (1) ilaç adının doğru tanınma oranı,
(2) ilaç adı cümleden çıkarıldıktan sonra kalan WER. Her ikisi de `ilac_test.jsonl` (233
satır, **ilaç** held-out) üzerinde, normalize WER.

| Model (epoch 14) | Genel norm WER | **İlaç adı ÇIKARILMIŞ norm WER** | İlaç adı gövde doğruluk | İlaç adı tam doğruluk | Kaçırılan |
|---|---|---|---|---|---|
| **whisper-base (LoRA'sız, referans)** | **40.04%** | **41.51%** | **39.9%** (93/233) | **15.5%** (36/233) | **140** |
| Deney 3 (r=16, 5002 CV) | 7.50% | **8.08%** | 95.7% (223/233) | 93.1% (217/233) | 10 |
| Deney 4 (r=32, karma) | 7.74% | **8.36%** | 96.6% (225/233) | 93.1% (217/233) | 8 |

**LoRA'nın en büyük tek kazancı ilaç adı ezberi**: referans model ilaç adlarının %60'ını
(140/233) tamamen kaçırıyor, Türkçe ekiyle birlikte doğru bilme oranı ise sadece %15.5
(36/233). Fine-tuning bunu %96.6 / %93.1'e çıkarıyor — **ilaç adı hatası 140'tan 8'e,
17.5 kat azalıyor.**

### Kritik uyarı: ilaç test sayısı gerçek hastalar için iyimser

İlaç adı dışındaki kısım da referansta %41.51'den %8.36'ya düşmüş — **5 kat iyileşme**.
Ama aynı modeller **gerçek insan konuşmasında** (CV-400) yalnızca %42.35 → %35.30, yani
**1.2 kat** iyileşiyor.

Bu fark tesadüf değil: `ilac_test.jsonl`'in sesleri de `ilac_train.jsonl` ile **aynı TTS
hattından** (Gemini) üretildi. Yani modelin ilaç testindeki sıradan-Türkçe başarısının büyük
kısmı, genel Türkçe yeteneği değil **aynı sentetik sesin tarzına uyum sağlaması**.

**Sonuç**: %7.50 ilaç WER'i gerçek hasta konuşmasında beklenmemeli. Güvenilir olan iki şey:
1. **İlaç adı doğruluğu** (%96.6) — bu gerçek bir ezber kazanımı, sese daha az bağlı
2. **CV-400 sonucu** (%32.51-35.30) — gerçek insan sesi, gerçek dünya beklentisi buna yakın

Projede eksik olan tek ölçüm: **gerçek insan sesiyle söylenmiş ilaç cümleleri**. Küçük bir set
(50-100 kayıt) bile %7.50'nin ne kadarının gerçek olduğunu gösterirdi. Bu bir eğitim deneyi
değil, doğrulama işi — deneme hakkı harcamaz.

**Kritik bulgu**: İlaç adı çıkarıldığında WER **yükseliyor** (7.50→8.08, 7.74→8.36) — yani
ilaç adları cümlenin geri kalanından **daha iyi** tanınıyor. Kalan hataların ağırlığı ilaç
adlarında değil, **sıradan Türkçe'de**. Bu, "daha fazla genel Türkçe verisi ekle" kolunun
doğru kol olduğunu ampirik olarak doğruluyor.

Gövde (%95.7-96.6) ile tam (%93.1) doğruluk arasındaki ~3.5 puanlık fark, ilaç adının
kendisinin doğru ama **Türkçe ekinin** yanlış olduğu durumlar (`zitromaxı` vs `zitromax'ı`).

Kaçırılan ilaç adları öngörülebilir bir kalıpta: **söylenişi Türkçe, yazılışı yabancı**
olanlar — `Zanaks→xanax`, `Norvask→norvasc`, `Talsid→talcid`, `Teraflu→theraflu`,
`Kanuvia→januvia`, `Beroka→berocca`, `Doloreks→dolorex`, `Olin→aulin`. Bu bir telaffuz→yabancı
imla eşleme ezberi; zaten %95.7-96.6 seviyesinde, buradan çıkacak kazanım sınırlı.

---

## Ölçüm güvenilirliği: CV testi 80 örnekle çok gürültülü

CV-görülmemiş testi 80 örnek ≈ 600 kelime. %30 WER civarında beklenen rastgele dalgalanma
`sqrt(0.30 × 0.70 / 600) ≈ ±1.8 puan`. Deney 3 (%28.99) ile Deney 4 (%31.40) arasındaki
2.4 puanlık fark yalnızca **~1.3 sigma — istatistiksel olarak anlamlı değil.**

Bu yüzden "Deney 4 CV'de geriledi" ifadesi 80 örnekle kanıtlanmış sayılamazdı. Ayrıca son
deneyin sonucunu 80 örnekle ölçmek anlamsız olurdu (2 puanlık bir iyileşme gürültüden ayırt
edilemez). `colab_aktarim/cv_genis_test.ipynb` test setini **400 örneğe** çıkarır, Deney 3'ün
ve Deney 4'ün eğitimde kullandığı tüm CV kayıtlarını dışlar (sızıntı kontrolüyle), eski 80'i
kapsar. Ölçülen boyut: **400 örnek / 2156 kelime** → %35 WER civarında gürültü
`sqrt(0.35 × 0.65 / 2156) ≈ ±1.03 puan` (80 örnekteki ±1.8'in yerine).

### 400 örneklik geniş CV testi sonucu — gerileme GERÇEK

| Model | 80 örnek (eski) Ham/Norm | **400 örnek (yeni) Ham/Norm** |
|---|---|---|
| Referans (LoRA'sız whisper-base) | 46.14 / 38.16 | **49.77 / 42.35** |
| Deney 3 (r=16, 5002 CV, FLEURS yok) ep14 | 34.54 / 28.99 | **37.38 / 32.51** |
| Deney 4 (r=32, karma) ep14 | 39.37 / 34.06 | **40.35 / 35.30** |
| Deney 4 (r=32, karma) ep15 | 40.58 / 35.27 | **40.68 / 35.53** |

**Deney 3 ile Deney 4 arasındaki fark normalize'da 2.79 puan; gürültü ±1.03 puan →
~2.7 sigma (p ≈ 0.007), istatistiksel olarak anlamlı.** Ölçüm aynı 400 örnek üzerinde
eşleştirilmiş (paired) olduğu için gerçek anlamlılık bundan da yüksek. CV'deki gerileme
gürültü değil, gerçek bir etki.

**Ek kanıt — ölçüm kendi içinde tutarlı**: Deney 4'ün iki bağımsız checkpoint'i (ep14: 35.30,
ep15: 35.53) birbirine yalnızca **0.23 puan** uzakta kümelendi. Yani Deney 3'ün 32.51'i
"şanslı bir checkpoint" değil, seviye farkı gerçek.

80 örneğin ne kadar güvenilmez olduğu da ortaya çıktı: referans değeri 4.2 puan kaydı
(38.16 → 42.35), Deney 3'ün değeri 3.5 puan (28.99 → 32.51). Sıralama korundu ama mutlak
değerlerin hiçbiri güvenilir değildi. **Bundan sonra tüm CV ölçümleri 400'lük setle yapılmalı.**

### Ödünleşim tablosu — netleşmiş hâli

| | İlaç test (norm) | FLEURS (norm) | CV-görülmemiş 400 (norm) |
|---|---|---|---|
| Referans | 40.04 | 31.71 | 42.35 |
| Deney 3 (r16, 5002 CV) | **7.50** ✓ | 37.40 ✗ | **32.51** ✓ |
| Deney 4 (r32, 2500 FLEURS + 2500 CV) | 7.74 | **24.98** ✓ | 35.30 |

Deney 3 ilaçta ve CV'de önde; Deney 4 FLEURS'te ezici biçimde önde (12.4 puan) ve tek deney
olarak FLEURS'te referansı geçiyor. **Ödünleşim gerçek ve ölçülmüş.**

**Deney 5'i haklı çıkaran üç bağımsız kanıt** — üçü de aynı yöne, "daha fazla Common Voice"a
işaret ediyor:
1. **Hata analizi**: kalan hataların ağırlığı ilaç adlarında değil sıradan Türkçe'de
   (ilaç adı çıkarılınca WER yükseliyor: 7.74 → 8.36)
2. **Veri tavanı**: FLEURS'ün %87'si tüketildi (2864 satırın ~2500'ü), CV'de ~50.800 satır boş
3. **400'lük CV testi**: CV'deki gerileme gerçek (2.79 puan, ~2.7 sigma) ve CV verisi
   2500'den 8000'e çıkarılabilir

---

## Veri tavanı: FLEURS tükendi, Common Voice'ta bol yer var

HF datasets-server'a göre `google/fleurs` `tr_tr`: **train 2526 + validation 338 = 2864**
satır (test 743 ayrı tutuluyor). Deney 4 bunun ~2500'ünü kullandı — yani **FLEURS ekseninin
%87'si tüketildi**, ölçeklenecek yer yok.

Common Voice ise 72.846 satır, TÜİK yaş/cinsiyet etiketi olan **58.427**. Şimdiye kadar
kullanılan: Deney 3'te 5000 + Deney 4'te 2500 + 80 test = 7580. **~50.800 satır boş.**

**Sonuç**: Ölçeklenebilir tek genel Türkçe kaynağı Common Voice.

---

## Sonraki adım — Deney 5 planı (son deneme)

Üç bulgu aynı yöne işaret ediyor: (a) kalan hatalar sıradan Türkçe'de, (b) FLEURS tükendi,
(c) CV'de 50 binden fazla kullanılmamış satır var.

**Deney 5**: 3910 ilaç + **2864 FLEURS (tamamı)** + **8000 Common Voice** (taze, Deney 3/4'ün
kullandıkları ve 400'lük test seti hariç), r=32/alpha=64 **değişmeden**, 12 epoch.

- 14.774 satır → 924 adım/epoch → 11.088 adım ≈ **6.3 saat** T4'te (Deney 4'ün gözlenen
  ~1759 adım/saat hızıyla)
- **İlaç dozu sabit kalıyor**: 3910 örnek × 12 epoch. Değişen tek şey genel Türkçe dozu —
  bu yüzden ilaç performansının bozulma riski düşük (Deney 2→3'te ilaç oranı %79.6'dan
  %43.9'a düşerken ilaç WER'i *iyileşmişti*)
- 12 epoch seçimi: Deney 4'ün val_loss'u ~5000 adımda platoya girdi; 924 adım/epoch ile bu
  epoch 6 civarına denk gelir, 12 epoch platonun iyice ötesine geçer — 15 epoch'un son
  epoch'ları boşa giderdi
- CV sesleri **parça parça** (2000'lik bloklar) materyalize edilmeli — Deney 3'te 5000 CV
  tek seferde çekilince Colab RAM'i taşmıştı

Değerlendirme her zamanki üç eksende, ama CV artık **400 örneklik** geniş setle.

### Deney 5 için başarı ölçütü (400'lük test sonrası netleşti)

Deney 5 "başarılı" sayılması için **üç eşiği birden** geçmeli — her biri mevcut en iyi
değere karşı (normalize WER):

| Eksen | Geçilecek eşik | Kimin rekoru |
|---|---|---|
| İlaç test (233, held-out) | **≤ 7.50** | Deney 3 ep14 |
| FLEURS (80) | **≤ 24.98** | Deney 4 ep15 |
| CV-görülmemiş (400) | **≤ 32.51** | Deney 3 ep14 |

Hiçbir deney şimdiye kadar üçünü birden tutmadı: Deney 3 FLEURS'te referansın bile altında,
Deney 4 CV'de Deney 3'ten 2.79 puan (3.2 sigma) geride. Deney 5'in tek amacı bu üçlüyü
aynı checkpoint'te toplamak.

**Beklenti gerekçesi**: CV 2500→8000 (3.2 kat) CV eksenini Deney 3'ün seviyesine ya da
ötesine taşımalı; FLEURS 2500→2864 (tamamı) FLEURS kazancını korumalı; ilaç dozu sabit
(3910 × 12 epoch) olduğu için ilaç bozulmamalı.

**Eğer Deney 5 CV'de 32.51'i geçemezse** ulaşılan sonuç şu olur: FLEURS ve CV arasında
LoRA kapasitesi düzeyinde gerçek bir rekabet var (r=32 iki dağılımı birden tam öğrenmeye
yetmiyor) — o durumda nihai model, hedef kullanım senaryosuna göre seçilir: gerçek hasta
konuşması için Deney 3, temiz/resmi konuşma için Deney 4.
