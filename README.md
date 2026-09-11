# whisper-base-medical-tr

OpenAI'nin **Whisper Base** konuşma tanıma modelinin, **LoRA (Low-Rank Adaptation)** tekniğiyle Türkçe tıbbi ön-görüşme konuşmalarını (hasta şikayetleri, semptomlar, ilaç isimleri dahil) tanıyacak şekilde özelleştirilmesi projesi.

Amaç, günlük hayatta kullanılabilecek, güvenilir bir tıbbi konuşma tanıma sistemi ortaya koymaktı — bunu yaparken modelin genel Türkçe konuşma tanıma yeteneğini kaybetmemesi (*catastrophic forgetting*'den kaçınılması) kritik bir mühendislik kısıtı olarak ele alındı.

## İçindekiler

- [`models/checkpoint-7712/`](models/checkpoint-7712) — en iyi LoRA adaptörü (Deney 5, bkz. Bulgular)
- [`data/`](data) — eğitim metni korpüsü, ses klipleri, split dosyaları
- [`scripts/`](scripts) — veri üretim ve eğitim kodu
- [`notebooks/`](notebooks) — Google Colab notebook'ları (ses üretimi, eğitim, test)
- [`docs/`](docs) — deney raporları ve bulgular

## Veri Seti

- **10.314 satır** sentetik hasta cümlesi — [`data/hasta_cumleleri.jsonl`](data/hasta_cumleleri.jsonl)
  - 18 tıbbi dal (dahiliye, kardiyoloji, nöroloji, ortopedi, kbb, göz, dermatoloji, pediatri, psikiyatri, kadın doğum, üroloji, romatoloji, endokrinoloji, enfeksiyon, diş, göğüs hastalıkları, gastroenteroloji, acil tıp) + ilaç + ortak (branşsız zaman/şiddet ifadeleri) kategorileri
  - Klinik anamnez alma yöntemine dayalı **SOCRATES** çerçevesiyle (site, onset, character, radiation, associations, time_course, exacerbating/relieving, severity) kurgulandı
  - İki hasta personası: 42 yaşında sakin/kısa konuşan erkek, 67 yaşında detaylı/şikayetçi konuşan kadın
- **6,29 saat** (5.512 klip, ortalama 4,11 sn/klip) sentetik konuşma sesi — [`data/sesler/`](data/sesler) (Git LFS)
  - Chatterbox Multilingual TTS ile, TÜİK'in yaş/cinsiyet nüfus dağılımına göre ağırlıklandırılmış 48 referans sesten ([`data/ses_referans/`](data/ses_referans)) üretildi
  - Çift katmanlı kalite kontrolü: süre eşiği (kelime başına ≥0,12sn) + Whisper-small geri-transkript doğrulaması (≥%70 eşleşme)
- Ayrıca hazır kaynaklar kullanıldı: Google FLEURS `tr_tr` (2.864 satır) ve Common Voice 17 Türkçe (58.427 satırlık havuzdan TÜİK ağırlıklı seçim)

## Model

- **Taban model:** [OpenAI Whisper Base](https://huggingface.co/openai/whisper-base) (74M parametre)
- **İnce ayar yöntemi:** LoRA (r=32, alpha=64, dropout=0,05)
- **En iyi checkpoint:** `checkpoint-7712` ([`models/checkpoint-7712/`](models/checkpoint-7712)) — bkz. Bulgular

### Kullanım

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
model = PeftModel.from_pretrained(base, "models/checkpoint-7712")
processor = WhisperProcessor.from_pretrained("models/checkpoint-7712")
```

## Bulgular

Beş karşılaştırmalı deney yürütüldü; her biri LoRA rank'i ve eğitim verisine karıştırılan genel Türkçe verisinin (rehearsal) kaynağı değiştirilerek tasarlandı. Performans üç bağımsız eksende ölçüldü: ilaç (hedef görev), FLEURS (resmi/temiz Türkçe), Common Voice (çeşitli/gerçekçi Türkçe).

| Deney | Veri | LoRA | İlaç (norm.) | FLEURS (norm.) | CV (norm.) |
|---|---|---|---|---|---|
| 1 — rehearsal yok | 3.910, sadece ilaç | r=64/128 | val %8,99 | %68,74 | — |
| 2 — düşük rank + FLEURS | 4.910 (+%20 FL) | r=16/32 | val %10,71 | %37,25 | — |
| 3 — Common Voice ile büyütme | 8.910 (+%56 CV) | r=16/32 | test %7,50 | %45,31 | %32,51 |
| 4 — FLEURS+CV karma | 8.910 (FL+CV) | r=32/64 | test %8,04 | %31,78 | %35,30 |
| **5 — genişletilmiş çok-dallı rehearsal** | **10.314, 18 dal** | r=32/64 | **%19,53** | **%24,32** | **%30,33** |

**Ana bulgular:**

1. **İlaç adı ezberi, LoRA'nın en büyük kazancı:** Referans model (LoRA'sız) ilaç isimlerinin %60'ını (140/233) tamamen kaçırırken, ince ayarlı model bu oranı %93,1 tam doğruluğa çıkardı.
2. **Genel Türkçe'nin korunması tek bir kaynakla mümkün değil:** Rehearsal'sız eğitim (Deney 1) ciddi/kalıcı unutmaya yol açtı. Tek kaynaklı rehearsal (Deney 2: sadece FLEURS, Deney 3: sadece Common Voice) bir eksende işe yarayıp diğerine transfer olmadı. FLEURS+CV karışımı (Deney 4) ilk kez her iki dağılımı birden referansın üzerine çıkardı.
3. **Deney 5 — önemli bir trade-off:** Metin korpüsü 10.314 satıra çıkarılıp 18 tıbbi dala yayılınca (yalnızca ilaç değil), model genel tıbbi Türkçe'yi (FLEURS, Common Voice) daha iyi öğrendi, **ancak ilaç adı tanıma performansından belirgin ölçüde ödün verdi** (Deney 4'ün %8,04'üne kıyasla %19,53). **Bu, kabul edilebilir bir trade-off olarak değerlendirilmedi** — ilaç adı tanıma projenin öncelikli hedefi. Eğitim verisindeki ilaç/genel-dal oranının bu kaybı önleyecek şekilde yeniden dengelenmesi bir sonraki adım olarak planlanıyor.
4. **Quantization bu ölçekte işe yaramadı:** İki platformda (Mac/qnnpack, Colab/bitsandbytes) denendi, ikisinde de hem hız hem WER kaybı gözlendi.

Detaylı deney raporları: [`docs/deney_ozetleri/`](docs/deney_ozetleri) ve [`docs/rapor_whisper_medikal_tr.html`](docs/rapor_whisper_medikal_tr.html).

## Yöntem Özeti

1. **Sentetik veri üretimi** — SOCRATES çerçevesi + dengeli tur kuyruk düzeniyle hasta cümleleri üretildi ([`scripts/veri_uretimi/`](scripts/veri_uretimi)); her cümle meta-sızıntı, yabancı kelime, kelime sayısı ve birebir-tekrar kontrolünden geçirildi.
2. **Ses üretimi** — Chatterbox Multilingual TTS ile Colab GPU üzerinde seslendirme ([`notebooks/ses_uretim_colab.ipynb`](notebooks/ses_uretim_colab.ipynb)).
3. **Eğitim** — LoRA ince ayar, val_wer/test_wer/loss izlenerek karar verildi ([`scripts/egitim/`](scripts/egitim), [`notebooks/ilac_genel8k_r32_egitim.ipynb`](notebooks/ilac_genel8k_r32_egitim.ipynb)).
4. **Değerlendirme** — İlaç test seti (görülmemiş 233 satır) + FLEURS test + Common Voice görülmemiş kısmı (400 örnek, sızıntı kontrollü, bkz. [`notebooks/cv_genis_test.ipynb`](notebooks/cv_genis_test.ipynb)).

## Lisans

Bu depo bir staj/araştırma projesinin çıktısıdır. Taban model (Whisper Base) OpenAI'nin kendi lisansına tabidir. Üretilen sentetik veri ve LoRA ağırlıkları için lisans henüz belirlenmemiştir.
