# whisper-base-medical-tr

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

OpenAI'nin **Whisper Base** konuşma tanıma modelinin, **LoRA (Low-Rank Adaptation)** tekniğiyle Türkçe tıbbi ön-görüşme konuşmalarını (hasta şikayetleri, semptomlar, ilaç isimleri dahil) tanıyacak şekilde özelleştirilmesi projesi.

Amaç, günlük hayatta kullanılabilecek, güvenilir bir tıbbi konuşma tanıma sistemi ortaya koymaktı — bunu yaparken modelin genel Türkçe konuşma tanıma yeteneğini kaybetmemesi (*catastrophic forgetting*'den kaçınılması) kritik bir mühendislik kısıtı olarak ele alındı.

## 🩺 SOCRATES-TR — Veri Seti

Bu projenin en önemli çıktısı, kendi ürettiğimiz **SOCRATES-TR** veri setidir: Türkçe tıbbi ön-görüşme konuşmalarını kapsayan, hem metin hem sesli, tamamen sentetik (gerçek hasta kaydı içermeyen) bir korpüs. Bildiğimiz kadarıyla 18 farklı tıbbi dalı bir arada kapsayan, açık şekilde paylaşılan ilk Türkçe tıbbi konuşma veri seti.

- **10.314 satır** sentetik hasta cümlesi — [`data/hasta_cumleleri.jsonl`](data/hasta_cumleleri.jsonl)
  - 18 tıbbi dal (dahiliye, kardiyoloji, nöroloji, ortopedi, kbb, göz, dermatoloji, pediatri, psikiyatri, kadın doğum, üroloji, romatoloji, endokrinoloji, enfeksiyon, diş, göğüs hastalıkları, gastroenteroloji, acil tıp) + ilaç + ortak (branşsız zaman/şiddet ifadeleri) kategorileri
  - Adını aldığı klinik anamnez alma yöntemi **SOCRATES** çerçevesiyle (site, onset, character, radiation, associations, time_course, exacerbating/relieving, severity) kurgulandı — her cümle bir şikayetin bu sekiz yönünden birini ifade eder
  - İki hasta personası: 42 yaşında sakin/kısa konuşan erkek, 67 yaşında detaylı/şikayetçi konuşan kadın
- **6,29 saat** (5.512 klip, ortalama 4,11 sn/klip) sentetik konuşma sesi — [`data/sesler/`](data/sesler) (Git LFS)
  - Chatterbox Multilingual TTS ile, TÜİK'in yaş/cinsiyet nüfus dağılımına göre ağırlıklandırılmış 48 referans sesten ([`data/ses_referans/`](data/ses_referans)) üretildi
  - Çift katmanlı kalite kontrolü: süre eşiği (kelime başına ≥0,12sn) + Whisper-small geri-transkript doğrulaması (≥%70 eşleşme)
  - Her cümle meta-sızıntı, yabancı kelime, kelime sayısı ve birebir-tekrar kontrolünden geçirildi

Veri seti bu depoda `data/` altında tam olarak yer alıyor; nasıl üretildiğinin kod ve notebook'ları [`scripts/veri_uretimi/`](scripts/veri_uretimi) ve [`notebooks/`](notebooks) altında.

Ayrıca modelin genel Türkçe'yi unutmamasını sağlamak için hazır kaynaklar da eğitime karıştırıldı: Google FLEURS `tr_tr` (2.864 satır) ve Common Voice 17 Türkçe (58.427 satırlık havuzdan TÜİK ağırlıklı seçim) — bunlar SOCRATES-TR'nin parçası değil, ayrı hazır veri setleridir.

## İçindekiler

- [`data/`](data) — **SOCRATES-TR**: eğitim metni korpüsü, ses klipleri, split dosyaları
- [`models/checkpoint-7712/`](models/checkpoint-7712) — Deney 5'in LoRA adaptörü (tek bir "en iyi model" yok, bkz. Bulgular)
- [`scripts/`](scripts) — veri üretim ve eğitim kodu
- [`notebooks/`](notebooks) — Google Colab notebook'ları (ses üretimi, eğitim, test)
- [`docs/`](docs) — deney raporları ve bulgular

## Model

- **Taban model:** [OpenAI Whisper Base](https://huggingface.co/openai/whisper-base) (74M parametre)
- **İnce ayar yöntemi:** LoRA (r=32, alpha=64, dropout=0,05)
- **Bu depoda paylaşılan checkpoint:** `checkpoint-7712` (Deney 5 — genel tıbbi Türkçe'de en iyi, ilaç WER'inde Deney 3'e göre geride; bkz. Bulgular)

### Kurulum

```bash
git clone https://github.com/halfjedi/whisper-base-medical-tr.git
cd whisper-base-medical-tr
pip install -r requirements.txt
```

### Kullanım

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
model = PeftModel.from_pretrained(base, "models/checkpoint-7712")
processor = WhisperProcessor.from_pretrained("models/checkpoint-7712")
```

> Not: `models/checkpoint-7712` yerel bir depo yoludur — yukarıdaki kodun çalışması için önce depoyu klonlaman gerekir. Hugging Face Hub üzerinden doğrudan (`from_pretrained("kullanici/model-adi")`) dağıtım henüz yapılmadı.

## Bulgular

Beş karşılaştırmalı deney yürütüldü; her biri LoRA rank'i ve eğitim verisine karıştırılan genel Türkçe verisinin (rehearsal) kaynağı değiştirilerek tasarlandı. Performans üç bağımsız eksende ölçüldü: ilaç (hedef görev), FLEURS (resmi/temiz Türkçe), Common Voice (çeşitli/gerçekçi Türkçe). Referans nokta, hiçbir ince ayar yapılmamış taban model:

| | İlaç (norm.) | FLEURS (norm.) | CV (norm.) |
|---|---|---|---|
| **whisper-base (LoRA'sız, referans)** | %40,04 | %31,78 | %42,35 |

Aşağıdaki tüm sonuçlar bu referansa göre okunmalı — örneğin Deney 5'in %19,53'lük ilaç WER'i tek başına vasat görünse de referansın (%40,04) yarısından azına iner:

| Deney | Veri | LoRA | İlaç (norm.) | FLEURS (norm.) | CV (norm.) |
|---|---|---|---|---|---|
| 1 — rehearsal yok *(katastrofik unutma)* | 3.910, sadece ilaç | r=64/128 | val %8,99 | %68,74 | — |
| 2 — düşük rank + FLEURS | 4.910 (+%20 FL) | r=16/32 | val %10,71 | %37,25 | — |
| 3 — Common Voice ile büyütme | 8.910 (+%56 CV) | r=16/32 | **test %7,50 🏆 en iyi ilaç** | %45,31 | %32,51 |
| 4 — FLEURS+CV karma | 8.910 (FL+CV) | r=32/64 | test %8,04 | %31,78 | %35,30 |
| 5 — genişletilmiş çok-dallı rehearsal | 10.314, 18 dal | r=32/64 | %19,53 | **%24,32 🏆 en iyi FLEURS** | **%30,33 🏆 en iyi CV** |

Tek bir "en iyi model" yok — hangisi daha uygun, önceliğe göre değişiyor:

- **İlaç adı tanıma önceliğiyse:** Deney 3 (test norm. %7,50) — katastrofik unutma yaşamayan deneyler arasında (2/3/4/5) en iyi ilaç WER'i bu deneyde elde edildi, Deney 4'ü (%8,04) bile geçiyor.
- **Genel tıbbi Türkçe (FLEURS/CV) + makul ilaç performansı önceliğiyse:** Deney 5 / `checkpoint-7712` (bu depoda paylaşılan model) — FLEURS ve CV'de en iyi sonuçlar, ama ilaç WER'i Deney 3/4'e göre belirgin şekilde geride (%19,53).

**Ana bulgular:**

1. **İlaç adı ezberi, LoRA'nın en büyük kazancı:** Referans model (LoRA'sız) ilaç isimlerinin %60'ını (140/233) tamamen kaçırırken, ince ayarlı model bu oranı %93,1 tam doğruluğa çıkardı.
2. **Genel Türkçe'nin korunması tek bir kaynakla mümkün değil:** Rehearsal'sız eğitim (Deney 1) ciddi/kalıcı unutmaya yol açtı. Tek kaynaklı rehearsal (Deney 2: sadece FLEURS, Deney 3: sadece Common Voice) bir eksende işe yarayıp diğerine transfer olmadı. FLEURS+CV karışımı (Deney 4) ilk kez her iki dağılımı birden referansın üzerine çıkardı.
3. **Deney 5 — önemli bir trade-off:** Metin korpüsü 10.314 satıra çıkarılıp 18 tıbbi dala yayılınca (yalnızca ilaç değil), model genel tıbbi Türkçe'yi (FLEURS, Common Voice) daha iyi öğrendi, **ancak ilaç adı tanıma performansından belirgin ölçüde ödün verdi** (Deney 3'ün %7,50'sine kıyasla %19,53). **Bu, kabul edilebilir bir trade-off olarak değerlendirilmedi** — ilaç adı tanıma projenin öncelikli hedefi. Eğitim verisindeki ilaç/genel-dal oranının bu kaybı önleyecek şekilde yeniden dengelenmesi bir sonraki adım olarak planlanıyor.
4. **Quantization bu ölçekte işe yaramadı:** İki platformda (Mac/qnnpack, Colab/bitsandbytes) denendi, ikisinde de hem hız hem WER kaybı gözlendi.

Detaylı deney raporları: [`docs/deney_ozetleri/`](docs/deney_ozetleri) ve [`docs/rapor_whisper_medikal_tr.html`](docs/rapor_whisper_medikal_tr.html).

## Yöntem Özeti

1. **Sentetik veri üretimi** — SOCRATES çerçevesi + dengeli tur kuyruk düzeniyle hasta cümleleri üretildi ([`scripts/veri_uretimi/`](scripts/veri_uretimi)); her cümle meta-sızıntı, yabancı kelime, kelime sayısı ve birebir-tekrar kontrolünden geçirildi.
2. **Ses üretimi** — Chatterbox Multilingual TTS ile Colab GPU üzerinde seslendirme ([`notebooks/ses_uretim_colab.ipynb`](notebooks/ses_uretim_colab.ipynb)).
3. **Eğitim** — LoRA ince ayar, val_wer/test_wer/loss izlenerek karar verildi ([`scripts/egitim/`](scripts/egitim), [`notebooks/ilac_genel8k_r32_egitim.ipynb`](notebooks/ilac_genel8k_r32_egitim.ipynb)).
4. **Değerlendirme** — İlaç test seti (görülmemiş 233 satır) + FLEURS test + Common Voice görülmemiş kısmı (400 örnek, sızıntı kontrollü, bkz. [`notebooks/cv_genis_test.ipynb`](notebooks/cv_genis_test.ipynb)).

## Lisans

Bu depodaki kod, SOCRATES-TR veri seti ve LoRA ağırlıkları [MIT lisansı](LICENSE) ile paylaşılmıştır. Taban model (Whisper Base) ayrıca OpenAI'nin kendi MIT lisansına tabidir.
