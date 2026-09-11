"""
Whisper LoRA eğitimi için veri hazırlama: augmentation (set_transform) ve
batch padding (data collator) fonksiyonları.

İki ayrı aşama:
  1. set_transform (dataset seviyesi): ham .wav -> augment edilmiş log-mel
     özellikler + token ID'ler. Sadece train split'te rastgele bozulma
     uygulanır (gürültü/reverb/kazanç/iki-cümle birleştirme), val/test hep
     temiz kalır. Kalıcı diske yazılmaz - her erişimde yeniden hesaplanır.
  2. DataCollatorSpeechSeq2SeqWithPadding (Trainer seviyesi): bir batch'teki
     örnekleri alıp aynı uzunluğa doldurup (padding) tensöre çevirir.

Kullanım (demo/smoke-test):
    source venv/bin/activate
    python egitim/veri_hazirlama.py
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torchaudio
from transformers import TrainerCallback


class MPSBellekTemizleCallback(TrainerCallback):
    """Her adim sonunda torch.mps.empty_cache() cagirir. MPS backend'i
    kullanilmayan onbellegi otomatik isletim sistemine iade etmiyor - bu,
    zamanla swap'a tasan bellek buyumesine yol aciyor (tani testiyle
    dogrulandi). Kucuk bir senkronizasyon maliyeti karsiliginda bu
    birikmeyi engeller."""

    def on_step_end(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# 1. Ham dalga formu üzerinde çalışan augmentation fonksiyonları
# ---------------------------------------------------------------------------

def gurultu_ekle(dalga: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Beyaz gürültüyü, istenen SNR (sinyal/gürültü oranı, dB) ile ekler."""
    sinyal_gucu = dalga.pow(2).mean()
    gurultu = torch.randn_like(dalga)
    gurultu_gucu = gurultu.pow(2).mean()
    hedef_gurultu_gucu = sinyal_gucu / (10 ** (snr_db / 10))
    olcek = (hedef_gurultu_gucu / gurultu_gucu).sqrt()
    return dalga + gurultu * olcek


def reverb_uygula(dalga: torch.Tensor, sr: int = 16000, sure_s: float = 0.3,
                   sonum_faktoru: float = 6.0, islak_oran: float = 0.3) -> torch.Tensor:
    """Sentetik, üstel sönümlü bir oda tepkisiyle (impulse response) konvolüe
    ederek basit bir yankı/reverb ekler. Gerçek IR kaydı gerektirmez."""
    n = int(sr * sure_s)
    t = torch.linspace(0, 1, n)
    zarf = torch.exp(-sonum_faktoru * t)
    ir = torch.randn(n) * zarf
    ir = ir / ir.abs().max()
    islenmis = torchaudio.functional.fftconvolve(dalga.unsqueeze(0), ir.unsqueeze(0), mode="full")[0]
    islenmis = islenmis[:dalga.shape[-1]]
    tepe = dalga.abs().max()
    if islenmis.abs().max() > 1e-8:
        islenmis = islenmis / islenmis.abs().max() * tepe
    return (1 - islak_oran) * dalga + islak_oran * islenmis


def kazanc_degistir(dalga: torch.Tensor, db: float) -> torch.Tensor:
    """Sesin genel yüksekliğini (dB) değiştirir - farklı mikrofon
    hassasiyetlerine karşı sağlamlık için."""
    carpan = 10 ** (db / 20)
    return dalga * carpan


def sessizlikle_birlestir(dalga1: torch.Tensor, dalga2: torch.Tensor,
                           sr: int = 16000, ara_ms: int = 300) -> torch.Tensor:
    """İki dalga formunu, aralarında sessizlik olacak şekilde birleştirir."""
    ara = torch.zeros(int(sr * ara_ms / 1000))
    return torch.cat([dalga1, ara, dalga2])


# ---------------------------------------------------------------------------
# 2. İki-cümle birleştirme için eşleme havuzu
# ---------------------------------------------------------------------------

def esleme_havuzu_olustur(ham_satirlar: list[dict]) -> dict[tuple, list[dict]]:
    """(dal, hedef_terim, ses_profili) anahtarına göre satırları gruplar.
    Sadece bu anahtarı paylaşan satırlar birbiriyle birleştirilebilir - aynı
    hasta sesi + aynı bağlam (bkz. proje notları)."""
    havuz: dict[tuple, list[dict]] = defaultdict(list)
    for satir in ham_satirlar:
        anahtar = (satir["dal"], satir["hedef_terim"], satir["ses_profili"])
        havuz[anahtar].append(satir)
    return havuz


# ---------------------------------------------------------------------------
# 3. set_transform fonksiyonları
# ---------------------------------------------------------------------------

def egitim_donusumu_olustur(processor, havuz, gurultu_p=0.45, reverb_p=0.25,
                             birlestir_p=0.0, kazanc_p=0.5):
    """TRAIN split için: her erişimde rastgele augmentation uygulayan
    set_transform fonksiyonu döndürür.

    birlestir_p varsayılan 0.0 (kapalı): iki-cümle birleştirme etiket
    uzunluğunu ~%20 ihtimalle ~2 katına çıkarıyor, bu genişlikte şekil
    çeşitliliği MPS'te (Apple GPU) bellek parçalanmasına/büyümesine yol
    açtığı tanı testiyle doğrulandı (adım süresi 5s'den 53s'ye çıkıyordu,
    kapatınca 4.2s'de sabitlendi). Gürültü/reverb/kazanç sesin uzunluğuna
    dokunmadığı için bu sorunu yaşatmıyor, açık kalabilirler."""

    def donusum(batch: dict[str, list]) -> dict[str, list]:
        cikti = {"input_features": [], "labels": []}
        n = len(batch["audio_path"])
        for i in range(n):
            dalga, sr = torchaudio.load(batch["audio_path"][i])
            dalga = dalga[0]
            metin = batch["text"][i]

            if random.random() < gurultu_p:
                dalga = gurultu_ekle(dalga, snr_db=random.uniform(5, 20))
            if random.random() < reverb_p:
                dalga = reverb_uygula(dalga, sr)
            if random.random() < birlestir_p:
                anahtar = (batch["dal"][i], batch["hedef_terim"][i], batch["ses_profili"][i])
                adaylar = [s for s in havuz.get(anahtar, [])
                           if s["audio_path"] != batch["audio_path"][i]]
                if adaylar:
                    esli = random.choice(adaylar)
                    esli_dalga, _ = torchaudio.load(esli["audio_path"])
                    dalga = sessizlikle_birlestir(dalga, esli_dalga[0], sr)
                    metin = metin + " " + esli["text"]
            if random.random() < kazanc_p:
                dalga = kazanc_degistir(dalga, db=random.uniform(-6, 6))

            ozellikler = processor.feature_extractor(dalga.numpy(), sampling_rate=sr).input_features[0]
            cikti["input_features"].append(ozellikler)
            cikti["labels"].append(processor.tokenizer(metin).input_ids)
        return cikti

    return donusum


def degerlendirme_donusumu_olustur(processor):
    """VAL/TEST split için: hiçbir bozulma uygulamayan, hep aynı sonucu
    üreten sabit set_transform fonksiyonu döndürür."""

    def donusum(batch: dict[str, list]) -> dict[str, list]:
        cikti = {"input_features": [], "labels": []}
        n = len(batch["audio_path"])
        for i in range(n):
            dalga, sr = torchaudio.load(batch["audio_path"][i])
            ozellikler = processor.feature_extractor(dalga[0].numpy(), sampling_rate=sr).input_features[0]
            cikti["input_features"].append(ozellikler)
            cikti["labels"].append(processor.tokenizer(batch["text"][i]).input_ids)
        return cikti

    return donusum


# ---------------------------------------------------------------------------
# 4. Batch padding (gerçek data collator - Trainer'a verilecek olan)
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, ozellikler: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # input_features her zaman sabit (80, 3000) sekilde - dogrudan yigabiliriz
        girdi_ozellikleri = [{"input_features": o["input_features"]} for o in ozellikler]
        batch = self.processor.feature_extractor.pad(girdi_ozellikleri, return_tensors="pt")

        # labels degisken uzunlukta - tokenizer.pad ile doldurup, doldurulan
        # kisimlari -100 yapiyoruz ki loss hesabinda goz ardi edilsin
        label_ozellikleri = [{"input_ids": o["labels"]} for o in ozellikler]
        label_batch = self.processor.tokenizer.pad(label_ozellikleri, return_tensors="pt")
        labels = label_batch["input_ids"].masked_fill(label_batch.attention_mask.ne(1), -100)

        # Elle kaydirma/cikarma YAPILMIYOR: WhisperForConditionalGeneration.forward()
        # labels verildiginde shift_tokens_right()'i kendi icinde otomatik cagirir -
        # hem kaydirmayi hem -100 -> pad_token_id donusumunu orada yapiyor
        # (dogrulandi: transformers/models/whisper/modeling_whisper.py kaynagi).
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path

    from datasets import load_dataset
    from transformers import WhisperProcessor

    KOK = Path(__file__).parent.parent
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="turkish", task="transcribe")

    veri = load_dataset("json", data_files={
        "train": str(KOK / "veri_uretimi/cikti/acil_tip_train.jsonl"),
        "val": str(KOK / "veri_uretimi/cikti/acil_tip_val.jsonl"),
    })

    ham_train_satirlar = [json.loads(l) for l in open(KOK / "veri_uretimi/cikti/acil_tip_train.jsonl", encoding="utf-8")]
    havuz = esleme_havuzu_olustur(ham_train_satirlar)
    print(f"Esleme havuzu: {len(havuz)} grup")

    veri["train"].set_transform(egitim_donusumu_olustur(processor, havuz))
    veri["val"].set_transform(degerlendirme_donusumu_olustur(processor))

    print("\n--- Ayni train ornegine iki kez erisim (rastgeleligi kanitlamak icin) ---")
    ornek1 = veri["train"][3]
    ornek2 = veri["train"][3]
    esit_mi = (ornek1["input_features"] == ornek2["input_features"])
    print("input_features tamamen ayni mi:", esit_mi if isinstance(esit_mi, bool) else "farkli (beklenen)")
    print("Etiket1 uzunluk:", len(ornek1["labels"]), "| Etiket2 uzunluk:", len(ornek2["labels"]))
    print("Etiket1 metin:", processor.tokenizer.decode(ornek1["labels"], skip_special_tokens=True))
    print("Etiket2 metin:", processor.tokenizer.decode(ornek2["labels"], skip_special_tokens=True))

    print("\n--- Ayni val ornegine iki kez erisim (hep ayni olmali) ---")
    v1 = veri["val"][3]
    v2 = veri["val"][3]
    import numpy as np
    print("input_features birebir ayni mi:", np.allclose(v1["input_features"], v2["input_features"]))

    print("\n--- Batch padding testi (data collator) ---")
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    ornekler_liste = [veri["train"][i] for i in range(4)]
    batch = collator(ornekler_liste)
    print("input_features batch sekli:", batch["input_features"].shape)
    print("labels batch sekli:", batch["labels"].shape)
    print("labels ilk satir (ilk 10 token, -100'ler padding):", batch["labels"][0][:10].tolist())
