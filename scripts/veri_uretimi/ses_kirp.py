"""
Chatterbox TTS çıktısının sonunda (bazen başında) beliren, metinde karşılığı
olmayan "drone" / uğultu artefaktını Silero VAD ile tespit edip kırpmak için.

Neden gerekli: Whisper eğitiminde (ses, metin) çifti kullanılıyor. Sesin
sonunda metinde olmayan bir gürültü/uğultu kalırsa, bu etiket-ses uyuşmazlığı
yaratır — model cümle biterken "hayali" bir ses beklemeyi öğrenebilir ya da
gereksiz gürültüye karşı sağlamlaşmak yerine kafası karışabilir. Çözüm:
gerçek konuşmanın başladığı/bittiği yeri Silero VAD ile bulup, dışında kalan
her şeyi (artefakt dahil) kırpmak.

Kullanım (tek dosya):
    source venv/bin/activate
    python ses_kirp.py girdi.wav [cikti.wav]

Başka scriptlerden fonksiyon olarak:
    from ses_kirp import sesi_kirp
    kirpilmis_wav = sesi_kirp(wav_tensor, sample_rate)
"""

import sys
from pathlib import Path

import torch
import torchaudio as ta
from silero_vad import get_speech_timestamps, load_silero_vad

_VAD_MODEL = None
_VAD_SR = 16000  # Silero VAD sadece 8000/16000 Hz destekliyor


def _vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


def sesi_kirp(wav: torch.Tensor, sr: int, pad_ms: int = 80) -> torch.Tensor:
    """wav: (1, N) ya da (N,) tensor. Konuşma dışı baş/son kısmı kırpar.
    Konuşma tespit edilemezse orijinal wav'ı değiştirmeden döner (güvenli varsayılan)."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    mono = wav[0]
    if sr != _VAD_SR:
        analiz_sesi = ta.functional.resample(mono, sr, _VAD_SR)
    else:
        analiz_sesi = mono

    zaman_damgalari = get_speech_timestamps(
        analiz_sesi, _vad_model(), sampling_rate=_VAD_SR, return_seconds=True
    )

    if not zaman_damgalari:
        print("  [ses_kirp uyarı] konuşma tespit edilemedi, kırpma yapılmadı")
        return wav

    baslangic_s = max(0.0, zaman_damgalari[0]["start"] - pad_ms / 1000)
    bitis_s = zaman_damgalari[-1]["end"] + pad_ms / 1000

    baslangic_idx = int(baslangic_s * sr)
    bitis_idx = min(wav.shape[-1], int(bitis_s * sr))

    return wav[:, baslangic_idx:bitis_idx]


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python ses_kirp.py girdi.wav [cikti.wav]")
        sys.exit(1)

    girdi_yolu = Path(sys.argv[1])
    cikti_yolu = Path(sys.argv[2]) if len(sys.argv) > 2 else girdi_yolu.with_stem(girdi_yolu.stem + "_kirpilmis")

    wav, sr = ta.load(str(girdi_yolu))
    orijinal_sure = wav.shape[-1] / sr

    kirpilmis = sesi_kirp(wav, sr)
    yeni_sure = kirpilmis.shape[-1] / sr

    ta.save(str(cikti_yolu), kirpilmis, sr)
    print(f"{girdi_yolu.name}: {orijinal_sure:.2f}s -> {yeni_sure:.2f}s ({orijinal_sure - yeni_sure:.2f}s kırpıldı)")
    print(f"Kaydedildi: {cikti_yolu}")


if __name__ == "__main__":
    main()
