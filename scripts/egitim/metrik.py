"""
WER (Word Error Rate - kelime hata orani) hesaplama.

WER = (ekleme + silme + degistirme) / referans_kelime_sayisi
Trainer'in predict_with_generate=True ile urettigi token ID dizilerini
metne cevirip jiwer ile karsilastirir.
"""

import evaluate

wer_metrigi = evaluate.load("wer")


def compute_metrics_olustur(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()

        # -100 (loss'ta gormezden gelinen dolgu) -> pad_token_id, aksi halde
        # tokenizer.decode -100'u gecersiz token ID olarak gorup hata verir
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metrigi.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


if __name__ == "__main__":
    # Elle dogrulama: bilinen bir referans/tahmin cifti icin WER'i elle hesaplayip
    # jiwer'in sonucuyla karsilastir
    referans = "Zanaks içtim ama içim yanıyor"
    tahmin_dogru = "Zanaks içtim ama içim yanıyor"
    tahmin_1hata = "Zanaks içtim ama içim yanıyordu"   # son kelime yanlis (degistirme)
    tahmin_2hata = "Zanaks içtim içim yanıyor"          # "ama" silinmis

    for isim, tahmin in [("birebir dogru", tahmin_dogru),
                          ("1 kelime hatali", tahmin_1hata),
                          ("1 kelime eksik", tahmin_2hata)]:
        wer = wer_metrigi.compute(predictions=[tahmin], references=[referans])
        ref_kelime_sayisi = len(referans.split())
        print(f"{isim}: WER={wer:.4f} (referans {ref_kelime_sayisi} kelime, "
              f"beklenen ~{1/ref_kelime_sayisi:.4f} ya da 0)")
