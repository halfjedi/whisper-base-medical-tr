"""
LM Studio üzerinden çalışan bir modelle (varsayılan: dolphin3.0-llama3.1-8b),
Whisper fine-tune için sentetik hasta cümleleri üreten toplu (batch) script.

Bu script gerçek zamanlı bir bileşen DEĞİLDİR — konusma_pipeline_denemesi.py'deki asistan
pipeline'ının aksine burada latency önemli değildir (offline/arka planda
çalışır), bu yüzden hız için seçilmiş küçük modeller yerine kalite/çeşitlilik
için daha büyük model (8B) kullanılması önerilir.

Ne üretir:
    Her satırı aşağıdaki formatta olan bir JSONL dosyası
    (veri_uretimi/cikti/hasta_cumleleri.jsonl):

        {
          "dal": "kardiyoloji",
          "socrates_asama": "character",
          "hedef_terim": "göğüs ağrısı",
          "hedef_kelimeler": ["göğüs ağrısı"],
          "hasta_cumlesi": "Göğsümde sanki üzerime bir şey oturmuş gibi bir baskı var.",
          "tekrar_no": 0,
          "persona": "24 yaşında kadın",
          "dusuk_cesitlilik": false
        }

Çeşitlilik güvencesi:
    Aynı (dal, aşama, terim) için üretilen "tekrar" sayısı kadar cümle, farklı
    hasta personalarına (terimler.PERSONAS) zorunlu olarak bağlanır ve her
    yeni cümle, aynı kombinasyondaki önceki cümlelerle SequenceMatcher ile
    karşılaştırılır (BENZERLIK_ESIGI=0.82). Çok benzerse farklı persona/daha
    yüksek sıcaklıkla yeniden denenir (MAX_CESITLILIK_DENEME kez). Tüm
    denemeler tükenirse satır yine de kaydedilir ama "dusuk_cesitlilik": true
    ile işaretlenir — bu satırlar eğitim öncesi filtrelenip incelenebilir.

Devam ettirilebilirlik:
    Script her satırı ürettikçe hemen diske yazar (append) ve başlangıçta
    var olan çıktı dosyasını okuyup hangi (dal, aşama, terim, tekrar_no)
    kombinasyonlarının zaten üretildiğini bir set'e alır. Yarıda kesilen bir
    çalıştırma, `python metin_uret.py` ile tekrar başlatıldığında kaldığı
    yerden devam eder.

Kullanım:
    source venv/bin/activate
    python veri_uretimi/metin_uret.py                     # tam çalıştırma
    python veri_uretimi/metin_uret.py --tekrar 5 --limit 50   # küçük test
    python veri_uretimi/metin_uret.py --dal kardiyoloji    # tek dal
    python veri_uretimi/metin_uret.py --sadece-ilac        # sadece ilaç isimleri
"""

import argparse
import json
import random
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from terimler import BRANCHES, COMMON_VOCAB, DRUGS, PERSONAS, SOCRATES_STAGES, persona_sec, tum_dal_terimleri

# --------------------------------------------------------------------------
# Ayarlar
# --------------------------------------------------------------------------

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LLM_MODEL = "qwen2.5-7b-instruct"  # dolphin3.0-llama3.1-8b denendi, ~%40 hata oranı (uydurma
# kelime, gramer bozukluğu) çıktı - TurkBench ölçümüne göre bu, tabanı olan Llama-3.1-8B-Instruct'ın
# Türkçe'de zayıf olmasından (%45.7 skor) kaynaklanıyor olabilir. Qwen2.5-7B-Instruct aynı
# benchmarkta %54.9 skorla belirgin daha iyi, hem RTX'in 12GB VRAM'ine hem Mac'e rahat sığıyor.
# NOT: LM Studio'da modelin gerçek adı farklıysa --model parametresiyle üzerine yazılabilir:
# python metin_uret.py --model "<lm-studio'daki-tam-ad>"

OUT_DIR = Path(__file__).parent / "cikti"
OUT_FILE = OUT_DIR / "hasta_cumleleri.jsonl"

DEFAULT_TEKRAR = 5  # dal-spesifik terimler + ilaç isimleri için
DEFAULT_TEKRAR_ORTAK = 8  # "ortak" kategorisi (ne zamandır, aniden başladı, vb.) - ÖNEMLİ DÜZELTME:
                          # başta "base Whisper'ın genel Türkçe eğitiminde zaten iyi" varsayımıyla
                          # 2'ye düşürülmüştü. Ampirik test bunu çürüttü: base model "ağrı" gibi
                          # Türkçe'nin en temel kelimelerini bile tutarlı şekilde yanlış tanıyor
                          # (bkz. genel_*.wav testleri - TTS net ama Whisper hatalı). Genel Türkçe
                          # akıcılığı da fine-tune'un gerçek bir hedefi olduğu için 5'in ÜZERİNE
                          # (8'e) çıkarıldı.
REQUEST_TIMEOUT = 60
MAX_RETRY = 3

# Aynı (dal, aşama, terim) için üretilen cümleler birbirine bu eşiğin üzerinde
# benziyorsa (SequenceMatcher oranı), çeşitlilik yetersiz sayılır ve yeniden
# üretilir (overfitting riskini azaltmak için).
BENZERLIK_ESIGI = 0.82
MAX_CESITLILIK_DENEME = 4

# Kullanıcı tarafından onaylanmış, gerçekten doğal duyulan örnek cümleler
# (bkz. veri_uretimi/dogal_ornekler_taslak.md) - modele soyut "halk ağzı
# kullan" talimatı yerine somut örnek göstermek için (few-shot). Amaç:
# üretilen cümlelerin "Bayılmam var, doktor" gibi çeviri-kokan yapay
# kalıplar yerine gerçek konuşma diline benzemesi.
DOGAL_ORNEKLER = [
    "Nefesim daralıyor bazen, özellikle yürürken.",
    "Sabaha karşı başladı, gece yarısı falan uyandım öyle.",
    "Aniden oldu, hiç belli etmeden geldi.",
    "Üç gündür bu böyle, bir türlü geçmiyor.",
    "Karnım öyle bir ağrıyor ki dayanamıyorum.",
    "Boğuluyormuşum gibi hissediyorum, hava alamıyorum sanki.",
    "Göğsümde bir sıkışma var, taş basmış gibi.",
    "Karnımda bıçak saplanıyormuş gibi bir ağrı var.",
    "Ağrı sırtıma da vuruyor.",
    "Sol kolum da tutuluyor bu arada.",
    "Bir de midem bulanıyor, kusacak gibiyim.",
    "Baş dönmesi tuttu beni, duramıyorum ayakta.",
    "Kalbim küt küt atıyor, çok hızlı.",
    "Terliyorum bir yandan da.",
    "İki gündür ateşim düşmüyor.",
    "Ara ara oluyor, geçip geliyor.",
    "Gittikçe kötüleşiyor, ilk günkü gibi değil.",
    "Yürüyünce daha kötü oluyor, otururken biraz rahatlıyorum.",
    "Yemek yiyince artıyor sanki.",
    "Parol içtim ama fayda etmedi.",
    "Hiç böyle bir şey yaşamamıştım, çok korkutucu.",
    "Dayanılır gibi değil, iş gücüm kalmadı.",
    "Hafif hafif oluyor, günlük işimi engellemiyor.",
    "Doktorum Augmentin yazmıştı, üç gündür kullanıyorum.",
    "Xanax'ı gece alıyorum, uyumak için.",
    "Nurofen içtim ama ağrı geçmedi.",
]

SYSTEM_PROMPT = (
    "Sen, gerçek hastaların doktor muayenesi öncesi bir asistana anlattığı "
    "şikayet cümlelerini üreten bir yardımcı yapay zekasın. Görevin, verilen "
    "tıbbi terimi/ifadeyi HALK AĞZIYLA, sokaktaki sıradan bir hastanın "
    "konuşacağı gibi TEK BİR KISA Türkçe cümle içine yerleştirmek.\n\n"
    "ÇIKTI FORMATI - ÇOK ÖNEMLİ:\n"
    "SADECE ve SADECE hasta cümlesini yaz. Açıklama, gerekçe, giriş, tırnak "
    "işareti, İngilizce kelime, kendi kendine yorum EKLEME. İlk kelimeden "
    "itibaren doğrudan hasta konuşuyormuş gibi başla.\n\n"
    "DİL KURALLARI:\n"
    "- HALK AĞZI: sıradan bir insanın günlük konuşması gibi, kısa ve net.\n"
    "- \"nezaket ediyorum\", \"maruz kalıyorum\" gibi resmi/yazınsal kalıplar YASAK.\n"
    "- ÇOĞU cümlede (yaklaşık %70'inde) HİÇBİR dolgu/ünlem kelime KULLANMA - "
    "direkt şikayeti anlat. Sadece bazen (yaklaşık %30) istersen doğal bir "
    "dolgu ekleyebilirsin ama HER SEFERİNDE FARKLI bir tanesini seç, tek bir "
    "kelimeyi (örn. hep aynı ünlemi) tekrar tekrar kullanmak YASAK.\n"
    "- SADECE GERÇEK VE DOĞRU TÜRKÇE KELİMELER kullan. Uydurma kelime, yanlış "
    "çekim, anlamsız/yarım cümle YASAK.\n"
    "- TEK VE BASİT bir cümle kur. Virgülle bağlanan 3-4 farklı fikri art arda "
    "sıralama, bu anlamsız/dağınık cümlelere yol açıyor. En fazla 1 virgül/bağlaç "
    "kullan (örn. \"...var, bir de ateşim çıktı\" gibi TEK bir ek bilgi yeterli).\n"
    "- Cümle en fazla 12-15 kelime olsun.\n"
    "- BİRİNCİ AĞIZDAN yaz (\"benim\", \"var\", \"hissediyorum\" gibi). Hastanın "
    "adını, yaşını ya da \"... yaşında bir kadın\" gibi üçüncü şahıs ifadeleri "
    "cümlenin içine YAZMA - yaş/cinsiyet sadece üslubu belirlesin, cümlede "
    "geçmesin.\n\n"
    "- Hasta farklı yaş, cinsiyet ve anlatım tarzlarında olabilir; çeşitlilik önemli.\n"
    "- Kullanıcı mesajındaki 'hedef terim/kelime' talimatına uy: bazen birebir "
    "geçmesi ZORUNLU (ilaç isimlerinde), bazen sadece konuyla ilgili olman "
    "yeterli (talimatta açıkça belirtilecek).\n"
    "- Asla tanı koyma, tedavi önerme; sadece hastanın kendi ifadesini üret.\n\n"
    "AŞAĞIDAKİ ÖRNEKLER GERÇEKTEN DOĞAL, BÖYLE BİR TARZDA YAZ (birebir kopyalama, "
    "sadece üslubu/doğallığı taklit et):\n"
    + "\n".join(f"- {ornek}" for ornek in DOGAL_ORNEKLER)
)

random.seed(42)


def build_user_prompt(dal: str, asama_key: str, terim: str, persona: dict) -> str:
    asama_aciklama = SOCRATES_STAGES[asama_key]

    if dal == "ilac":
        terim_talimati = (
            f"Hedef ilaç/kelime (cümlede BİREBİR, marka adıyla mutlaka geçmeli): \"{terim}\"\n\n"
            "Bu hastanın ağzından, yukarıdaki anlatım türüne uygun, ilacın adını "
            "AÇIKÇA ve birebir söyleyen TEK bir Türkçe cümle yaz (gerçek hastalar "
            "ilaç isimlerini genelde birebir söyler, örn. \"Parol içtim\")."
        )
    else:
        terim_talimati = (
            f"Konu/şikayet (bu KONUYLA İLGİLİ bir cümle yaz, ama kelimesi kelimesine "
            f"tekrar etmen ŞART DEĞİL): \"{terim}\"\n\n"
            "Bu hastanın ağzından, yukarıdaki anlatım türüne uygun, bu şikayet "
            "konusuyla ilgili TEK bir Türkçe cümle yaz. Gerçek hastalar çoğu zaman "
            "klinik terimi birebir söylemez, günlük tabirle anlatır (örn. \"nefes "
            "darlığı\" yerine \"nefesim daralıyor\" gibi) - böyle DOĞAL bir "
            "paraphrase tamamen tercih edilir, klinik terimi zorla sokma."
        )

    return (
        f"Poliklinik: {dal}\n"
        f"İstenen anlatım türü: {asama_aciklama}\n"
        f"{terim_talimati}\n"
        f"Hasta profili: {persona['yas']}, {persona['cinsiyet']}, "
        f"{persona['uslup']} biri.\n\n"
        "Cümle bu hasta profiline uygun kelime seçimi ve üslupla yazılsın "
        "(ama profili cümlenin içine açıkça yazma, sadece üslubu yansıt)."
    )


def cok_benzer_mi(aday: str, mevcutlar: list[str]) -> bool:
    aday_norm = aday.lower().strip()
    for mevcut in mevcutlar:
        oran = SequenceMatcher(None, aday_norm, mevcut.lower().strip()).ratio()
        if oran >= BENZERLIK_ESIGI:
            return True
    return False


def call_llm(dal: str, asama_key: str, terim: str, persona: dict, temperature: float = 1.0) -> str | None:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(dal, asama_key, terim, persona)},
        ],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": 80,
        # Qwen ChatML formatı: stop token'ları tanımlanmazsa model bazen kendi
        # turn'ünü bitirmeyip hayali bir "user" mesajı üretmeye devam ediyor
        # (gözlemlendi: <|im_start|>user... sızıntısı + Çince'ye kayma).
        "stop": ["<|im_start|>", "<|im_end|>", "\n\n"],
    }
    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = requests.post(
                f"{LM_STUDIO_BASE_URL}/chat/completions",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            text = text.strip('"').strip()
            if text:
                return text
        except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"  [uyarı] deneme {attempt}/{MAX_RETRY} başarısız ({dal}/{terim}): {e}")
            time.sleep(2 * attempt)
    return None


def cumle_uret_cesitli(dal: str, asama_key: str, terim: str, tekrar_no: int, mevcut_cumleler: list[str]) -> tuple[str | None, bool]:
    """Persona'ya göre cümle üretir; aynı kombinasyondaki önceki cümlelere çok
    benzerse (BENZERLIK_ESIGI üzeri) farklı persona/sıcaklıkla yeniden dener.
    Dönüş: (cümle, dusuk_cesitlilik_mi)."""
    persona = persona_sec(tekrar_no)
    for deneme in range(MAX_CESITLILIK_DENEME):
        # her denemede farklı bir persona ve biraz artan sıcaklık dene
        deneme_persona = PERSONAS[(tekrar_no + deneme) % len(PERSONAS)]
        temperature = min(1.0 + 0.15 * deneme, 1.3)
        cumle = call_llm(dal, asama_key, terim, deneme_persona, temperature=temperature)
        if cumle is None:
            continue
        if not cok_benzer_mi(cumle, mevcut_cumleler):
            return cumle, False
    # tüm denemeler tükendi, en son üretilen (varsa) çeşitlilik uyarısıyla kabul edilir
    if cumle is None:
        return None, False
    return cumle, True


def load_done_keys() -> tuple[set[tuple[str, str, str, int]], dict[tuple[str, str, str], list[str]]]:
    """(dal,asama,terim,tekrar_no) tekillik seti + (dal,asama,terim) -> üretilmiş
    cümleler sözlüğünü döner. İkincisi, resume sonrası da benzerlik kontrolünün
    önceki çalıştırmalardaki cümleleri hesaba katması için gerekli."""
    done = set()
    combo_sentences: dict[tuple[str, str, str], list[str]] = {}
    if not OUT_FILE.exists():
        return done, combo_sentences
    with open(OUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            dal, asama, terim = row.get("dal"), row.get("socrates_asama"), row.get("hedef_terim")
            key = (dal, asama, terim, row.get("tekrar_no", 0))
            done.add(key)
            cumle = row.get("hasta_cumlesi")
            if cumle:
                combo_sentences.setdefault((dal, asama, terim), []).append(cumle)
    return done, combo_sentences


def append_row(row: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_task_list(dal_filter: str | None, sadece_ilac: bool) -> list[tuple[str, str, str]]:
    tasks: list[tuple[str, str, str]] = []
    asama_keys = list(SOCRATES_STAGES.keys())

    if sadece_ilac:
        terim_kaynagi = [("ilac", ilac) for ilac in DRUGS]
    elif dal_filter:
        if dal_filter not in BRANCHES:
            raise SystemExit(f"Bilinmeyen dal: {dal_filter}. Geçerli dallar: {list(BRANCHES.keys())}")
        terim_kaynagi = [(dal_filter, t) for t in BRANCHES[dal_filter]]
    else:
        terim_kaynagi = list(tum_dal_terimleri())
        terim_kaynagi += [("ortak", t) for t in COMMON_VOCAB]
        terim_kaynagi += [("ilac", t) for t in DRUGS]

    for dal, terim in terim_kaynagi:
        for asama_key in asama_keys:
            tasks.append((dal, asama_key, terim))

    random.shuffle(tasks)
    return tasks


def main():
    global LLM_MODEL

    parser = argparse.ArgumentParser(description="LM Studio ile sentetik hasta cümlesi üretimi")
    parser.add_argument("--tekrar", type=int, default=DEFAULT_TEKRAR, help="dal-spesifik ve ilaç terimleri için tekrar sayısı")
    parser.add_argument("--tekrar-ortak", type=int, default=DEFAULT_TEKRAR_ORTAK,
                         help="'ortak' kategorisi için tekrar sayısı (düşük tutulur: bu ifadeler zaten "
                              "base Whisper'ın genel Türkçe eğitiminde iyi temsil ediliyor, ilaç/terim "
                              "çeşitliliğine öncelik veriyoruz)")
    parser.add_argument("--limit", type=int, default=None, help="toplam üretilecek satır sayısını sınırla (test için)")
    parser.add_argument("--dal", type=str, default=None, help="sadece belirli bir dalı üret")
    parser.add_argument("--sadece-ilac", action="store_true", help="sadece ilaç isimleri kategorisini üret")
    parser.add_argument("--model", type=str, default=LLM_MODEL, help="LM Studio'da yüklü model adı")
    args = parser.parse_args()

    LLM_MODEL = args.model

    tasks = build_task_list(args.dal, args.sadece_ilac)
    done, combo_sentences = load_done_keys()

    def tekrar_sayisi(dal: str) -> int:
        return args.tekrar_ortak if dal == "ortak" else args.tekrar

    toplam_hedef_cumle = sum(tekrar_sayisi(dal) for dal, _, _ in tasks)

    print(f"Model: {LLM_MODEL}")
    print(f"Toplam (dal, aşama, terim) kombinasyonu: {len(tasks)}")
    print(f"Tekrar: dal/ilaç={args.tekrar}, ortak={args.tekrar_ortak}")
    print(f"Hedef toplam cümle: {toplam_hedef_cumle}")
    print(f"Zaten üretilmiş satır sayısı: {len(done)}")
    print(f"Çıktı dosyası: {OUT_FILE}")

    produced_this_run = 0
    dusuk_cesitlilik_sayisi = 0
    total_target = toplam_hedef_cumle if args.limit is None else args.limit

    for dal, asama_key, terim in tasks:
        combo_key = (dal, asama_key, terim)
        for tekrar_no in range(tekrar_sayisi(dal)):
            if args.limit is not None and produced_this_run >= args.limit:
                print(f"\nLimit ({args.limit}) ulaşıldı, durduruluyor.")
                break

            key = (dal, asama_key, terim, tekrar_no)
            if key in done:
                continue

            mevcut_cumleler = combo_sentences.setdefault(combo_key, [])
            cumle, dusuk_cesitlilik = cumle_uret_cesitli(dal, asama_key, terim, tekrar_no, mevcut_cumleler)
            if cumle is None:
                print(f"  [atlandı] {dal}/{asama_key}/{terim} (tekrar {tekrar_no}) - LLM yanıt vermedi")
                continue

            if dusuk_cesitlilik:
                dusuk_cesitlilik_sayisi += 1

            persona = PERSONAS[tekrar_no % len(PERSONAS)]
            row = {
                "dal": dal,
                "socrates_asama": asama_key,
                "hedef_terim": terim,
                "hedef_kelimeler": [terim],
                "hasta_cumlesi": cumle,
                "tekrar_no": tekrar_no,
                "persona": f"{persona['yas']} {persona['cinsiyet']}",
                "dusuk_cesitlilik": dusuk_cesitlilik,
            }
            append_row(row)
            mevcut_cumleler.append(cumle)
            produced_this_run += 1

            if produced_this_run % 10 == 0:
                print(f"  [{produced_this_run}/{total_target}] {dal}/{asama_key}/{terim}: \"{cumle}\"")
        else:
            continue
        break

    print(f"\nTamamlandı. Bu çalıştırmada üretilen: {produced_this_run} satır.")
    if dusuk_cesitlilik_sayisi:
        print(f"UYARI: {dusuk_cesitlilik_sayisi} satır, {MAX_CESITLILIK_DENEME} denemeye rağmen "
              f"benzerlik eşiğinin altına inemedi (dusuk_cesitlilik=true olarak işaretlendi).")
    print(f"Toplam dosya: {OUT_FILE}")


if __name__ == "__main__":
    main()
