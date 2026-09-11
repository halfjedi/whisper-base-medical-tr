"""
Claude'un ürettiği cümleleri (claude_chunk_yanit.txt, "N) cümle" formatında)
claude_kuyruk.jsonl'un EN BAŞINDAKİ N görevle eşleştirir, kalite filtresinden
geçirir, kabul edilenleri hasta_cumleleri.jsonl'a ekler, elenenleri
hard_filtre_basarisiz_claude.jsonl'a yazar (kaybolmaz, sonra elle düzeltilip
tekrar denenebilir). En önemlisi: işlenen N görevi (kabul/red fark etmez)
kuyruğun başından SİLER ve dosyayı kalıcı olarak günceller - böylece ilerleme
her çalıştırmada diske yazılmış olur, oturum kesilirse kaldığı yerden gidilir.

Kullanım:
    source venv/bin/activate
    python veri_uretimi/claude_uret_isle.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import kalite_kontrol
import terimler

OUT_DIR = Path(__file__).parent / "cikti"
BATCH_DIR = OUT_DIR / "manuel_batch"
KUYRUK = BATCH_DIR / "claude_kuyruk.jsonl"
YANIT = BATCH_DIR / "claude_chunk_yanit.txt"
HASTA_CUMLELERI = OUT_DIR / "hasta_cumleleri.jsonl"
ELENEN = OUT_DIR / "hard_filtre_basarisiz_claude.jsonl"
ILERLEME_LOG = BATCH_DIR / "claude_ilerleme.log"

SATIR_REGEX = re.compile(r"^\s*(\d+)\s*[).:]\s*(.+?)\s*$")


def yaniti_parse_et(metin: str) -> dict[int, str]:
    sonuc = {}
    for satir in metin.splitlines():
        m = SATIR_REGEX.match(satir)
        if m:
            sonuc[int(m.group(1))] = m.group(2).strip()
    return sonuc


def satir_kur(meta: dict, cumle: str) -> dict:
    persona = terimler.PERSONAS[meta["persona_no"] - 1]
    return {
        "dal": meta["dal"],
        "socrates_asama": meta["socrates_asama"],
        "hedef_terim": meta["hedef_terim"],
        "hedef_kelimeler": [meta["hedef_terim"]],
        "hasta_cumlesi": cumle,
        "tekrar_no": meta["tekrar_no"],
        "persona": f"{persona['yas']} {persona['cinsiyet']}",
        "dusuk_cesitlilik": False,
        "kaynak": "claude",
    }


def main():
    if not KUYRUK.exists():
        raise SystemExit(f"Kuyruk yok: {KUYRUK}")
    if not YANIT.exists():
        raise SystemExit(f"Yanit dosyasi yok: {YANIT}")

    kuyruk_satirlari = [l for l in open(KUYRUK, encoding="utf-8") if l.strip()]
    kuyruk = [json.loads(l) for l in kuyruk_satirlari]

    yanit_metni = YANIT.read_text(encoding="utf-8")
    cumleler = yaniti_parse_et(yanit_metni)
    if not cumleler:
        raise SystemExit("Yanit ayristirilamadi - 'N) cumle' formatinda mi?")

    n = max(cumleler.keys())
    if n > len(kuyruk):
        raise SystemExit(f"Yanitta {n} satir var ama kuyrukta sadece {len(kuyruk)} gorev kaldi")

    islenen = kuyruk[:n]
    kalan = kuyruk[n:]

    gorulen_normalize = set()
    for l in open(HASTA_CUMLELERI, encoding="utf-8"):
        l = l.strip()
        if l:
            c = json.loads(l)["hasta_cumlesi"]
            gorulen_normalize.add(" ".join(c.lower().split()).rstrip(".!?"))

    kabul, elendi, tekrar_eden = 0, 0, 0
    with open(HASTA_CUMLELERI, "a", encoding="utf-8") as f_ok, \
         open(ELENEN, "a", encoding="utf-8") as f_kotu:
        for i, meta in enumerate(islenen, 1):
            cumle = cumleler.get(i)
            if cumle is None:
                # yanitta bu satir yok - kaybetmeyelim, kalan kuyruga geri koy
                kalan.insert(0, meta)
                continue

            norm = " ".join(cumle.lower().split()).rstrip(".!?")
            row = satir_kur(meta, cumle)
            problemler = kalite_kontrol.hard_filtre_uygula(row)
            if norm in gorulen_normalize:
                problemler = list(problemler) + ["birebir_tekrar"]

            if problemler:
                row["_problemler"] = problemler
                f_kotu.write(json.dumps(row, ensure_ascii=False) + "\n")
                elendi += 1
                if "birebir_tekrar" in problemler:
                    tekrar_eden += 1
            else:
                gorulen_normalize.add(norm)
                f_ok.write(json.dumps(row, ensure_ascii=False) + "\n")
                kabul += 1

    # Kuyrugu KALICI olarak guncelle - islenen kisim (kabul/red farketmez) cikti
    with open(KUYRUK, "w", encoding="utf-8") as f:
        for g in kalan:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    with open(ILERLEME_LOG, "a", encoding="utf-8") as f:
        f.write(f"islendi={len(islenen)} kabul={kabul} elendi={elendi} "
                f"(tekrar={tekrar_eden}) kalan_kuyruk={len(kalan)}\n")

    print(f"Islendi: {len(islenen)}  Kabul: {kabul}  Elendi: {elendi} (tekrar={tekrar_eden})")
    print(f"Kalan kuyruk: {len(kalan)} gorev")
    print(f"hasta_cumleleri.jsonl ve claude_kuyruk.jsonl GUNCELLENDI (kalici)")


if __name__ == "__main__":
    main()
