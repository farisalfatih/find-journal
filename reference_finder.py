"""
reference_finder.py
==================
Step 1: Fetch data mentah dari OpenAlex API (TANPA Groq/AI).

Mengambil paper tentang bitcoin (2021-2026) dari OpenAlex,
filter HANYA bahasa Indonesia dan Inggris, lalu menyimpannya
ke 2 file JSON terpisah.

Output:
  - openalex_english.json      (paper bahasa Inggris)
  - openalex_indonesian.json   (paper bahasa Indonesia)

Mendukung RESUME — jika proses terputus, jalankan lagi
dan akan melanjutkan dari cursor terakhir.

Usage:
    # Fetch semua (English + Indonesian)
    python reference_finder.py

    # Fetch hanya English
    python reference_finder.py --lang en

    # Fetch hanya Indonesian
    python reference_finder.py --lang id

    # Reset & fetch ulang
    python reference_finder.py --reset

    # Batasi jumlah per bahasa
    python reference_finder.py --max 50
"""

import os
import json
import time
import argparse
import requests

# ─────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────
BASE_URL = "https://api.openalex.org/works"
FILTER_BASE = (
    "title_and_abstract.search:bitcoin,"
    "has_abstract:true,"
    "publication_year:2021-2026"
)
SELECT = "display_name,doi,abstract_inverted_index,publication_year,authorships"
PER_PAGE = 200
OUTPUT_DIR = "output"

LANGUAGE_CONFIG = {
    "en": {
        "label": "English",
        "output": os.path.join(OUTPUT_DIR, "openalex_english.json"),
        "progress": os.path.join(OUTPUT_DIR, "openalex_fetch_progress_en.json"),
    },
    "id": {
        "label": "Indonesian",
        "output": os.path.join(OUTPUT_DIR, "openalex_indonesian.json"),
        "progress": os.path.join(OUTPUT_DIR, "openalex_fetch_progress_id.json"),
    },
}


# ─────────────────────────────────────────
# HELPER: Decode abstract_inverted_index
# ─────────────────────────────────────────
def decode_abstract(inverted_index: dict) -> str:
    if not inverted_index:
        return ""
    try:
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in word_positions)
    except Exception:
        return ""


# ─────────────────────────────────────────
# PROGRESS MANAGEMENT
# ─────────────────────────────────────────
def save_progress(progress: dict, progress_file: str):
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def load_progress(progress_file: str) -> dict | None:
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_results(results: list, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_results(output_file: str) -> list:
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ─────────────────────────────────────────
# FETCH LOGIC (per language)
# ─────────────────────────────────────────
def fetch_language(lang_code: str, reset: bool = False, max_results: int | None = None):
    """
    Fetch data untuk satu bahasa tertentu dari OpenAlex.
    Mendukung resume via cursor-based pagination.
    """
    config = LANGUAGE_CONFIG[lang_code]
    output_file = config["output"]
    progress_file = config["progress"]
    lang_label = config["label"]

    url_filter = f"{FILTER_BASE},language:{lang_code}"

    print("\n" + "=" * 65)
    print(f"  FETCH DATA — Bahasa: {lang_label} ({lang_code})")
    print("=" * 65)
    print(f"  Filter     : bitcoin | 2021-2026 | has_abstract:true | {lang_code}")
    print(f"  Output     : {output_file}")
    print("=" * 65)

    # --- Resume check ---
    if reset:
        print(f"\n[RESET] Menghapus progress & data lama ({lang_label})...")
        for f_path in [progress_file, output_file]:
            if os.path.exists(f_path):
                os.remove(f_path)
                print(f"  Dihapus: {f_path}")
        cursor = "*"
        results = []
        total_available = None
    else:
        progress = load_progress(progress_file)
        results = load_results(output_file)

        if progress and results:
            cursor = progress.get("next_cursor")
            total_available = progress.get("total_available")
            print(f"\n[RESUME] {lang_label}: {len(results):,} data tersimpan.")
            print(f"  Melanjutkan dari cursor...")
            if total_available:
                pct = len(results) / total_available * 100
                print(f"  Progress: {len(results):,}/{total_available:,} ({pct:.1f}%)")
        else:
            cursor = "*"
            results = []
            total_available = None
            print(f"\n[START] Memulai fetch {lang_label} dari awal...")

    # --- Early exit jika sudah cukup ---
    if max_results is not None and len(results) >= max_results:
        print(f"\n  Sudah mencapai batas max ({max_results}). Skip.")
        return results

    # --- Fetch loop ---
    batch_count = 0
    consecutive_empty = 0

    while True:
        if max_results is not None and len(results) >= max_results:
            print(f"\n  Mencapai batas max ({max_results}). Berhenti.")
            break

        batch_count += 1

        try:
            params = {
                "filter": url_filter,
                "select": SELECT,
                "per-page": PER_PAGE,
                "cursor": cursor,
            }
            resp = requests.get(BASE_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"\n  [ERROR] Request gagal: {e}")
            print(f"  Progress tersimpan. Jalankan lagi untuk resume.")
            save_progress({
                "next_cursor": cursor,
                "total_available": total_available,
                "last_count": len(results),
                "language": lang_code,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, progress_file)
            save_results(results, output_file)
            return results

        meta = data.get("meta", {})
        if total_available is None:
            total_available = meta.get("count", 0)
            print(f"\n  Total {lang_label} di OpenAlex: {total_available:,} paper")
            print(f"  Memulai pengambilan...\n")

        batch_results = data.get("results", [])

        if not batch_results:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                print(f"  3x batch kosong berturut-turut. Selesai.")
                break
            print(f"  Batch {batch_count}: kosong (retry {consecutive_empty}/3)")
            time.sleep(1)
            continue
        else:
            consecutive_empty = 0

        # Parse & decode
        for item in batch_results:
            if max_results is not None and len(results) >= max_results:
                break
            abstract = decode_abstract(item.get("abstract_inverted_index", {}))
            doi_raw = item.get("doi", "")
            doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None

            penulis = []
            for auth in item.get("authorships", []):
                author_name = auth.get("author", {}).get("display_name", "")
                if author_name:
                    penulis.append(author_name)

            results.append({
                "title": item.get("display_name", ""),
                "doi": doi,
                "publication_year": item.get("publication_year"),
                "penulis": penulis,
                "abstract": abstract,
                "language": lang_code,
                "_enriched": False
            })

        # Progress report
        pct = len(results) / total_available * 100 if total_available else 0
        print(f"  Batch {batch_count}: +{len(batch_results):>3} | "
              f"Total: {len(results):>6,}/{total_available:>6,} ({pct:>5.1f}%)")

        # Save progress
        next_cursor = meta.get("next_cursor")
        if next_cursor:
            cursor = next_cursor
            save_progress({
                "next_cursor": cursor,
                "total_available": total_available,
                "last_count": len(results),
                "language": lang_code,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, progress_file)
            save_results(results, output_file)
        else:
            print(f"\n  Semua halaman sudah diambil.")
            break

        time.sleep(0.5)

    # --- Final save ---
    save_results(results, output_file)
    if os.path.exists(progress_file):
        os.remove(progress_file)

    # --- Summary ---
    print(f"\n{'='*65}")
    print(f"  {lang_label.upper()} — FETCH SELESAI!")
    print(f"{'='*65}")
    print(f"  Total paper   : {len(results):,}")
    years = {}
    for r in results:
        y = str(r.get("publication_year", "?"))
        years[y] = years.get(y, 0) + 1
    print(f"  Distribusi tahun:")
    for y in sorted(years.keys()):
        print(f"    {y}: {years[y]:>5,}")
    print(f"  Output file   : {output_file}")
    if os.path.exists(output_file):
        print(f"  File size     : {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    print(f"{'='*65}")

    return results


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fetch data OpenAlex — filter bahasa Inggris & Indonesia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python reference_finder.py                 # Fetch English + Indonesian
  python reference_finder.py --lang en       # Hanya English
  python reference_finder.py --lang id       # Hanya Indonesian
  python reference_finder.py --max 50        # Batasi per bahasa
  python reference_finder.py --reset         # Hapus semua, mulai ulang
        """
    )
    parser.add_argument("--lang", type=str, default="all",
                        choices=["all", "en", "id"],
                        help="Bahasa yang di-fetch (default: all)")
    parser.add_argument("--max", type=int, default=None,
                        help="Batas maksimal paper per bahasa (default: semua)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset progress & mulai dari awal")

    args = parser.parse_args()

    print("=" * 65)
    print("  REFERENCE FINDER — OpenAlex (Tanpa AI/Groq)")
    print("=" * 65)
    print(f"  Bahasa     : {'English + Indonesian' if args.lang == 'all' else LANGUAGE_CONFIG[args.lang]['label']}")
    print(f"  Max/bahasa : {args.max if args.max else 'semua'}")
    print(f"  Reset      : {'Ya' if args.reset else 'Tidak'}")
    print("=" * 65)

    # Tentukan bahasa mana yang perlu di-fetch
    langs = ["en", "id"] if args.lang == "all" else [args.lang]

    total_all = 0
    for lang in langs:
        results = fetch_language(lang, reset=args.reset, max_results=args.max)
        total_all += len(results)

    # --- Final Summary ---
    print(f"\n{'='*65}")
    print(f"  KESULURUHAN")
    print(f"{'='*65}")
    print(f"  Total paper (semua bahasa) : {total_all:,}")
    for lang in langs:
        config = LANGUAGE_CONFIG[lang]
        if os.path.exists(config["output"]):
            size = os.path.getsize(config["output"]) / 1024 / 1024
            with open(config["output"], "r") as f:
                count = len(json.load(f))
            print(f"    {config['label']:<15}: {count:>6,} paper  ({size:.1f} MB)")
    print(f"\n  Langkah selanjutnya:")
    print(f"    1. Set GROQ_API_KEY di .env")
    print(f"    2. python enrich_with_groq.py --input openalex_english.json")
    print(f"    3. python enrich_with_groq.py --input openalex_indonesian.json")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
