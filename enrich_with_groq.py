"""
enrich_with_groq.py
===================
Step 2: Enrich data paper dari OpenAlex menggunakan Groq AI.

Membaca file JSON hasil dari reference_finder.py, lalu untuk setiap paper
yang belum di-enrich, memanggil Groq AI untuk:
  1. Menentukan cocok untuk bagian skripsi mana
  2. Mengekstrak klaim utama paper
  3. Generate ringkasan dalam bahasa Indonesia

FITUR RESUME:
  - Setiap paper yang sudah diproses langsung disimpan ke file output
  - Jika terkena rate limit (429) atau error, proses bisa dihentikan
  - Saat dijalankan lagi, otomatis melanjutkan dari paper yang belum diproses
  - Tidak ada data yang terduplikasi atau diproses ulang

Output filename otomatis dari input:
  openalex_english.json     → enriched_english.json
  openalex_indonesian.json  → enriched_indonesian.json

Usage:
    # Jalankan untuk English
    python enrich_with_groq.py openalex_english.json

    # Jalankan untuk Indonesian
    python enrich_with_groq.py openalex_indonesian.json

    # Atau jalankan keduanya
    python enrich_with_groq.py openalex_english.json
    python enrich_with_groq.py openalex_indonesian.json

    # Jika rate limit, tunggu lalu jalankan lagi (otomatis resume)
    python enrich_with_groq.py openalex_english.json

    # Atur delay manual
    python enrich_with_groq.py openalex_english.json --delay 5

    # Hanya proses 50 paper
    python enrich_with_groq.py openalex_english.json --max 50
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq
from groq import RateLimitError

load_dotenv()

# ─────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────
GROQ_MODEL = "llama-3.1-8b-instant"
OUTPUT_DIR = "output"
# Input/output akan ditentukan dari nama file input otomatis
# Contoh: openalex_english.json → enriched_english.json
#         openalex_indonesian.json → enriched_indonesian.json
DEFAULT_BATCH_SIZE = 5      # Save ke file setiap N paper berhasil di-enrich
RATE_LIMIT_DELAY = 3.0       # Delay antar request Groq (detik) — aman untuk 30 RPM
MAX_RETRIES = 5              # Retry jika rate limit (429)
RETRY_BASE_DELAY = 65        # Base delay saat retry setelah 429 (detik) — ~menunggu 1 menit window reset


# ─────────────────────────────────────────
# GROQ ENRICHMENT
# ─────────────────────────────────────────
def create_groq_client():
    """Inisialisasi Groq client dari env variable."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[ERROR] GROQ_API_KEY tidak ditemukan!")
        print("  Tambahkan ke file .env:")
        print("    GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx")
        print("  Dapatkan API key di: https://console.groq.com/keys")
        sys.exit(1)
    return Groq(api_key=api_key)


def enrich_paper(client: Groq, title: str, abstract: str) -> dict:
    """
    Gunakan Groq AI untuk menganalisis satu paper.
    Return dict: {cocok_untuk, mengklaim_bahwa, ringkasan_ai}
    
    CATATAN: Error rate limit (429) TIDAK ditangkap di sini —
    biarkan propagate ke call_groq_with_retry() agar retry logic jalan.
    """
    if not abstract or len(abstract) < 30:
        return {
            "cocok_untuk": [],
            "mengklaim_bahwa": [],
            "ringkasan_ai": "Abstract tidak tersedia atau terlalu pendek."
        }

    prompt = f"""Kamu adalah asisten akademik yang membantu mahasiswa skripsi menganalisis paper ilmiah tentang bitcoin/cryptocurrency.

Judul paper: "{title}"

Abstract:
{abstract}

Berikan analisis dalam format JSON yang VALID (tanpa markdown, langsung JSON saja):
{{
  "cocok_untuk": ["pilih bagian skripsi yang paling relevan dari: Pendahuluan, Tinjauan Pustaka, Metodologi, Hasil dan Pembahasan"],
  "mengklaim_bahwa": ["klaim/temuan utama 1 (kalimat pendek bahasa Indonesia)", "klaim 2", "klaim 3"],
  "ringkasan_ai": "ringkasan 2-3 kalimat dalam bahasa Indonesia tentang isi paper ini"
}}

Catatan penting:
- "cocok_untuk": pilih bagian yang PALING relevan, boleh lebih dari satu
- "mengklaim_bahwa": ekstrak temuan/klaim utama, sertakan angka/metode jika ada
- "ringkasan_ai": jelaskan inti paper dalam bahasa Indonesia
- JAWAB HANYA JSON VALID, tidak ada teks lain sebelum atau sesudah"""

    res = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600
    )
    raw = res.choices[0].message.content.strip()

    # Bersihkan markdown code block jika ada
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:])  # Hapus baris pertama (```json)
        if raw.endswith("```"):
            raw = raw[:-3]  # Hapus ``` di akhir
        raw = raw.strip()

    parsed = json.loads(raw)
    return {
        "cocok_untuk": parsed.get("cocok_untuk", []),
        "mengklaim_bahwa": parsed.get("mengklaim_bahwa", []),
        "ringkasan_ai": parsed.get("ringkasan_ai", "")
    }


def call_groq_with_retry(client: Groq, title: str, abstract: str, retries: int = MAX_RETRIES, delay: float = RATE_LIMIT_DELAY) -> tuple[dict | None, bool]:
    """
    Panggil Groq dengan retry logic untuk rate limit (429).
    
    Return: (result, should_stop)
    - result: dict enrichment jika berhasil, None jika gagal
    - should_stop: True jika harus stop total
    
    Logic:
    - 429 / RateLimitError → tunggu ~65 detik, retry
    - JSONDecodeError → return fallback (bukan error fatal)
    - Network error → tunggu 15 detik, retry
    - Error lain → stop total
    """
    for attempt in range(retries):
        try:
            result = enrich_paper(client, title, abstract)
            return result, False

        except RateLimitError as e:
            # Groq SDK RateLimitError — pasti 429
            wait = RETRY_BASE_DELAY + (5 * attempt)
            print(f"\n    [429 RATE LIMIT] attempt {attempt+1}/{retries}")
            print(f"    Menunggu {wait} detik agar window reset...")
            time.sleep(wait)
            continue

        except json.JSONDecodeError:
            # AI response bukan JSON valid — fallback, bukan error fatal
            return {
                "cocok_untuk": ["Tinjauan Pustaka"],
                "mengklaim_bahwa": [],
                "ringkasan_ai": abstract[:300] + "..." if len(abstract) > 300 else abstract
            }, False

        except Exception as e:
            error_str = str(e).lower()

            # Cek rate limit (kalau Groq SDK tidak throw RateLimitError)
            if "429" in error_str or "rate_limit" in error_str:
                wait = RETRY_BASE_DELAY + (5 * attempt)
                print(f"\n    [429 RATE LIMIT] attempt {attempt+1}/{retries}")
                print(f"    Menunggu {wait} detik...")
                time.sleep(wait)
                continue

            # Connection / timeout error — bisa retry
            if "connection" in error_str or "timeout" in error_str:
                wait = 15 * (attempt + 1)
                print(f"\n    [NETWORK ERROR] {str(e)[:80]}")
                print(f"    Menunggu {wait} detik... ({attempt+1}/{retries})")
                time.sleep(wait)
                continue

            # Error fatal — stop
            print(f"\n    [FATAL ERROR] {str(e)[:100]}")
            return None, True

    # Semua retry habis
    print(f"\n    [GAGAL] Semua {retries}x retry habis. Menyimpan progress...")
    return None, True


# ─────────────────────────────────────────
# FILE I/O
# ─────────────────────────────────────────
def load_input(input_file: str) -> list:
    """Muat data mentah dari reference_finder.py."""
    if not os.path.exists(input_file):
        print(f"[ERROR] File input tidak ditemukan: {input_file}")
        print(f"  Jalankan dulu: python reference_finder.py")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  Data dimuat: {len(data):,} paper dari {input_file}")
    print(f"  File size  : {os.path.getsize(input_file) / 1024 / 1024:.1f} MB")
    return data


def load_output(output_file: str) -> list | None:
    """Muat data yang sudah di-enrich (untuk resume)."""
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_output(data: list, output_file: str):
    """Simpan hasil enrichment ke file JSON."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────
# MAIN ENRICHMENT LOGIC
# ─────────────────────────────────────────
def run_enrichment(
    input_file: str,
    output_file: str,
    batch_size: int,
    start_index: int = 0,
    max_items: int | None = None,
    delay: float = RATE_LIMIT_DELAY
):
    """
    Proses enrichment dengan Groq AI.
    Resume otomatis: skip paper yang sudah punya flag _enriched=True
    """
    print("=" * 65)
    print("  ENRICH WITH GROQ — Step 2: Analisis AI")
    print("=" * 65)
    print(f"  Input       : {input_file}")
    print(f"  Output      : {output_file}")
    print(f"  Model       : {GROQ_MODEL}")
    print(f"  Batch size  : {batch_size}")
    print(f"  Delay       : {delay}s antar request")
    print(f"  Start index : {start_index}")
    print("=" * 65)

    # --- Load data ---
    papers = load_input(input_file)
    total = len(papers)

    # --- Load existing output (resume) ---
    existing = load_output(output_file)
    if existing is not None:
        print(f"\n[RESUME] Ditemukan {len(existing):,} data yang sudah tersimpan.")
        print(f"  Output file: {output_file}")

        # Hitung yang sudah di-enrich
        already_enriched = sum(1 for p in existing if p.get("_enriched", False))
        print(f"  Sudah di-enrich: {already_enriched:,}")
        print(f"  Belum di-enrich: {len(existing) - already_enriched:,}")

        # Gunakan data existing sebagai base (sudah termasuk yang enriched)
        results = existing
    else:
        print("\n[START] Memulai enrichment dari awal...")
        results = []

    # --- Inisialisasi Groq ---
    client = create_groq_client()

    # --- Hitung range yang perlu diproses ---
    # Tentukan start point
    if start_index > 0:
        # Pastikan results sudah berisi data hingga start_index
        if len(results) < start_index:
            print(f"\n  Menyalin {start_index - len(results):,} data mentah dari input...")
            for i in range(len(results), min(start_index, total)):
                results.append(papers[i])
    elif len(results) == 0:
        # Mulai dari index 0
        pass

    # Hitung paper mana yang belum di-enrich
    start_processing = 0
    for i, item in enumerate(results):
        if not item.get("_enriched", False):
            start_processing = i
            break
    else:
        # Semua sudah enriched
        start_processing = len(results)

    if start_processing >= total:
        print(f"\n  Semua {total:,} paper sudah di-enrich!")
        return

    # --- Processing loop ---
    enriched_count = sum(1 for p in results if p.get("_enriched", False))
    error_count = 0
    batch_count = 0
    start_time = time.time()

    print(f"\n  Mulai proses dari index {start_processing}...")
    print(f"  Sudah enriched: {enriched_count:,} | Target: {total:,}")
    print(f"  Sisa: {total - enriched_count:,} paper")
    print(f"  Delay antar request: {delay} detik\n")
    print("-" * 65)

    for i in range(start_processing, total):
        # Cek max_items
        if max_items is not None and (i - start_processing) >= max_items:
            print(f"\n  [LIMIT] Mencapai batas {max_items} item. Berhenti.")
            break

        # Ambil paper dari input (kalau results belum punya index ini)
        if i >= len(results):
            results.append(papers[i])

        paper = results[i]

        # Skip jika sudah di-enrich
        if paper.get("_enriched", False):
            continue

        title = paper.get("title", "Unknown")
        abstract = paper.get("abstract", "")

        # Progress info
        elapsed = time.time() - start_time
        speed = (i - start_processing + 1) / elapsed * 60 if elapsed > 0 else 0
        eta_min = (total - i) / speed if speed > 0 else float('inf')
        eta_str = f"{eta_min:.0f} menit" if eta_min < 120 else f"{eta_min/60:.1f} jam"

        print(f"  [{i+1:>5}/{total}] {title[:65]}...")
        print(f"           speed: {speed:.1f} paper/menit | ETA: {eta_str}")

        # Panggil Groq dengan retry
        ai_data, should_stop = call_groq_with_retry(client, title, abstract, delay=delay)

        if should_stop:
            if ai_data is None:
                # Error fatal atau semua retry habis — simpan & stop
                error_count += 1
                print(f"\n  [STOP] Rate limit persisten atau error. Menyimpan progress...")
                print(f"  Jalankan lagi nanti: python enrich_with_groq.py\n")
                break
            else:
                # Berhasil tapi ada warning
                pass

        if ai_data is not None:
            # Update paper dengan hasil enrichment
            results[i] = {
                **paper,
                "cocok_untuk": ai_data.get("cocok_untuk", []),
                "mengklaim_bahwa": ai_data.get("mengklaim_bahwa", []),
                "ringkasan_ai": ai_data.get("ringkasan_ai", ""),
                "_enriched": True
            }
            enriched_count += 1
        else:
            error_count += 1
            results[i]["_enrich_error"] = True

        # Batch save
        batch_count += 1
        if batch_count >= batch_size:
            save_output(results, output_file)
            batch_count = 0
            print(f"           [SAVED] {enriched_count:,} enriched -> {output_file}")

        # Delay antar request (agar tidak kena rate limit)
        time.sleep(delay)

    # --- Final save ---
    save_output(results, output_file)

    # --- Summary ---
    elapsed_total = time.time() - start_time
    print(f"\n{'='*65}")
    print(f"  ENRICHMENT SELESAI!")
    print(f"{'='*65}")
    print(f"  Total paper       : {total:,}")
    print(f"  Berhasil enriched : {enriched_count:,}")
    print(f"  Gagal/Error       : {error_count:,}")
    print(f"  Belum diproses    : {total - enriched_count - error_count:,}")
    print(f"  Waktu elapsed     : {elapsed_total/60:.1f} menit")
    if enriched_count > 0:
        print(f"  Rata-rata speed   : {enriched_count / (elapsed_total/60):.1f} paper/menit")
    print(f"  Output file       : {output_file}")
    print(f"  File size         : {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    print(f"{'='*65}")

    if enriched_count < total:
        print(f"\n  [INFO] Masih ada {total - enriched_count:,} paper yang belum di-enrich.")
        print(f"  Jalankan lagi: python enrich_with_groq.py")
        print(f"  (Akan otomatis melanjutkan dari yang belum diproses)")

    # --- Statistik bagian skripsi ---
    bagian_stats = {}
    for p in results:
        if p.get("_enriched"):
            for bag in p.get("cocok_untuk", []):
                bagian_stats[bag] = bagian_stats.get(bag, 0) + 1

    if bagian_stats:
        print(f"\n  Distribusi bagian skripsi:")
        for bag, count in sorted(bagian_stats.items(), key=lambda x: -x[1]):
            pct = count / enriched_count * 100 if enriched_count > 0 else 0
            print(f"    {bag:<25}: {count:>5} ({pct:>5.1f}%)")

    print(f"{'='*65}\n")


# ─────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────
def auto_output_path(input_file: str) -> str:
    """
    Generate output filename dari input filename.
    Contoh:
      openalex_english.json     → enriched_english.json
      openalex_indonesian.json  → enriched_indonesian.json
      data.json                 → enriched_data.json
    """
    basename = os.path.basename(input_file)
    dirname = os.path.dirname(input_file) or OUTPUT_DIR

    if basename.startswith("openalex_"):
        # openalex_english.json → enriched_english.json
        out_name = basename.replace("openalex_", "enriched_")
    else:
        # fallback: data.json → enriched_data.json
        name, ext = os.path.splitext(basename)
        out_name = f"enriched_{name}{ext}"

    return os.path.join(dirname, out_name)


def main():
    parser = argparse.ArgumentParser(
        description="Enrich data paper dari OpenAlex menggunakan Groq AI (dengan resume)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python enrich_with_groq.py openalex_english.json              # English
  python enrich_with_groq.py openalex_indonesian.json           # Indonesian
  python enrich_with_groq.py openalex_english.json --delay 5    # Delay 5 detik
  python enrich_with_groq.py openalex_english.json --max 50     # Hanya 50 paper
  python enrich_with_groq.py openalex_english.json --batch-size 10
        """
    )
    parser.add_argument("input", type=str,
                        help="File input JSON dari reference_finder.py")
    parser.add_argument("--output", type=str, default=None,
                        help="File output (default: otomatis dari nama input)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Save ke file setiap N paper berhasil (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--delay", type=float, default=RATE_LIMIT_DELAY,
                        help=f"Delay antar request Groq dalam detik (default: {RATE_LIMIT_DELAY})")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Mulai dari index tertentu (default: 0, auto-detect jika resume)")
    parser.add_argument("--max", type=int, default=None,
                        help="Batas maksimal paper yang diproses (default: semua)")

    args = parser.parse_args()

    input_file = args.input
    if not os.path.exists(input_file):
        # Coba dengan OUTPUT_DIR prefix
        alt = os.path.join(OUTPUT_DIR, input_file)
        if os.path.exists(alt):
            input_file = alt
        else:
            print(f"[ERROR] File tidak ditemukan: {args.input}")
            sys.exit(1)

    output_file = args.output or auto_output_path(input_file)

    run_enrichment(
        input_file=input_file,
        output_file=output_file,
        batch_size=args.batch_size,
        start_index=args.start_index,
        max_items=args.max,
        delay=args.delay
    )


if __name__ == "__main__":
    main()
