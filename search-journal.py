#!/usr/bin/env python3
"""
Journal Search CLI
==================
Pencarian journal dari database JSON yang sudah di-enriched.

Cara pakai:
  python search-journal.py "keyword" data.json
  python search-journal.py "bitcoin prediction" enriched_english.json
  python search-journal.py "hukum islam" enriched_indonesian.json
  python search-journal.py "random forest" data.json --lang id
  python search-journal.py "bitcoin" data.json --year 2021 2023
  python search-journal.py "LSTM" data.json --limit 5

Output: Full metadata JSON dari journal yang cocok.
"""

import json
import sys
import re
import argparse


def normalize_text(text: str) -> str:
    """Normalisasi teks: lowercase, hapus karakter spesial."""
    return re.sub(r"[^\w\s]", " ", text.lower()).strip()


def search_journal(journal: dict, query: str) -> bool:
    """
    Cek apakah SEMUA kata kunci muncul di salah satu field journal.
    Pencarian bersifat AND (semua kata harus ada).
    """
    query_terms = [t for t in normalize_text(query).split() if t]
    if not query_terms:
        return False

    return query_terms.every if hasattr(query_terms, "every") else all(
        any(
            normalize_text(str(journal.get(field, ""))).contains(term)
            if hasattr(normalize_text(str(journal.get(field, ""))), "contains")
            else term in normalize_text(str(journal.get(field, "")))
            for field in [
                "title",
                "abstract",
                "ringkasan_ai",
                "doi",
                "language",
            ]
        )
        or any(
            term in normalize_text(str(item))
            for item in journal.get("penulis", [])
        )
        or any(
            term in normalize_text(str(item))
            for item in journal.get("cocok_untuk", [])
        )
        or any(
            term in normalize_text(str(item))
            for item in journal.get("mengklaim_bahwa", [])
        )
        or str(journal.get("publication_year", "")) == term
        for term in query_terms
    )


def clean_html(text: str) -> str:
    """Hapus tag HTML dari teks."""
    return re.sub(r"<[^>]+>", "", text)


def clean_journal(journal: dict) -> dict:
    """Bersihkan journal dari HTML tags dan format output."""
    cleaned = {}

    cleaned["title"] = clean_html(journal.get("title", ""))
    cleaned["doi"] = journal.get("doi")

    # publication_year sebagai integer
    py = journal.get("publication_year")
    cleaned["publication_year"] = int(py) if py is not None else None

    cleaned["penulis"] = journal.get("penulis", [])

    # abstract: bersihkan dari HTML tags
    abstract = journal.get("abstract", "")
    cleaned["abstract"] = clean_html(abstract).strip()

    cleaned["language"] = journal.get("language", "")
    cleaned["_enriched"] = journal.get("_enriched", False)
    cleaned["cocok_untuk"] = journal.get("cocok_untuk", [])
    cleaned["mengklaim_bahwa"] = journal.get("mengklaim_bahwa", [])
    cleaned["ringkasan_ai"] = journal.get("ringkasan_ai", "")

    # Hapus field yang None/empty
    return {k: v for k, v in cleaned.items() if v is not None}


def main():
    parser = argparse.ArgumentParser(
        description="Cari journal dari database JSON yang sudah di-enriched.",
        epilog='Contoh: python search-journal.py "bitcoin prediction" data.json',
    )
    parser.add_argument("keyword", help="Kata kunci pencarian")
    parser.add_argument("data_file", help="Path ke file JSON database journal")
    parser.add_argument(
        "--lang",
        choices=["id", "en"],
        default=None,
        help="Filter bahasa: id (Indonesia) atau en (English)",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=None,
        help="Filter tahun minimum (inklusif)",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=None,
        help="Filter tahun maksimum (inklusif)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batas jumlah hasil (default: semua)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Output JSON dengan indentasi rapi",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output raw JSON (default: format lengkap per journal)",
    )

    args = parser.parse_args()

    # Load data
    try:
        with open(args.data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] File tidak ditemukan: {args.data_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] File JSON tidak valid: {e}", file=sys.stderr)
        sys.exit(1)

    # Pastikan data berupa list
    if not isinstance(data, list):
        data = [data]

    # Filter bahasa
    if args.lang:
        data = [j for j in data if j.get("language") == args.lang]

    # Filter tahun
    if args.year_start is not None:
        data = [j for j in data if j.get("publication_year", 0) >= args.year_start]
    if args.year_end is not None:
        data = [j for j in data if j.get("publication_year", 0) <= args.year_end]

    # Cari
    results = [clean_journal(j) for j in data if search_journal(j, args.keyword)]

    # Limit
    if args.limit is not None:
        results = results[: args.limit]

    # Output
    indent = 2 if args.pretty else None
    if args.raw:
        output = json.dumps(results, ensure_ascii=False, indent=indent)
    else:
        if len(results) == 0:
            print(f"Tidak ditemukan journal untuk keyword: \"{args.keyword}\"")
            print(f"Total data yang di-scan: {len(data)} journal")
            sys.exit(0)

        print(f"=== Hasil Pencarian ===")
        print(f'Keyword : "{args.keyword}"')
        print(f"Ditemukan: {len(results)} journal\n")

        for i, journal in enumerate(results, 1):
            print(f"--- Journal #{i} ---")
            print(json.dumps(journal, ensure_ascii=False, indent=indent))
            print()

    if args.raw:
        print(output)


if __name__ == "__main__":
    main()
