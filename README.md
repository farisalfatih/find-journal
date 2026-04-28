# find-journal

CLI tools untuk mencari, mengambil, dan menganalisis journal akademik. Data diambil dari [OpenAlex](https://openalex.org/) lalu di-enrich menggunakan Groq AI untuk mendapatkan metadata tambahan seperti ringkasan, klaim penelitian, dan relevansi bagian paper.

## Alur Kerja

```
OpenAlex → reference_finder.py → enrich_with_groq.py → search-journal.py
(fetch)                      (enrich)                (search)
```

## Struktur File

| File | Fungsi |
|------|--------|
| `reference_finder.py` | Fetch journal dari OpenAlex berdasarkan keyword |
| `enrich_with_groq.py` | Tambahkan metadata/analisis tambahan menggunakan Groq AI |
| `search-journal.py` | Cari journal yang sudah ada di database JSON lokal |
| `requirements.txt` | Dependensi Python |

## Output Data

Setiap journal memiliki format lengkap:

```json
{
  "title": "...",
  "doi": "...",
  "publication_year": 2024,
  "penulis": ["...", "..."],
  "abstract": "...",
  "language": "id",
  "_enriched": true,
  "cocok_untuk": ["Pendahuluan", "Tinjauan Pustaka"],
  "mengklaim_bahwa": ["..."],
  "ringkasan_ai": "..."
}
```

## Instalasi

```bash
pip install -r requirements.txt
```

## Penggunaan

### 1. Fetch journal dari OpenAlex
```bash
python reference_finder.py "keyword"
```

### 2. Enrich data dengan Groq AI
```bash
python enrich_with_groq.py
```

### 3. Cari journal di database lokal
```bash
python search-journal.py "keyword" data.json
python search-journal.py "bitcoin" enriched_indonesian.json --lang id --max 5
python search-journal.py "LSTM" enriched_english.json --lang en --year-start 2021 --year-end 2023 --pretty
```
