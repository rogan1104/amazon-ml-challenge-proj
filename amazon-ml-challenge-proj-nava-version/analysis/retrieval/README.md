# High-Recall Candidate Retrieval Experiment

Experimental retrieval layer for the Amazon ML Business Entity Resolution challenge.
**Independent of** `analysis/phase10_blocking.py` (reference baseline E simulator, ~77.3% cached recall).

## Architecture

```
S1 (query)                          S2 / S3 (indexed separately)
    │                                      │
    ├─ BaselineChannel ────────┐           │  inverted index (SQLite, disk)
    ├─ CharNgramChannel ───────┤           │  key → entity_id postings
    ├─ AddressNumericChannel ──┼─ query ───┤
    └─ TransliterationChannel ─┘           │
                    │                      │
                    └──── union + dedupe ──┘
                              │
                    candidate_pairs.tsv
```

### Channels (each independently switchable)

| Mode | Channel | Keys indexed |
|------|---------|--------------|
| `baseline` | Baseline | norm name, norm addr, `country\|name_prefix5` |
| `char_ngram` | CharNgram | 3-gram chars on normalized name (`ng:...`) |
| `address_numeric` | AddressNumeric | postal/PIN/ZIP, house number, digits ≥3 chars |
| `transliteration` | Transliteration | Latin translit of Indic names + translit n-grams |
| `all` | Union of above | All keys |

### Design properties

- **No O(S1×S2) comparisons** — disk-backed SQLite inverted index; lookup is O(keys × avg_postings_per_key).
- **S2 and S3 separate** — each target has its own index partition; candidates unioned per S1.
- **Deduplicated output** — `set` union across channels and targets before writing TSV.
- **Stop-key filtering** — high document-frequency keys skipped (`max_df`) to limit explosion.
- **Per-S1 caps** — configurable `max_candidates_per_s1` per channel.
- **Streaming build** — TSV read in batches (`batch_size=50000`); original TSV files never modified.

### Reused utilities

- `analysis/normalize.py` — `normalize_name`, `char_ngrams`, `extract_digits`, `extract_postal`
- `analysis/config.py` — dataset paths (`TRAIN`, `TEST`)

## Expected outputs

| File | Description |
|------|-------------|
| `reports/cache/retrieval_index/index_{split}.sqlite` | Disk-backed inverted index |
| `reports/retrieval/candidate_pairs_{split}_{mode}.tsv` | Challenge-format candidates |
| `reports/retrieval/retrieval_metrics_{split}_{mode}.json` | Counts, timing, optional recall |

### candidate_pairs.tsv format

```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S3-00812
S1-00002	
```

## PC commands (do not run on Mac dev machine)

From project root with venv activated:

```bash
# Single channel benchmarks
python -m analysis.retrieval.cli full --mode baseline --split train --evaluate
python -m analysis.retrieval.cli full --mode char_ngram --split train --evaluate
python -m analysis.retrieval.cli full --mode address_numeric --split train --evaluate
python -m analysis.retrieval.cli full --mode transliteration --split train --evaluate

# Combined high-recall run
python -m analysis.retrieval.cli full --mode all --split train --evaluate

# Test-set inference (no GT evaluation)
python -m analysis.retrieval.cli full --mode all --split test \
  --output output/candidate_pairs.tsv

# Restartable: build index once, run multiple modes
python -m analysis.retrieval.cli build-index --mode all --split train
python -m analysis.retrieval.cli run --mode char_ngram --split train --evaluate
python -m analysis.retrieval.cli run --mode transliteration --split train --evaluate

# Profile optimized retrieval on 10K S1 rows (existing index; no rebuild)
python -m analysis.retrieval.cli benchmark --mode baseline --split train --limit 10000

# Smoke test write path on 10K rows
python -m analysis.retrieval.cli run --mode baseline --split train --limit 10000 \
  --output reports/retrieval/candidate_pairs_train_baseline_10k.tsv
```

### Retrieval performance

S1 lookup uses **batched SQLite joins** (`batch_query` temp table + single join per S2/S3/channel sub-batch) instead of one query per S1 row. Progress logs every 25K rows by default.

## Tuning (analysis/retrieval/config.py)

- `CharNgramConfig.min_overlap` — minimum shared n-grams (default 2)
- `CharNgramConfig.max_df` — skip n-grams appearing in >50K entities
- `AddressNumericConfig.min_digit_token_len` — ignore short digits (default 3)
- `TransliterationConfig` — Indic mapping tables in `transliterate.py`

## Not included (future work)

- LightGBM matcher
- Embedding / FAISS retrieval
- Modifications to `phase10_blocking.py`
