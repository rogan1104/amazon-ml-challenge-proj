# Mmap baseline candidate retrieval

Fast baseline (`bn` / `ba` / `cp5`) candidate generation using a memory-mapped CSR index.
**Does not replace** the SQLite path in `analysis.retrieval.cli`.

## Index (one-time per split)

```bash
python -m analysis.retrieval.mmap_baseline.cli build \
  --split train \
  --output reports/cache/baseline_mmap/train
```

## Production candidate TSV

Writes the same schema as `analysis.retrieval.runner.write_candidate_pairs` for baseline mode:

- Header: `source1_entity_id`, `candidate_entity_ids`
- Comma-separated, sorted S2/S3 ids; empty second column if no candidates
- Default output: `reports/retrieval/candidate_pairs_{split}_mmap.tsv`

```bash
# Full train run (~388 S1/s on PC benchmark; ~2.2M rows)
python -m analysis.retrieval.mmap_baseline.cli run --split train

# Smoke (100 S1 + schema validate)
python -m analysis.retrieval.mmap_baseline.cli run \
  --split train --limit 100 --validate
```

Downstream scoring / features can consume the TSV via `analysis.scoring.pairs.stream_candidate_pairs`.

## Other commands

- `verify` — mmap vs SQLite equivalence (use `--partial` for smoke indexes)
- `benchmark` — timing only, no TSV
