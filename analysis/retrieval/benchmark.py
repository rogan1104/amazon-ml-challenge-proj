"""Profile retrieval on a small S1 sample (for PC tuning; no full-dataset run)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from analysis.retrieval.build import _index_path
from analysis.retrieval.channels import active_channels
from analysis.retrieval.config import RetrievalConfig
from analysis.retrieval.index import InvertedIndex
from analysis.retrieval.runner import _stream_s1_batches


def _time_baseline_row_loop(
    index_path: Path,
    cfg: RetrievalConfig,
    limit: int,
) -> dict:
    """Legacy per-row SQLite path (for profiling comparison only)."""
    from analysis.retrieval.channels import BaselineChannel

    channel = BaselineChannel(cfg.baseline)
    t_read = t_sql = t_keys = 0.0
    n = 0

    with InvertedIndex(index_path, read_only=True) as index:
        for batch in _stream_s1_batches(cfg, max_s1_rows=limit):
            for row in batch:
                s1_id = row.get("entity_id", "").strip()
                if not s1_id:
                    continue
                t0 = time.perf_counter()
                keys = list(channel.extract_query_keys(row))
                t_keys += time.perf_counter() - t0

                for target in ("S2", "S3"):
                    t0 = time.perf_counter()
                    channel.retrieve(index, target, row)
                    t_sql += time.perf_counter() - t0
                n += 1

    return {
        "rows": n,
        "key_extract_sec": round(t_keys, 4),
        "sqlite_sec": round(t_sql, 4),
        "total_sec": round(t_keys + t_sql, 4),
    }


def _time_batched_loop(
    index_path: Path,
    cfg: RetrievalConfig,
    limit: int,
) -> dict:
    channels = active_channels(cfg)
    t_read = t_sql = t_keys = 0.0
    n = 0

    with InvertedIndex(index_path, read_only=True) as index:
        for batch in _stream_s1_batches(cfg, max_s1_rows=limit):
            t0 = time.perf_counter()
            rows = [r for r in batch if r.get("entity_id", "").strip()]
            t_read += time.perf_counter() - t0

            for row in rows:
                t0 = time.perf_counter()
                for ch in channels:
                    list(ch.extract_query_keys(row))
                t_keys += time.perf_counter() - t0

            for target in ("S2", "S3"):
                for channel in channels:
                    t0 = time.perf_counter()
                    channel.retrieve_batch(index, target, rows)  # type: ignore[arg-type]
                    t_sql += time.perf_counter() - t0
            n += len(rows)

    return {
        "rows": n,
        "tsv_batch_sec": round(t_read, 4),
        "key_extract_sec": round(t_keys, 4),
        "sqlite_sec": round(t_sql, 4),
        "total_sec": round(t_read + t_keys + t_sql, 4),
    }


def verify_baseline_semantics(
    index_path: Path,
    cfg: RetrievalConfig,
    sample_rows: int = 200,
) -> dict:
    """Ensure batched baseline retrieval matches per-row results on a sample."""
    from analysis.retrieval.channels import BaselineChannel

    channel = BaselineChannel(cfg.baseline)
    mismatches = 0
    checked = 0

    with InvertedIndex(index_path, read_only=True) as index:
        for batch in _stream_s1_batches(cfg, max_s1_rows=sample_rows):
            rows = [r for r in batch if r.get("entity_id", "").strip()]
            for target in ("S2", "S3"):
                batch_hits = channel.retrieve_batch(index, target, rows)
                for row in rows:
                    s1_id = row["entity_id"].strip()
                    row_hits = channel.retrieve(index, target, row)
                    if row_hits != batch_hits.get(s1_id, set()):
                        mismatches += 1
                    checked += 1

    return {"checked": checked, "mismatches": mismatches, "ok": mismatches == 0}


def run_benchmark(
    cfg: RetrievalConfig,
    *,
    limit: int = 10_000,
    index_path: Path | None = None,
    compare_row_loop: bool = True,
    verify: bool = True,
) -> dict:
    index_path = index_path or _index_path(cfg)
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    report: dict = {
        "split": cfg.split,
        "mode": cfg.mode,
        "limit_s1": limit,
        "index_path": str(index_path),
    }

    if verify and cfg.mode == "baseline":
        report["semantics_check"] = verify_baseline_semantics(index_path, cfg)

    report["batched"] = _time_batched_loop(index_path, cfg, limit)

    if compare_row_loop and cfg.mode == "baseline":
        # Row loop is intentionally slow; cap comparison sample for long runs.
        row_limit = min(limit, 500)
        report["per_row_sqlite_note"] = (
            f"Per-row loop timed on {row_limit} rows only (full {limit} would take hours)."
        )
        report["per_row"] = _time_baseline_row_loop(index_path, cfg, row_limit)
        batched = report["batched"]
        per = report["per_row"]
        if per["sqlite_sec"] > 0:
            # Extrapolate speedup at same row count as per-row sample.
            batched_sub = _time_batched_loop(index_path, cfg, row_limit)
            report["batched_at_row_limit"] = batched_sub
            report["sqlite_speedup_vs_row_loop"] = round(
                per["sqlite_sec"] / max(batched_sub["sqlite_sec"], 1e-6),
                2,
            )

    batched = report["batched"]
    rows = batched["rows"] or 1
    report["throughput_s1_per_sec"] = round(rows / max(batched["total_sec"], 1e-6), 2)
    report["estimated_full_train_hours"] = round(
        (2_206_821 / report["throughput_s1_per_sec"]) / 3600,
        2,
    )

    out_path = cfg.output_dir / f"benchmark_{cfg.split}_{cfg.mode}_{limit}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["benchmark_json"] = str(out_path)
    return report
