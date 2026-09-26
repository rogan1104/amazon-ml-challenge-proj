"""Profile retrieval on a small S1 sample (for PC tuning; no full-dataset run)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from analysis.retrieval.build import _index_path
from analysis.retrieval.channels import active_channels
from analysis.retrieval.config import RetrievalConfig
from analysis.retrieval.index import InvertedIndex
from analysis.retrieval.runner import _stream_s1_batches

# Semantics check size is fixed — never scales with CLI --limit.
SEMANTICS_VERIFY_ROWS = 50


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _collect_s1_rows(cfg: RetrievalConfig, max_rows: int) -> list[dict]:
    rows: list[dict] = []
    for batch in _stream_s1_batches(cfg, max_s1_rows=max_rows):
        for row in batch:
            if row.get("entity_id", "").strip():
                rows.append(row)
                if len(rows) >= max_rows:
                    return rows
    return rows


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
    sample_rows: int = SEMANTICS_VERIFY_ROWS,
) -> dict:
    """
    Compare batched vs per-row baseline retrieval on a tiny fixed sample.

    sample_rows is capped independently of benchmark --limit.
    """
    from analysis.retrieval.channels import BaselineChannel

    sample_rows = min(sample_rows, SEMANTICS_VERIFY_ROWS)
    rows = _collect_s1_rows(cfg, sample_rows)
    if not rows:
        return {
            "sample_s1_rows": 0,
            "comparisons": 0,
            "mismatches": 0,
            "ok": True,
            "batched_verify_sec": 0.0,
            "row_by_row_verify_sec": 0.0,
            "total_verify_sec": 0.0,
        }

    channel = BaselineChannel(cfg.baseline)
    mismatches = 0
    checked = 0
    t_total0 = time.perf_counter()

    with InvertedIndex(index_path, read_only=True) as index:
        batch_hits_by_target: dict[str, dict[str, set[str]]] = {}

        t0 = time.perf_counter()
        for target in ("S2", "S3"):
            batch_hits_by_target[target] = channel.retrieve_batch(index, target, rows)
        t_batched = time.perf_counter() - t0

        t0 = time.perf_counter()
        for target in ("S2", "S3"):
            batch_hits = batch_hits_by_target[target]
            for row in rows:
                s1_id = row["entity_id"].strip()
                row_hits = channel.retrieve(index, target, row)
                if row_hits != batch_hits.get(s1_id, set()):
                    mismatches += 1
                checked += 1
        t_row = time.perf_counter() - t0

    t_total = time.perf_counter() - t_total0

    return {
        "sample_s1_rows": len(rows),
        "comparisons": checked,
        "mismatches": mismatches,
        "ok": mismatches == 0,
        "batched_verify_sec": round(t_batched, 4),
        "row_by_row_verify_sec": round(t_row, 4),
        "total_verify_sec": round(t_total, 4),
    }


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
        "semantics_verify_rows": SEMANTICS_VERIFY_ROWS,
        "index_path": str(index_path),
    }

    _log(f"[benchmark] batched retrieval on {limit:,} S1 rows (mode={cfg.mode})...")
    t0 = time.perf_counter()
    report["batched"] = _time_batched_loop(index_path, cfg, limit)
    report["batched"]["wall_sec"] = round(time.perf_counter() - t0, 4)
    _log(
        f"[benchmark] batched done: {report['batched']['rows']:,} rows in "
        f"{report['batched']['wall_sec']:.2f}s "
        f"(sqlite={report['batched']['sqlite_sec']:.2f}s)"
    )

    if verify and cfg.mode == "baseline":
        _log(
            f"[benchmark] semantics check on {SEMANTICS_VERIFY_ROWS} S1 rows only "
            f"(independent of --limit)..."
        )
        t0 = time.perf_counter()
        report["semantics_check"] = verify_baseline_semantics(index_path, cfg)
        wall = round(time.perf_counter() - t0, 4)
        report["semantics_check"]["wall_sec"] = wall
        sc = report["semantics_check"]
        _log(
            f"[benchmark] semantics done in {wall:.2f}s "
            f"(batched={sc['batched_verify_sec']:.2f}s, "
            f"row-by-row={sc['row_by_row_verify_sec']:.2f}s, "
            f"ok={sc['ok']})"
        )
    elif verify and cfg.mode != "baseline":
        _log("[benchmark] skipping semantics check (baseline mode only)")

    if compare_row_loop and cfg.mode == "baseline" and report.get("semantics_check"):
        sc = report["semantics_check"]
        if sc["row_by_row_verify_sec"] > 0:
            report["sqlite_speedup_row_vs_batch_on_verify_sample"] = round(
                sc["row_by_row_verify_sec"] / max(sc["batched_verify_sec"], 1e-6),
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
    _log(f"[benchmark] report written: {out_path}")
    return report
