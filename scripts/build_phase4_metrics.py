"""Regenerate Phase 4 scenario metrics from preserved CSV/JSONL inputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from relay.evaluation.metrics import summarize_scenario


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    results_path = ROOT / "experiments/raw/results.csv"
    if not results_path.exists():
        raise FileNotFoundError("No raw results.csv yet; run the local load scenarios first")
    grouped = defaultdict(list)
    with results_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("source") == "stand-in":
                grouped[row["scenario"]].append(row)
    runs_path = ROOT / "experiments/raw/load_runs.jsonl"
    runs = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("valid", False)]
    run_summary = defaultdict(lambda: {"runs": 0, "offered_requests": 0, "offered_seconds": 0.0})
    for run in runs:
        item = run_summary[run["scenario"]]
        item["runs"] += 1
        item["offered_requests"] += int(run.get("offered_requests", 0))
        item["offered_seconds"] += float(run.get("offered_elapsed_seconds", 0))
    valid_request_ids = {request["request_id"] for run in runs for request in run.get("requests", [])
                         if request.get("request_id")}
    summary = {}
    for scenario, rows in sorted(grouped.items()):
        rows = [row for row in rows if row.get("request_id") in valid_request_ids]
        if not rows:
            continue
        metrics = summarize_scenario(rows)
        load = run_summary[scenario]
        metrics["run_count"] = load["runs"]
        metrics["offered_requests"] = load["offered_requests"]
        metrics["mean_offered_rate_per_second"] = (
            load["offered_requests"] / load["offered_seconds"] if load["offered_seconds"] else None
        )
        summary[scenario] = metrics
    output = ROOT / "experiments/summaries/phase4_load_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"source": "stand-in", "percentile_method": "NumPy linear quantile",
                                  "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
