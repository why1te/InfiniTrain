import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np

STEP = re.compile(r"step\s+(\d+)/(\d+)\s+\|\s+train loss\s+(\S+).*?\((\S+)\s+ms\s+\|\s+(\S+)\s+tok/s")
TAG = re.compile(r"^Tag:\s*Step_(\d+)\s*$")
RECORD = re.compile(r"^\s*\d+\s+\S+\s+\S+\s+(.+?)\s+(Device\(CUDA,\s*\d+\)|cuda:\d+|cpu)\s*(\d+)\s+(\d+)\s+(\d+)\s*$")


def describe(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Empty or non-finite observations")
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p50": float(np.percentile(values, 50)), "p90": float(np.percentile(values, 90))}


def training(folder, total=60, warmup=10):
    if int((folder / "exit-code.txt").read_text()) != 0:
        raise ValueError(f"Failed run: {folder}")
    rows = STEP.findall((folder / "training.log").read_text())
    if [(int(s), int(n)) for s, n, *_ in rows] != [(i, total) for i in range(1, total + 1)]:
        raise ValueError(f"Incomplete or repeated steps: {folder}")
    for _, _, loss, ms, tps in rows:
        if not all(math.isfinite(float(x)) for x in (loss, ms, tps)) or float(ms) <= 0 or float(tps) <= 0:
            raise ValueError(f"Invalid training values: {folder}")
    return [float(row[3]) for row in rows[warmup:]]


def profile(path):
    training(path.parent)
    current = None
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith("Tag:"):
            match = TAG.match(line)
            current = int(match[1]) if match else None
            continue
        match = RECORD.match(line)
        if current is None or not 10 <= current < 60:
            continue
        if not match:
            if "cuda" in line.lower() and re.match(r"^\s*\d+\s", line):
                raise ValueError(f"Unrecognized CUDA record in {path}: {line}")
            continue
        if match[2].lower().startswith(("cuda", "device(cuda")) and "nccl" not in match[1].lower():
            values[current] = values.get(current, 0) + int(match[4]) / 1000
    if set(values) != set(range(10, 60)) or any(x <= 0 for x in values.values()):
        raise ValueError(f"Missing/empty stable profiler tags: {path}")
    return [values[i] for i in range(10, 60)]


def union_ms(intervals):
    total = 0.0
    end = -math.inf
    for left, right in sorted(intervals):
        if right > end:
            total += right - max(left, end)
            end = right
    return total


def trace(path, start_ms, end_ms, gpu_ids):
    if len(set(gpu_ids)) != 2 or not all(math.isfinite(x) for x in (start_ms, end_ms)) or end_ms <= start_ms:
        raise ValueError("Require two distinct GPU IDs and a finite increasing window")
    rows = list(csv.reader(path.read_text().splitlines()))
    header = next((i for i, row in enumerate(rows) if row and row[0].lstrip('\ufeff').strip().lower().startswith("start")), None)
    if header is None:
        raise ValueError("Cannot find cuda_gpu_trace header")
    names = [x.strip().lower() for x in rows[header]]
    groups = {gpu: {k: [] for k in ("compute", "nccl", "memory")} for gpu in gpu_ids}
    seen = set()
    for raw in rows[header + 1:]:
        if not raw or not any(raw):
            continue
        if len(raw) != len(names):
            raise ValueError(f"Unrecognized CSV row: {raw}")
        row = dict(zip(names, raw))
        times = []
        for prefix in ("start", "duration"):
            keys = [key for key in names if key.startswith(prefix)]
            if len(keys) != 1:
                raise ValueError(f"Missing/ambiguous {prefix} column")
            key = keys[0]
            unit = re.search(r"\((ns|us|µs|μs|ms|s)\)", key)
            if not unit:
                raise ValueError(f"Unknown time unit: {key}")
            factor = {"ns": 1e-6, "us": 1e-3, "µs": 1e-3, "μs": 1e-3, "ms": 1, "s": 1000}[unit[1]]
            times.append(float(row[key].replace(",", "")) * factor)
        if not all(math.isfinite(x) for x in times) or times[1] < 0:
            raise ValueError("Invalid event time")
        device = next((v for k, v in row.items() if k.startswith("device")), "")
        digits = re.findall(r"\d+", device)
        if not digits:
            raise ValueError(f"Unknown device: {device}")
        gpu = int(digits[-1])
        if gpu not in groups:
            raise ValueError(f"Unexpected GPU {gpu}; check mapping")
        seen.add(gpu)
        left, right = max(start_ms, times[0]), min(end_ms, sum(times))
        if right <= left:
            continue
        name = row.get("name", "").lower()
        if "nccl" in name:
            kind = "nccl"
        elif "memcpy" in name or "memset" in name:
            kind = "memory"
        elif any(v.strip() for k, v in row.items() if k.startswith("grid")) or "kernel" in row.get("category", "").lower():
            kind = "compute"
        else:
            raise ValueError(f"Unclassified event in window: {name}")
        groups[gpu][kind].append((left, right))
    if seen != set(gpu_ids):
        raise ValueError("Both GPUs must be present in trace")
    window = end_ms - start_ms
    result = {"window_start_ms": start_ms, "window_end_ms": end_ms, "stages": []}
    for stage, gpu in enumerate(gpu_ids):
        parts = groups[gpu]
        active = union_ms([item for intervals in parts.values() for item in intervals])
        result["stages"].append({"stage": stage, "gpu": gpu,
            "compute_utilization": union_ms(parts["compute"]) / window,
            "gpu_idle_ratio": 1 - active / window})
    result["overall_gpu_idle_ratio"] = statistics.mean(x["gpu_idle_ratio"] for x in result["stages"])
    return result


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    stage = modes.add_parser("stage")
    stage.add_argument("records", nargs=2, type=Path, help="Stage 0 and Stage 1 records, in that order")
    tps = modes.add_parser("tps")
    tps.add_argument("root", type=Path)
    tps.add_argument("--tokens", type=int, default=1024)
    ns = modes.add_parser("trace")
    ns.add_argument("csv", type=Path)
    ns.add_argument("start_ms", type=float)
    ns.add_argument("end_ms", type=float)
    ns.add_argument("gpu_ids", nargs=2, type=int, help="GPU IDs for Stage 0 and Stage 1")
    args = parser.parse_args()
    if args.mode == "stage":
        result = {"stages": [describe(profile(path)) for path in args.records]}
        means = [x["mean"] for x in result["stages"]]
        result.update(imbalance=max(means) / statistics.mean(means) - 1, slowest_stage_ms=max(means))
    elif args.mode == "tps":
        if args.tokens <= 0:
            raise ValueError("tokens must be positive")
        result = {}
        for label, prefix in (("default", "a"), ("custom", "b")):
            runs = []
            for number in (1, 2, 3):
                values = training(args.root / f"tps-{prefix}{number}")
                runs.append({"tok_per_s": len(values) * args.tokens / (sum(values) / 1000), "step_ms": describe(values)})
            result[label] = {"runs": runs, "tok_per_s": describe([x["tok_per_s"] for x in runs])}
        result["speedup"] = result["custom"]["tok_per_s"]["mean"] / result["default"]["tok_per_s"]["mean"] - 1
    else:
        result = trace(args.csv, args.start_ms, args.end_ms, args.gpu_ids)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
