"""Summarize synchronized per-task wall time; never call it pure GPU compute time."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def describe(values):
    """Return ms mean and sample std; quantiles use linear interpolation."""
    ordered = sorted(values)
    def percentile(q):
        index = (len(ordered) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return {'mean': statistics.mean(values), 'std': statistics.stdev(values) if len(values) > 1 else 0,
            'p50': percentile(.5), 'p90': percentile(.9)}


def summarize(root, stages, steps, warmup):
    """Require every rank and zero-based step; warmup is a count, not a timestamp."""
    if stages < 2 or steps <= warmup or warmup < 0:
        raise ValueError('invalid dimensions')
    if {p.name for p in root.glob('rank-*')} != {f'rank-{i}' for i in range(stages)}:
        raise ValueError('rank set mismatch')
    result = []
    for stage in range(stages):
        rank = root / f'rank-{stage}'
        with (rank / 'stage-times.csv').open() as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['step', 'stage', 'forward_ms', 'backward_ms']:
                raise ValueError('invalid stage CSV header')
            rows = list(reader)
        if [int(row['step']) for row in rows] != list(range(steps)):
            raise ValueError('incomplete/duplicate stage steps')
        totals = {}
        for row in rows:
            fwd, bwd = float(row['forward_ms']), float(row['backward_ms'])
            if int(row['stage']) != stage or not all(math.isfinite(x) and x >= 0 for x in (fwd, bwd)):
                raise ValueError('invalid stage time')
            totals[int(row['step'])] = [fwd, bwd]
        observed = {step: [0., 0.] for step in range(steps)}
        counts = {step: [0, 0] for step in range(steps)}
        seen = set()
        with (rank / 'tasks.csv').open() as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['step', 'stage', 'microbatch', 'gid', 'direction', 'start_ms', 'end_ms']:
                raise ValueError('invalid task CSV header')
            for row in reader:
                step, mb, gid = (int(row[k]) for k in ('step', 'microbatch', 'gid'))
                direction = row['direction']
                key = (step, mb, gid, direction)
                if step not in observed or int(row['stage']) != stage or mb < 0 or gid < 0 or key in seen:
                    raise ValueError('invalid/duplicate task identity')
                if direction not in ('forward', 'backward'):
                    raise ValueError('invalid task direction')
                begin, end = float(row['start_ms']), float(row['end_ms'])
                if not all(math.isfinite(x) for x in (begin, end)) or begin < 0 or end < begin:
                    raise ValueError('invalid task interval')
                seen.add(key)
                index = 0 if direction == 'forward' else 1
                observed[step][index] += end - begin
                counts[step][index] += 1
        for step in totals:
            if min(counts[step]) == 0 or any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)
                                           for a, b in zip(totals[step], observed[step])):
                raise ValueError('task coverage or stage total mismatch')
        kept = [totals[step] for step in range(warmup, steps)]
        result.append({'stage': stage, 'forward_ms': describe([v[0] for v in kept]),
                       'backward_ms': describe([v[1] for v in kept]),
                       'total_ms': describe([sum(v) for v in kept])})
    return {'measurement': 'synchronized_task_wall_time_including_waits',
            'steps': steps, 'warmup': warmup, 'stages': result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--stages', type=int, default=2)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.stages, args.steps, args.warmup), indent=2))


if __name__ == '__main__':
    main()
