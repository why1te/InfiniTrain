"""Strict comparison of complete per-rank parameter dumps, including scalar loss."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def load(root):
    """Return global-name -> tensor and the sole logical-output-stage loss."""
    tensors, losses = {}, []
    ranks = sorted(root.glob('rank-*'))
    if not ranks or any(not p.is_dir() for p in ranks):
        raise ValueError('missing rank directories')
    for rank in ranks:
        with (rank / 'parameters.tsv').open() as stream:
            reader = csv.DictReader(stream, delimiter='\t')
            if reader.fieldnames != ['name', 'shape', 'status']:
                raise ValueError('invalid parameter manifest')
            rows = list(reader)
        if not rows:
            raise ValueError('empty parameter manifest')
        expected = set()
        for row in rows:
            name = row['name']
            if not name or name in tensors or row['status'] != 'OK':
                raise ValueError(f'duplicate/missing gradient: {name}')
            path = rank / (name + '.npy')
            array = np.load(path, allow_pickle=False)
            shape = tuple(int(x) for x in row['shape'].split(',') if x)
            if array.shape != shape or array.dtype != np.float32 or not array.size or not np.isfinite(array).all():
                raise ValueError(f'invalid tensor: {path}')
            tensors[name] = array
            expected.add(path.name)
        if {p.name for p in rank.glob('*.npy')} != expected:
            raise ValueError('manifest and NPY file set differ')
        if (rank / 'loss.txt').exists():
            loss = float((rank / 'loss.txt').read_text())
            if not np.isfinite(loss):
                raise ValueError('non-finite loss')
            losses.append(loss)
    if len(losses) != 1:
        raise ValueError('expected exactly one logical-output-stage loss')
    return tensors, losses[0]


def compare_inputs(reference, candidate):
    """Require identical full PP inputs/targets on every rank and in both runs."""
    ranks = [sorted(root.glob('rank-*')) for root in (reference, candidate)]
    if len(ranks[0]) < 2 or [p.name for p in ranks[0]] != [p.name for p in ranks[1]]:
        raise ValueError('input check requires matching PP rank sets (at least two ranks)')
    for name in ('inputs.bin', 'targets.bin'):
        values = [(rank / name).read_bytes() for group in ranks for rank in group]
        if not values[0] or any(value != values[0] for value in values[1:]):
            raise ValueError(f'PP input/target mismatch: {name}')


def compare(reference, candidate, atol, check_inputs=False):
    """Compare by global names across different PP layouts, with rtol fixed to zero."""
    if not np.isfinite(atol) or atol < 0:
        raise ValueError('invalid atol')
    if check_inputs:
        compare_inputs(reference, candidate)
    left, loss_left = load(reference)
    right, loss_right = load(candidate)
    if left.keys() != right.keys():
        raise ValueError('global parameter sets differ')
    failures = [name for name in left if left[name].shape != right[name].shape
                or not np.allclose(left[name], right[name], atol=atol, rtol=0)]
    if failures or abs(loss_left - loss_right) > atol:
        raise ValueError(f'gradient/loss mismatch: {failures}; losses={loss_left},{loss_right}')
    return {'parameters': len(left), 'loss_reference': loss_left,
            'loss_candidate': loss_right, 'atol': atol, 'rtol': 0, 'status': 'PASS'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--check-inputs', action='store_true', help='Require identical PP batch dumps on all ranks')
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.candidate, args.atol, args.check_inputs), indent=2))


if __name__ == '__main__':
    main()
