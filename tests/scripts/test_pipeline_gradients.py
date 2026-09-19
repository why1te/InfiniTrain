"""CPU regression checks for global parameter-gradient comparison."""
import importlib.util
import sys
sys.dont_write_bytecode = True
import tempfile
import unittest
from pathlib import Path
import numpy as np


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[2] / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


gradients = module('compare_pipeline_gradients')


class DiagnosticScriptsTest(unittest.TestCase):
    def dump(self, root, rank, names, loss=False):
        path = root / f'rank-{rank}'
        path.mkdir(parents=True)
        (path / 'parameters.tsv').write_text('name\tshape\tstatus\n' + ''.join(f'{n}\t2,\tOK\n' for n in names))
        for name in names:
            np.save(path / (name + '.npy'), np.array([1, 2], dtype=np.float32))
        if loss:
            (path / 'loss.txt').write_text('1.25')
        return path

    def test_global_names_allow_different_rank_partition(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / 'a', Path(d) / 'b'
            self.dump(a, 0, ['layer.0', 'layer.1'], True)
            self.dump(b, 0, ['layer.0'])
            self.dump(b, 1, ['layer.1'], True)
            self.assertEqual(gradients.compare(a, b, 1e-5)['parameters'], 2)
            np.save(b / 'rank-1/layer.1.npy', np.array([1, 3], dtype=np.float32))
            with self.assertRaises(ValueError): gradients.compare(a, b, 1e-5)

    def test_input_check_rejects_rank_mismatch_and_different_batches(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / 'a', Path(d) / 'b'
            for root in (a, b):
                for rank in range(2):
                    path = self.dump(root, rank, [f'layer.{rank}'], rank == 1)
                    (path / 'inputs.bin').write_bytes(b'3\n1,2,\ninput')
                    (path / 'targets.bin').write_bytes(b'3\n1,2,\ntarget')
            self.assertEqual(gradients.compare(a, b, 1e-5, check_inputs=True)['status'], 'PASS')
            for name in ('inputs.bin', 'targets.bin'):
                path = b / 'rank-1' / name
                original = path.read_bytes()
                path.write_bytes(original + b'different')
                with self.assertRaises(ValueError): gradients.compare(a, b, 1e-5, check_inputs=True)
                path.unlink()
                with self.assertRaises(FileNotFoundError): gradients.compare(a, b, 1e-5, check_inputs=True)
                path.write_bytes(original)
            (b / 'rank-2').mkdir()
            with self.assertRaises(ValueError): gradients.compare(a, b, 1e-5, check_inputs=True)

    def test_missing_extra_duplicate_and_nonfinite_fail(self):
        for mode in ['missing', 'extra', 'duplicate', 'nan', 'loss', 'multiple_losses', 'shape', 'dtype']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                p = self.dump(root, 0, ['x'], True)
                if mode == 'missing': (p / 'parameters.tsv').write_text('name\tshape\tstatus\nx\t2,\tMISSING\n')
                if mode == 'extra': np.save(p / 'extra.npy', np.ones(2, dtype=np.float32))
                if mode == 'duplicate': self.dump(root, 1, ['x'])
                if mode == 'nan': np.save(p / 'x.npy', np.array([np.nan, 1], dtype=np.float32))
                if mode == 'loss': (p / 'loss.txt').unlink()
                if mode == 'multiple_losses': self.dump(root, 1, ['y'], True)
                if mode == 'shape': np.save(p / 'x.npy', np.ones((1, 2), dtype=np.float32))
                if mode == 'dtype': np.save(p / 'x.npy', np.ones(2, dtype=np.float64))
                with self.assertRaises(ValueError): gradients.load(root)


if __name__ == '__main__':
    unittest.main()
