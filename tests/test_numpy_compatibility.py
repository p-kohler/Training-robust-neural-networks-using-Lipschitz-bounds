"""Check ragged layer weights and their MATLAB-file representation."""
import ast
from io import BytesIO
from pathlib import Path
import unittest

import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[1]


class NumpyCompatibilityTest(unittest.TestCase):
    def test_weight_conversion_and_mat_roundtrip(self):
        for dims in ([2, 10, 10, 3], [196, 50, 10],
                     [196, 50, 50, 10], [784, 50, 50, 10]):
            with self.subTest(dims=dims):
                weights = [np.arange(a*b, dtype=np.float64).reshape(b, a)
                           for a, b in zip(dims[:-1], dims[1:])]
                biases = [np.arange(b, dtype=np.float64) for b in dims[1:]]
                cells = np.array(weights, dtype=object)
                bias_cells = np.array(biases, dtype=object)
                for i, weight in enumerate(weights):
                    self.assertEqual(cells[i].tolist(), weight.tolist())
                    self.assertEqual(bias_cells[i].tolist(), biases[i].tolist())
                stream = BytesIO()
                savemat(stream, {'weights': cells})
                stream.seek(0)
                restored = loadmat(stream)['weights']
                self.assertEqual(restored.shape, (1, len(weights)))
                for i, weight in enumerate(weights):
                    np.testing.assert_array_equal(restored[0, i], weight)

    def test_no_removed_alias_in_python_sources(self):
        for path in ROOT.glob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    self.assertFalse(node.value.id == 'np' and node.attr == 'object', str(path))


if __name__ == '__main__':
    unittest.main()
