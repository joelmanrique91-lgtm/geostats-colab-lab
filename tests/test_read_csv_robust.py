import os
import tempfile
import unittest

import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.io_utils import read_csv_robust


class TestReadCsvRobust(unittest.TestCase):
    def test_read_csv_robust_semicolon(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.csv")
            df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
            df.to_csv(path, sep=";", index=False)

            out = read_csv_robust(path)
            self.assertEqual(out.shape, (2, 2))
            self.assertListEqual(list(out.columns), ["A", "B"])


if __name__ == "__main__":
    unittest.main()
