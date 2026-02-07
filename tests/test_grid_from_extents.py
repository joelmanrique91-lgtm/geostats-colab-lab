import os
import unittest

import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grid import grid_from_extents


class TestGridFromExtents(unittest.TestCase):
    def test_grid_spec_basic(self):
        df = pd.DataFrame({"x": [0, 100], "y": [0, 100], "z": [0, 50]})
        spec = grid_from_extents(df, "x", "y", "z", dx=10, dy=10, dz=5, pad=0)
        self.assertEqual(spec["nx"], 10)
        self.assertEqual(spec["ny"], 10)
        self.assertEqual(spec["nz"], 10)


if __name__ == "__main__":
    unittest.main()
