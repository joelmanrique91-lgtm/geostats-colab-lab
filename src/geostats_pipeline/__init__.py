"""Pipeline geostatístico con utilidades de bloques y kriging."""

from .block_model import discretize_block, discretize_blocks, block_covariance
from .kriging import SearchParameters, ordinary_kriging, select_neighbors

__all__ = [
    "discretize_block",
    "discretize_blocks",
    "block_covariance",
    "SearchParameters",
    "ordinary_kriging",
    "select_neighbors",
]
