from enum import Enum
from dataclasses import dataclass


class DimensionFormat(Enum):
    """Dimension format."""

    SPARSE = "s"
    DENSE = "d"


@dataclass
class Dimension:
    """Represents a single dimension of a tensor."""

    size: int
    format: DimensionFormat

    @property
    def is_sparse(self) -> bool:
        return self.format == DimensionFormat.SPARSE

    @property
    def is_dense(self) -> bool:
        return self.format == DimensionFormat.DENSE
