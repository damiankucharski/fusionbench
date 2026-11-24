"""fusionbench: a controlled benchmark for stacking / prediction-fusion methods.

The benchmark isolates base-model failure modes and measures fusion quality and
validation-to-test transfer under fixed or deliberately changed distributions.
"""

from fusionbench.types import Dataset, Task

__all__ = ["Dataset", "Task"]
__version__ = "0.1.0"
