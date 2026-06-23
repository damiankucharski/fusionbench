"""fusionbench: a controlled benchmark for stacking / prediction-fusion methods.

The benchmark isolates specific *base-model pathologies* (the issues stacking is
meant to solve) and evaluates fusion methods under a *guaranteed-reliable*
validation environment, so observed differences reflect a method's expressiveness
on a pathology rather than optimizer's-curse / validation-test luck.
"""

from fusionbench.types import Dataset, Task

__all__ = ["Dataset", "Task"]
__version__ = "0.1.0"
