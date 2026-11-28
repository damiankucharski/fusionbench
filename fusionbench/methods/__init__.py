"""Fusion methods under test."""

from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.base import Method
from fusionbench.methods.okapi import OKAPIMethod
from fusionbench.methods.registry import DEFAULT_METHODS, FACTORIES, default_methods
from fusionbench.methods.stacking import StackingLR

__all__ = [
    "DEFAULT_METHODS",
    "FACTORIES",
    "Method",
    "OKAPIMethod",
    "SimpleAverage",
    "StackingLR",
    "TopN",
    "default_methods",
]
