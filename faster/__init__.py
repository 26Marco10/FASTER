"""
FASTER 1.0

A comprehensive text sentiment analysis toolkit supporting
both traditional ML and deep learning models with advanced reporting.
"""

__version__ = "1.0.0"
__author__ = "Marco Recca"

from .core import (
    TextPreprocessor,
    MLModels,
    DLModels,
    FineTuningModels,
    ModelEvaluator,
    load_dataset
)

from .utils import generate_report

__all__ = [
    'TextPreprocessor',
    'MLModels',
    'DLModels',
    'FineTuningModels',
    'ModelEvaluator',
    'load_dataset',
    'generate_report'
]