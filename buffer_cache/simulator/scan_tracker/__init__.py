"""
Scan Tracker - PBM-style scan registration and tracking.

This is the key component that enables predictive buffer management.
"""
from .scan_entry import ScanEntry, ScanType
from .scan_registry import ScanRegistry
from .next_access_estimator import NextAccessEstimator

__all__ = [
    'ScanEntry',
    'ScanType', 
    'ScanRegistry',
    'NextAccessEstimator',
]

