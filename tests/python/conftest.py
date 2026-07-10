"""
Pytest configuration file for CLI inference tests.
Defines command-line options for testing different hardware modules (DPU, DMA, SHAVE).
"""

import pytest

def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--hw_module",
        action="store",
        default="DPU",
        choices=["DPU", "DMA_2_7", "DMA_4_0", "SHAVE"],
        help="Hardware module to test: DPU, DMA_2_7, DMA_4_0, or SHAVE"
    )
    parser.addoption(
        "--dpu-csv",
        action="store",
        default="test_CLI_dpu.csv",
        help="CSV file containing DPU test workloads"
    )
    parser.addoption(
        "--dma-csv",
        action="store",
        default="test_CLI_dma.csv",
        help="CSV file containing DMA test workloads (for both DMA_2_7 and DMA_4_0)"
    )
    parser.addoption(
        "--shave-csv",
        action="store",
        default="test_CLI_shave.csv",
        help="CSV file containing SHAVE test workloads"
    )
