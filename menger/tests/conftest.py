"""
Pytest configuration file for menger tests.
"""

import matplotlib
import matplotlib.pyplot as plt
import pytest

# Set matplotlib to use non-interactive backend for all tests
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def reset_matplotlib():
    """Reset matplotlib settings after each test."""
    yield
    # Clean up any figures after each test to prevent memory leaks
    plt.close("all")
