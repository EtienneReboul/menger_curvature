from dataclasses import dataclass, field

import numpy as np


@dataclass
class DummyData:
    """A data class containing dummy data for testing and numba compiling purposes."""

    frame: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [13.31, 34.22, 34.36],
                [16.89, 33.47, 35.28],
                [20.4, 34.65, 34.76],
                [23.99, 33.21, 34.96],
                [27.52, 34.44, 34.73],
                [31.27, 33.34, 35.16],
                [34.95, 34.55, 34.84],
                [38.57, 33.49, 35.07],
                [42.11, 34.67, 34.64],
                [45.72, 33.37, 34.84],
                [49.49, 34.3, 34.62],
                [53.24, 33.33, 34.85],
                [56.58, 35.18, 34.74],
            ],
            dtype=np.float32,
        )
    )
