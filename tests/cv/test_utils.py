from contextlib import ExitStack as DoesNotRaise
from typing import Optional, Tuple

import numpy as np
import pytest
import math

from onemetric.const import EPSILON
from onemetric.cv.utils import compute_ap


@pytest.mark.parametrize(
    "recall, precision, expected_result, exception",
    [
        (
            np.array([0.0]),
            np.array([0.0]),
            (
                0.,
                np.array([0.0, 0.0, EPSILON]),
                np.array([1.0, 0.0, 0.0])
            ),
            DoesNotRaise()
        ),
        (
            np.array([0.0, 1.0]),
            np.array([1.0, 0.5]),
            (
                0.5,
                np.array([0.0, 0.0, 1.0, 1.0 + EPSILON]),
                np.array([1.0, 1.0, 0.5, 0.0])
            ),
            DoesNotRaise()
        ),
        (
            np.array([0.0, 0.5, 1.0]),
            np.array([1.0, 0.6, 0.4]),
            (
                0.5,
                np.array([0.0, 0.0, 0.5, 1.0, 1.0 + EPSILON]),
                np.array([1.0, 1.0, 0.6, 0.4, 0.0])
            ),
            DoesNotRaise()
        ),
        (
            np.array([0.0, 0.2, 0.2, 0.8, 0.8, 1.0]),
            np.array([0.7, 0.8, 0.4, 0.5, 0.1, 0.2]),
            (
                0.5,
                np.array([0.0, 0.0, 0.2, 0.2, 0.8, 0.8, 1.0, 1.0 + EPSILON]),
                np.array([1.0, 0.8, 0.8, 0.5, 0.5, 0.2, 0.2, 0.0])
            ),
            DoesNotRaise()
        )
    ]
)
def test_compute_ap(
    recall: np.ndarray,
    precision: np.ndarray,
    expected_result: Optional[Tuple[float, np.ndarray, np.ndarray]],
    exception: Exception
) -> None:
    with exception:
        result = compute_ap(recall=recall, precision=precision)
        assert len(result) == 3
        assert math.isclose(result[0], expected_result[0])
        np.testing.assert_array_equal(result[1], expected_result[1])
        np.testing.assert_array_equal(result[2], expected_result[2])
