from contextlib import ExitStack as DoesNotRaise
from typing import Optional

import numpy as np
import pytest
import math

from onemetric.const import EPSILON
from onemetric.cv.object_detection.average_precision import AveragePrecision


@pytest.mark.parametrize(
    "recall, precision, expected_result, exception",
    [
        (
            np.array([0.0]),
            np.array([0.0]),
            AveragePrecision(
                value=0.,
                recall_values=np.array([0.0, 0.0, EPSILON]),
                precision_values=np.array([1.0, 0.0, 0.0])
            ),
            DoesNotRaise()
        ),
        (
            np.array([0.0, 1.0]),
            np.array([1.0, 0.5]),
            AveragePrecision(
                value=0.5,
                recall_values=np.array([0.0, 0.0, 1.0, 1.0 + EPSILON]),
                precision_values=np.array([1.0, 1.0, 0.5, 0.0])
            ),
            DoesNotRaise()
        ),
        (
            np.array([0.0, 0.5, 1.0]),
            np.array([1.0, 0.6, 0.4]),
            AveragePrecision(
                value=0.5,
                recall_values=np.array([0.0, 0.0, 0.5, 1.0, 1.0 + EPSILON]),
                precision_values=np.array([1.0, 1.0, 0.6, 0.4, 0.0])
            ),
            DoesNotRaise()
        ),
        (
            np.array([0.0, 0.2, 0.2, 0.8, 0.8, 1.0]),
            np.array([0.7, 0.8, 0.4, 0.5, 0.1, 0.2]),
            AveragePrecision(
                value=0.5,
                recall_values=np.array([0.0, 0.0, 0.2, 0.2, 0.8, 0.8, 1.0, 1.0 + EPSILON]),
                precision_values=np.array([1.0, 0.8, 0.8, 0.5, 0.5, 0.2, 0.2, 0.0])
            ),
            DoesNotRaise()
        )
    ]
)
def test_from_precision_recall(
    recall: np.ndarray,
    precision: np.ndarray,
    expected_result: Optional[AveragePrecision],
    exception: Exception
) -> None:
    with exception:
        result = AveragePrecision.from_precision_recall(recall=recall, precision=precision)
        assert math.isclose(result.value, expected_result.value)
        np.testing.assert_array_equal(result.recall_values, expected_result.recall_values)
        np.testing.assert_array_equal(result.precision_values, expected_result.precision_values)
