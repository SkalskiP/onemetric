from typing import Tuple

import pytest
from contextlib import ExitStack as DoesNotRaise

from onemetric.cv import box_iou


@pytest.mark.parametrize(
    "box_true, box_prediction, expected_result, exc",
    [
        (None, None, None, pytest.raises(ValueError)),
        ((0., 0., 1., 1.), (0., 0., 1., 1.), 1., DoesNotRaise())
    ]
)
def test_box_iou(
    box_true: Tuple[float, float, float, float],
    box_prediction: Tuple[float, float, float, float],
    expected_result: float,
    exc: Exception
):
    with exc:
        result = box_iou(box_true=box_true, box_prediction=box_prediction)
        assert result == expected_result
