from typing import Tuple

import pytest
from contextlib import ExitStack as DoesNotRaise

from onemetric.cv import box_iou, mask_iou

import numpy as np


@pytest.mark.parametrize(
    "box_true, box_prediction, expected_result, exc",
    [
        (None, None, None, pytest.raises(ValueError)),
        ((0., 0., 1.), (0., 0., 1., 1.), None, pytest.raises(ValueError)),
        ((0., 0., 1., 1.), (0., 0., 1.), None, pytest.raises(ValueError)),
        ([0., 0., 1., 1.], [0., 0., 1., 1.], None, pytest.raises(ValueError)),
        ((0., 0., 1., 1.), (0., 1., 1., 2.), 0., DoesNotRaise()),
        ((0, 1., 1., 2.), (0., 0., 1., 1.), 0., DoesNotRaise()),
        ((0., 0., 1., 1.), (1., 0., 2., 1.), 0., DoesNotRaise()),
        ((1., 0., 2., 1.), (0., 0., 1., 1.), 0., DoesNotRaise()),
        ((0., 0., 1., 1.), (0.25, 0., 1.25, 1.), 0.6, DoesNotRaise()),
        ((0.25, 0., 1.25, 1.), (0., 0., 1., 1.), 0.6, DoesNotRaise()),
        ((0., 0., 1., 1.), (0., 0.25, 1., 1.25), 0.6, DoesNotRaise()),
        ((0., 0.25, 1., 1.25), (0., 0., 1., 1.), 0.6, DoesNotRaise()),
        ((0., 0., 1., 1.), (0., 0., 1., 1.), 1., DoesNotRaise()),
        ((0., 0., 3., 3.), (1., 1., 2., 2.), 1/9, DoesNotRaise()),
        ((1., 1., 2., 2.), (0., 0., 3., 3.), 1/9, DoesNotRaise())
    ]
)
def test_box_iou(
    box_true: Tuple[float, float, float, float],
    box_prediction: Tuple[float, float, float, float],
    expected_result: float,
    exc: Exception
) -> None:
    with exc:
        result = box_iou(box_true=box_true, box_prediction=box_prediction)
        assert result == expected_result


QUARTER_MASK = np.zeros((10, 10)).astype('uint8')
QUARTER_MASK[0:5, 0:5] = 1


@pytest.mark.parametrize(
    "mask_true, mask_prediction, expected_result, exc",
    [
        (None, None, None, pytest.raises(ValueError)),
        (np.zeros((10, 10)).astype('uint8'), np.zeros((20, 20)).astype('uint8'), None, pytest.raises(ValueError)),
        (np.zeros((20, 20)).astype('uint8'), np.zeros((10, 10)).astype('uint8'), None, pytest.raises(ValueError)),
        (np.ones((10, 10)).astype('int16'), np.zeros((10, 10)).astype('int16'), None, pytest.raises(ValueError)),
        (np.ones((10, 10)).astype('uint8') * 2, np.zeros((10, 10)).astype('uint8'), 0., pytest.raises(ValueError)),
        (np.ones((10, 10)).astype('uint8'), np.zeros((10, 10)).astype('uint8'), 0., DoesNotRaise()),
        (np.zeros((10, 10)).astype('uint8'), np.ones((10, 10)).astype('uint8'), 0., DoesNotRaise()),
        (np.zeros((10, 10)).astype('uint8'), np.zeros((10, 10)).astype('uint8'), None, DoesNotRaise()),
        (np.ones((10, 10)).astype('uint8'), np.ones((10, 10)).astype('uint8'), 1., DoesNotRaise()),
        (np.ones((10, 10)).astype('uint8'), QUARTER_MASK, 0.25, DoesNotRaise())
    ]
)
def test_mask_iou(mask_true: np.array, mask_prediction: np.array, expected_result: float, exc: Exception) -> None:
    with exc:
        result = mask_iou(mask_true=mask_true, mask_prediction=mask_prediction)
        assert result == expected_result
