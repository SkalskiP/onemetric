import math
from contextlib import ExitStack as DoesNotRaise
from typing import Optional, List

import numpy as np
import pytest

from onemetric.const import EPSILON
from onemetric.cv.object_detection.average_precision import AveragePrecision


@pytest.mark.parametrize(
    "recall, precision, expected_result, exception",
    [
        (
            np.array([0.0, 1.0]),
            np.array([1.0]),
            None,
            pytest.raises(ValueError)
        ),  # Precision and recall arrays have unequal length
        (
            np.array([0.0]),
            np.array([1.0, 0.5]),
            None,
            pytest.raises(ValueError)
        ),  # Precision and recall arrays have unequal length
        (
            np.array([]),
            np.array([]),
            AveragePrecision(
                value=0.,
                recall_values=np.array([0.0, EPSILON]),
                precision_values=np.array([1.0, 0.0])
            ),
            DoesNotRaise()
        ),  # Precision and recall arrays are of insufficient length
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
            np.array([1.0]),
            np.array([1.0]),
            AveragePrecision(
                value=1.,
                recall_values=np.array([0.0, 1.0, 1.0 + EPSILON]),
                precision_values=np.array([1.0, 1.0, 0.0])
            ),
            DoesNotRaise()
        ),
        (
            np.array([0.0, 1.0]),
            np.array([0.0, 0.5]),
            AveragePrecision(
                value=0.5,
                recall_values=np.array([0.0, 0.0, 1.0, 1.0 + EPSILON]),
                precision_values=np.array([1.0, 0.5, 0.5, 0.0])
            ),
            DoesNotRaise()
        ),
        (
            np.array([0.5, 0.5]),
            np.array([1.0, 0.5]),
            AveragePrecision(
                value=0.5,
                recall_values=np.array([0.0, 0.5, 0.5, 0.5 + EPSILON]),
                precision_values=np.array([1.0, 1.0, 0.5, 0.0])
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


@pytest.mark.parametrize(
    "true_batch, detection_batch, class_idx, iou_threshold, expected_result, exception",
    [
        (
            np.array([
                [0.0, 0.0, 1.0, 1.0, 1],
            ]),
            np.zeros((0, 6)),
            1,
            0.5,
            np.zeros((0, 3)),
            DoesNotRaise()
        ),  # FN; object undetected
        (
            np.zeros((0, 5)),
            np.array([
                [0.0, 0.0, 1.0, 1.0, 1, 0.9],
            ]),
            1,
            0.5,
            np.array([
                [0.9, 0, 1],
            ]),
            DoesNotRaise()
        ),  # FP object detected; object belongs to selected class
        (
            np.array([
                [0.0, 0.0, 1.0, 1.0, 1],
            ]),
            np.array([
                [0.0, 0.0, 1.0, 1.0, 1, 0.9],
            ]),
            1,
            0.5,
            np.array([
                [0.9, 1, 0],
            ]),
            DoesNotRaise()
        ),  # TP object detected; object belongs to selected class
        (
            np.array([
                [0.0, 0.0, 1.0, 1.0, 1],
            ]),
            np.array([
                [0.0, 0.0, 1.0, 1.0, 1, 0.9],
            ]),
            2,
            0.5,
            np.zeros((0, 3)),
            DoesNotRaise()
        ),  # TP object detected; object does not belong to selected class
        (
            np.array([
                [0.0, 0.0, 1.0, 1.0, 1],
            ]),
            np.array([
                [0.0, 0.0, 1.0, 1.0, 1, 0.9],
                [0.1, 0.1, 1.1, 1.1, 1, 0.8],
            ]),
            1,
            0.5,
            np.array([
                [0.9, 1, 0],
                [0.8, 0, 1],
            ]),
            DoesNotRaise()
        ),  # Multiple detections of the same class
        (
            np.array([
                [0.0, 0.0, 1.0, 1.0, 1],
            ]),
            np.array([
                [0.5, 0.5, 1.5, 1.5, 1, 0.9],
            ]),
            1,
            0.5,
            np.array([
                [0.9, 0, 1],
            ]),
            DoesNotRaise()
        ),  # FP; IoU lower than threshold
        (
            np.array([
                [0.0, 0.0, 3.0, 3.0, 1],
                [2.0, 2.0, 5.0, 5.0, 1],
                [6.0, 1.0, 8.0, 3.0, 2],
            ]),
            np.array([
                [0.0, 0.0, 3.0, 3.0, 1, 0.9],
                [0.1, 0.1, 3.0, 3.0, 0, 0.9],
                [6.0, 1.0, 8.0, 3.0, 1, 0.8],
                [1.0, 6.0, 2.0, 7.0, 1, 0.8],
            ]),
            1,
            0.5,
            np.array([
                [0.9, 1, 0],
                [0.8, 0, 1],
                [0.8, 0, 1],
            ]),
            DoesNotRaise()
        ),  # General use case
    ]
)
def test_evaluate_detection_batch(
    true_batch: np.ndarray,
    detection_batch: np.ndarray,
    class_idx: int,
    iou_threshold: float,
    expected_result: Optional[np.ndarray],
    exception: Exception
) -> None:
    with exception:
        result = AveragePrecision._evaluate_detection_batch(
            true_batch=true_batch,
            detection_batch=detection_batch,
            class_idx=class_idx,
            iou_threshold=iou_threshold
        )
        np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    "true_batches, detection_batches, class_idx, iou_threshold, expected_result, exception",
    [
        (
            np.array([]),
            np.array([]),
            1,
            0.5,
            None,
            pytest.raises(ValueError)
        ),  # Incorrect input data format
        (
            [
                np.array([
                    [0.0, 0.0, 1.0, 1.0, 1],
                ]),
            ],
            [],
            1,
            0.5,
            None,
            pytest.raises(ValueError)
        ),  # Unequal length of true_batches and detection_batches
        (
            [],
            [
                np.array([
                    [0.0, 0.0, 1.0, 1.0, 1, 0.9],
                ]),
            ],
            1,
            0.5,
            None,
            pytest.raises(ValueError)
        ),  # Unequal length of true_batches and detection_batches
        (
            [
                np.array([
                    [0.0, 0.0, 1.0, 1.0, 1, 0.9],
                ]),
            ],
            [
                np.array([
                    [0.0, 0.0, 1.0, 1.0, 1],
                ]),
            ],
            1,
            0.5,
            None,
            pytest.raises(ValueError)
        ),  # Incorrect batch shape
        (
            [
                np.array([
                    [0.0, 0.0, 1.0, 1.0, 1],
                ]),
            ],
            [
                np.zeros((0, 6)),
            ],
            1,
            0.5,
            AveragePrecision(
                value=0.,
                recall_values=np.array([0.0, EPSILON]),
                precision_values=np.array([1.0, 0.0])
            ),
            DoesNotRaise()
        ),  # Single image with no detections found
        (
            [
                np.array([
                    [0.0, 0.0, 1.0, 1.0, 1],
                ]),
            ],
            [
                np.array([
                    [0.0, 0.0, 1.0, 1.0, 1, 0.9],
                ]),
            ],
            1,
            0.5,
            AveragePrecision(
                value=1.,
                recall_values=np.array([0.0, 1.0, 1.0 + EPSILON]),
                precision_values=np.array([1.0, 1.0, 0.0])
            ),
            DoesNotRaise()
        ),  # Single image with single detection found
    ]
)
def test_from_detections(
    true_batches: List[np.ndarray],
    detection_batches: List[np.ndarray],
    class_idx: int,
    iou_threshold: float,
    expected_result: Optional[AveragePrecision],
    exception: Exception
) -> None:
    with exception:
        result = AveragePrecision.from_detections(
            true_batches=true_batches,
            detection_batches=detection_batches,
            class_idx=class_idx,
            iou_threshold=iou_threshold
        )
        assert math.isclose(result.value, expected_result.value)
        np.testing.assert_array_equal(result.recall_values, expected_result.recall_values)
        np.testing.assert_array_equal(result.precision_values, expected_result.precision_values)
