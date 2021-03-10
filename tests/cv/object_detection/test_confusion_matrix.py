from typing import List, Optional

import pytest
from contextlib import ExitStack as DoesNotRaise

import numpy as np

from onemetric.cv.object_detection import ConfusionMatrix


@pytest.mark.parametrize(
    "num_classes, true_batches, detection_batches, expected_result, exception",
    [
        (
            10,
            [
                np.array([
                    [0., 0., 1., 1., 1, 1.],
                    [2., 2., 2.5, 2.5, 2, 1.]
                ])
            ],
            [
                np.array([
                    [0., 0., 1., 1., 1],
                    [2., 2., 2.5, 2.5, 2]
                ])
            ],
            None,
            pytest.raises(ValueError)
        ),  # Wrong input shape
        (10, [], [], np.zeros((11, 11)), DoesNotRaise()),  # Initial state of ConfusionMatrix object
        (
            3,
            [
                np.array([
                    [0., 0., 1., 1., 1],
                    [2., 2., 2.5, 2.5, 2]
                ])
            ],
            [
                np.array([
                    [0., 0., 1., 1., 1, 1.],
                    [2., 2., 2.5, 2.5, 2, 1.]
                ])
            ],
            np.array([
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0]
            ]),
            DoesNotRaise()
        ),  # Single image with perfect match
(
            3,
            [
                np.array([
                    [0.1, 0.1, 1.1, 1.1, 1],
                    [2., 2., 2.5, 2.5, 2]
                ])
            ],
            [
                np.array([
                    [0., 0., 1., 1., 1, 1.],
                    [2., 2., 2.5, 2.5, 2, 1.]
                ])
            ],
            np.array([
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0]
            ]),
            DoesNotRaise()
        ),  # Single image with near perfect match
        # (
        #     3,
        #     [
        #         # np.zeros((0, 5)),
        #         np.array([
        #             [10, 10, 11, 11, 1],
        #         ])
        #     ],
        #     [
        #         np.array([
        #             [0., 0.25, 1., 1.25, 1, 0.8],
        #             [1., 1., 2., 2., 2, 0.9],
        #             [0., 0.75, 1., 1.75, 3, 0.9],
        #         ]),
        #     ],
        #     np.ones((4, 4)),
        #     DoesNotRaise()
        # )
    ]
)
def test_confusion_matrix_submit_batch(
    num_classes: int,
    true_batches: List[np.ndarray],
    detection_batches: List[np.ndarray],
    expected_result: Optional[np.ndarray],
    exception: Exception
) -> None:
    with exception:
        confusion_matrix = ConfusionMatrix(num_classes=num_classes)
        for true_batch, detection_batch in zip(true_batches, detection_batches):
            confusion_matrix.submit_batch(true_batch=true_batch, detection_batch=detection_batch)
        np.testing.assert_array_equal(confusion_matrix.matrix, expected_result)
