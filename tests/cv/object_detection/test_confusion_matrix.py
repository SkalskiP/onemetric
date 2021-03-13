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
        (
            10,
            [
                np.array([
                    [0., 0., 1., 1., 10],
                ])
            ],
            [
                np.array([
                    [0., 0., 1., 1., 1, 1.],
                ])
            ],
            None,
            pytest.raises(ValueError)
        ),  # Wrong ground-truth class index
        (
            10,
            [
                np.array([
                    [0., 0., 1., 1., 1],
                ])
            ],
            [
                np.array([
                    [0., 0., 1., 1., 10, 1.],
                ])
            ],
            None,
            pytest.raises(ValueError)
        ),  # Wrong detection class index
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
        (
            3,
            [
                np.array([
                    [0., 0., 1., 1., 0],
                    [2., 2., 2.5, 2.5, 1]
                ])
            ],
            [
                np.array([
                    [0., 0., 1., 1., 1, 1.],
                    [2., 2., 2.5, 2.5, 2, 1.]
                ])
            ],
            np.array([
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]),
            DoesNotRaise()
        ),  # Correct boxes but mixed classes
        (
            3,
            [
                np.array([
                    [10, 10, 11, 11, 1],
                ])
            ],
            [
                np.zeros((0, 6)),
            ],
            np.array([
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]),
            DoesNotRaise()
        ),  # Object was not detected
        (
            3,
            [
                np.zeros((0, 5))
            ],
            [
                np.array([
                    [0., 0., 1., 1., 1, 1.]
                ]),
            ],
            np.array([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 0]
            ]),
            DoesNotRaise()
        ),  # Object was falsely detected
        (
            3,
            [
                np.array([
                    [0., 0., 1., 1., 1],
                ])
            ],
            [
                np.array([
                    [0., 0., 1., 1., 1, 0.8],
                    [0., 0., 1., 1., 1, 0.9],
                    [0., 0., 1., 1., 1, 0.9],
                ]),
            ],
            np.array([
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 2, 0, 0]
            ]),
            DoesNotRaise()
        ),  # Multiple detections of the same object
        (
            3,
            [
                np.array([
                    [0.0, 0.0, 3.0, 3.0, 0],  # [0] detected
                    [2.0, 2.0, 5.0, 5.0, 1],  # [1] undetected - FN
                    [6.0, 1.0, 8.0, 3.0, 2],  # [2] correct detection with incorrect class
                ])
            ],
            [
                np.array([
                    [0.0, 0.0, 3.0, 3.0, 0, 0.9],  # correct detection of [0]
                    [0.1, 0.1, 3.0, 3.0, 0, 0.9],  # additional detection of [0] - FP
                    [6.0, 1.0, 8.0, 3.0, 1, 0.8],  # correct detection with incorrect class
                    [1.0, 6.0, 2.0, 7.0, 1, 0.8],  # incorrect detection - FP
                ])
            ],
            np.array([
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [1, 1, 0, 0]
            ]),
            DoesNotRaise()
        ),  # General use case
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
        # confusion_matrix.plot('/Users/piotrskalski/Downloads/onemetric_test/matrix.png', ['a', 'b', 'c'])
        np.testing.assert_array_equal(confusion_matrix.matrix, expected_result)
