from typing import List, Optional

import pytest
from contextlib import ExitStack as DoesNotRaise

import numpy as np


# @pytest.mark.parametrize(
#     "true_batches, detection_batches, expected_result, exception",
#     [
#         ()
#     ]
# )
# def test_confusion_matrix_submit_batch(
#     true_batches: List[np.ndarray],
#     detection_batches: List[np.ndarray],
#     expected_result: Optional[np.ndarray],
#     exception: Exception
# ) -> None:
#     pass
