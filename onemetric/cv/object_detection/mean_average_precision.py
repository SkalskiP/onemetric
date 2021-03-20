from typing import Optional, List, Tuple

import numpy as np


class MeanAveragePrecision:
    """
    Calculate and visualize mean average precision (mAP) of Object Detection model.
    """

    def __init__(self, num_classes: int, iou_thresholds: Optional[List[float]] = None) -> None:
        if iou_thresholds is None:
            self._iou_thresholds = [0.5]
        self._num_classes = num_classes
        self.__batches: List[Tuple[np.ndarray, np.ndarray]] = []

    def submit_batch(self, true_batch: np.ndarray, detection_batch: np.ndarray) -> None:
        """
        Args:
            true_batch: 2d `np.ndarray` representing ground-truth objects. `shape = (N, 5)` where `N` is number of annotated objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class)`.
            detection_batch: `2d np.ndarray` representing detected objects. `shape = (M, 6)` where `M` is number of detected objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class, conf)`.
        """
        _validate_true_batch(true_batch=true_batch)
        _validate_detection_batch(detection_batch=detection_batch)

    def calculate(self) -> None:
        pass


def _validate_true_batch(true_batch: np.ndarray):
    if type(true_batch) != np.ndarray or len(true_batch.shape) != 2 or true_batch.shape[1] != 5:
        raise ValueError(
            f"True batch must be defined as 2d np.array with (N, 5) shape, where N is number of is number of "
            f"annotated objects and each row is in (x_min, y_min, x_max, y_max, class) format. {true_batch} given."
        )


def _validate_detection_batch(detection_batch: np.ndarray):
    if type(detection_batch) != np.ndarray or len(detection_batch.shape) != 2 or detection_batch.shape[1] != 6:
        raise ValueError(
            f"Detected batch must be defined as 2d np.array with (M, 6) shape, where M is number of is number of "
            f"detected objects and each row is in (x_min, y_min, x_max, y_max, class, conf) format. "
            f"{detection_batch} given."
        )
