from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from onemetric.const import EPSILON


# Updated version of compute_ap from https://github.com/ultralytics/yolov3


@dataclass(frozen=True)
class AveragePrecision:
    """
    """
    value: float
    recall_values: np.ndarray
    precision_values: np.ndarray
    class_idx: Optional[int] = None
    iou_threshold: Optional[float] = None

    @classmethod
    def from_detections(cls, true_batches: np.ndarray, detection_batches: np.ndarray, class_idx: int, iou_threshold: float = 0.5) -> AveragePrecision:
        """
        Calculate average precision (AP) metric based on ground-true and detected objects.

        Args:
            true_batches: 3d `np.ndarray` representing ground-truth objects. `shape = (I, N, 5)` where `I` is number of images and `N` is number of ground-truth objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class)`.
            detection_batches: `3d np.ndarray` representing detected objects. `shape = (I, M, 6)` where `I` is number of images and `M` is number of detected objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class, conf)`.
        """
        pass

    @classmethod
    def from_precision_recall(cls, recall: np.ndarray, precision: np.ndarray, class_idx: Optional[int] = None, iou_threshold: Optional[float] = None) -> AveragePrecision:
        """
        Calculate average precision (AP) metric based on given precision recall curve.
        """
        recall_values = np.concatenate(([0.], recall, [recall[-1] + EPSILON]))
        precision_values = np.concatenate(([1.], precision, [0.]))
        precision_values = np.flip(np.maximum.accumulate(np.flip(precision_values)))
        i = np.where(recall_values[1:] != recall_values[:-1])[0]
        value = np.sum((recall_values[i + 1] - recall_values[i]) * precision_values[i + 1])
        return AveragePrecision(
            value=value,
            recall_values=recall_values,
            precision_values=precision_values,
            class_idx=class_idx,
            iou_threshold=iou_threshold
        )

    def plot(self) -> None:
        """
        """
        pass

    @staticmethod
    def _process_batch(true_batch: np.ndarray, detection_batches: np.ndarray, class_idx: int, iou_threshold: float = 0.5) -> np.ndarray:
        pass
