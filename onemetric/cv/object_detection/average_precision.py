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

    @classmethod
    def from_detections(cls, true_batches: np.ndarray, detection_batches: np.ndarray, class_idx: int, iou_threshold: float = 0.5) -> AveragePrecision:
        """
        """
        pass

    @classmethod
    def from_precision_recall(cls, recall: np.ndarray, precision: np.ndarray) -> AveragePrecision:
        """
        """
        recall_values = np.concatenate(([0.], recall, [recall[-1] + EPSILON]))
        precision_values = np.concatenate(([1.], precision, [0.]))
        precision_values = np.flip(np.maximum.accumulate(np.flip(precision_values)))
        i = np.where(recall_values[1:] != recall_values[:-1])[0]
        value = np.sum((recall_values[i + 1] - recall_values[i]) * precision_values[i + 1])
        return AveragePrecision(value=value, recall_values=recall_values, precision_values=precision_values)

    def plot(self) -> None:
        """
        """
        pass
