from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np

from onemetric.cv.object_detection.average_precision import AveragePrecision


@dataclass(frozen=True)
class MeanAveragePrecision:
    value: float
    per_class: List[AveragePrecision]

    @classmethod
    def from_detections(
        cls,
        true_batches: List[np.ndarray],
        detection_batches: List[np.ndarray],
        num_classes: int,
        iou_threshold: float = 0.5
    ) -> MeanAveragePrecision:
        """
        Calculate mean average precision (mAP) metric based on ground-true and detected objects across all images in concerned dataset.

        Args:
            true_batches: `List[np.ndarray]` representing ground-truth objects across all images in concerned dataset. Each element of `true_batches` list describe single image and has `shape = (N, 5)` where `N` is number of ground-truth objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class)`.
            detection_batches: `List[np.ndarray]` representing detected objects across all images in concerned dataset. Each element of `detection_batches` list describe single image and has `shape = (M, 6)` where `M` is number of detected objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class, conf)`.
            num_classes: `int` number of classes detected by model.
            iou_threshold: `float` detection iou  threshold between 0 and 1. Detections with lower iou will be classified as FP.
        """
        per_class = [
            AveragePrecision.from_detections(
                true_batches=true_batches,
                detection_batches=detection_batches,
                class_idx=class_idx,
                iou_threshold=iou_threshold
            )
            for class_idx
            in range(num_classes)
        ]
        values = [ap.value for ap in per_class]
        return cls(value=sum(values) / num_classes, per_class=per_class)
