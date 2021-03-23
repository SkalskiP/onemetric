from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from onemetric.const import EPSILON
from onemetric.cv.utils.iou import box_iou_batch
from onemetric.cv.utils.validators import validate_detections, validate_precision_recall, validate_true_batch, \
    validate_detection_batch


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
    def from_detections(
        cls,
        true_batches: List[np.ndarray],
        detection_batches: List[np.ndarray],
        class_idx: int,
        iou_threshold: float = 0.5
    ) -> AveragePrecision:
        """
        Calculate average precision (AP) metric based on ground-true and detected objects across all images in concerned dataset.

        Args:
            true_batches: `List[np.ndarray]` representing ground-truth objects across all images in concerned dataset. Each element of `true_batches` list describe single image and has `shape = (N, 5)` where `N` is number of ground-truth objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class)`.
            detection_batches: `List[np.ndarray]` representing detected objects across all images in concerned dataset. Each element of `detection_batches` list describe single image and has `shape = (M, 6)` where `M` is number of detected objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class, conf)`.
            class_idx: `int` index of the class for which you want to calculate average precision (AP).
            iou_threshold: `float` detection iou  threshold between 0 and 1. Detections with lower iou will be classified as FP.
        """
        validate_detections(true_batches=true_batches, detection_batches=detection_batches)
        evaluated_detections = np.concatenate([
            AveragePrecision._evaluate_detection_batch(
                true_batch=true_batch,
                detection_batch=detection_batch,
                class_idx=class_idx,
                iou_threshold=iou_threshold
            )
            for true_batch, detection_batch
            in zip(true_batches, detection_batches)
        ])
        evaluated_detections = evaluated_detections[evaluated_detections[:, 0].argsort()[::-1]]
        tp = np.cumsum(evaluated_detections[:, 1])
        all_detections = evaluated_detections.shape[0]
        precision = tp / np.arange(1, all_detections + 1)
        recall = tp / all_detections
        return cls.from_precision_recall(precision=precision, recall=recall, class_idx=class_idx, iou_threshold=iou_threshold)

    @classmethod
    def from_precision_recall(
        cls,
        recall: np.ndarray,
        precision: np.ndarray,
        class_idx: Optional[int] = None,
        iou_threshold: Optional[float] = None
    ) -> AveragePrecision:
        """
        Calculate average precision (AP) metric based on given precision/recall curve.
        """
        validate_precision_recall(precision=precision, recall=recall)
        if precision.shape[0] == 0:
            recall_values = np.array([0., EPSILON])
            precision_values = np.array([1., 0.])
        else:
            recall_values = np.concatenate(([0.], recall, [recall[-1] + EPSILON]))
            precision_values = np.concatenate(([1.], precision, [0.]))

        precision_values = np.flip(np.maximum.accumulate(np.flip(precision_values)))
        i = np.where(recall_values[1:] != recall_values[:-1])[0]
        value = np.sum((recall_values[i + 1] - recall_values[i]) * precision_values[i + 1])
        return cls(
            value=value,
            recall_values=recall_values,
            precision_values=precision_values,
            class_idx=class_idx,
            iou_threshold=iou_threshold
        )

    @staticmethod
    def _evaluate_detection_batch(
        true_batch: np.ndarray,
        detection_batch: np.ndarray,
        class_idx: int,
        iou_threshold: float
    ) -> np.ndarray:
        validate_true_batch(true_batch=true_batch)
        validate_detection_batch(detection_batch=detection_batch)

        true_batch_filtered = true_batch[true_batch[:, 4] == class_idx]
        detection_batch_filtered = detection_batch[detection_batch[:, 4] == class_idx]

        # confidence, tp, fp
        result_matrix = np.zeros((detection_batch_filtered.shape[0], 3))

        true_boxes = true_batch_filtered[:, :4]
        detection_boxes = detection_batch_filtered[:, :4]
        detection_conf = detection_batch_filtered[:, 5]
        iou_batch = box_iou_batch(boxes_true=true_boxes, boxes_detection=detection_boxes)
        matched_idx = np.asarray(iou_batch > iou_threshold).nonzero()

        if matched_idx[0].shape[0]:
            matches = np.stack((matched_idx[0], matched_idx[1], iou_batch[matched_idx]), axis=1)
            matches = AveragePrecision._drop_extra_matches(matches=matches)
        else:
            matches = np.zeros((0, 3))

        matched_true_class_idx, matched_detection_class_idx, _ = matches.transpose().astype(np.int16)

        for i, conf in enumerate(detection_conf):
            if any(matched_detection_class_idx == i):
                result_matrix[i] = np.array([conf, 1, 0])
            else:
                result_matrix[i] = np.array([conf, 0, 1])

        return result_matrix

    @staticmethod
    def _drop_extra_matches(matches: np.ndarray) -> np.ndarray:
        if matches.shape[0] > 0:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        return matches
