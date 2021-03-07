from typing import Tuple, Optional

import numpy as np


def box_iou(
        box_true: Tuple[float, float, float, float],
        box_prediction: Tuple[float, float, float, float]
) -> Optional[float]:
    """
    Compute Intersection over Union of two bounding boxes - box_true and box_prediction. Both boxes are expected to be
    tuples in (x_min, y_min, x_max, y_max) format.

    Args:
        box_true: Tuple representing ground-truth bounding boxes.
        box_prediction: Tuple representing prediction bounding boxes.

    Returns:
        iou: Float value between 0 and 1. None if union is equal to 0.
    """
    _validate_box(box=box_true)
    _validate_box(box=box_prediction)

    x_min_true, y_min_true, x_max_true, y_max_true = box_true
    x_min_prediction, y_min_prediction, x_max_prediction, y_max_prediction = box_prediction

    x_min = max(x_min_true, x_min_prediction)
    y_min = max(y_min_true, y_min_prediction)
    x_max = min(x_max_true, x_max_prediction)
    y_max = min(y_max_true, y_max_prediction)

    area_intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    area_true = (x_max_true - x_min_true) * (y_max_true - y_min_true)
    area_prediction = (x_max_prediction - x_min_prediction) * (y_max_prediction - y_min_prediction)
    area_union = area_true + area_prediction - area_intersection

    if area_union == 0:
        return None

    return area_intersection / area_union


def box_iou_batch(boxes_true: np.ndarray, boxes_prediction: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union of two sets of bounding boxes - boxes_true and boxes_prediction. Both sets of boxes
    are expected to be in (x_min, y_min, x_max, y_max) format. Updated version of
    https://github.com/kaanakan/object_detection_confusion_matrix

    Args:
        boxes_true: 2d np.ndarray representing ground-truth boxes. shape = (N, 4) where N is number of objects.
        boxes_prediction: 2d np.ndarray representing prediction boxes. shape = (M, 4) where M is number of objects.

    Returns:
        iou: 2d np.ndarray representing pairwise Intersection over Union of boxes_true and boxes_prediction.
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_true = box_area(boxes_true.T)
    area_prediction = box_area(boxes_prediction.T)

    top_left = np.maximum(boxes_true[:, None, :2], boxes_true[:, :2])
    bottom_right = np.minimum(boxes_prediction[:, None, 2:], boxes_prediction[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    return area_inter / (area_true[:, None] + area_prediction - area_inter)


def mask_iou(mask_true: np.ndarray, mask_prediction: np.ndarray) -> Optional[float]:
    """
    Compute Intersection over Union of two masks - mask_true and mask_prediction. Shapes of mask_true and
    mask_prediction should be identical. Both arrays are expected to be np.uint8 type and contain binary values (0 or 1).

    Args:
        mask_true: 2d np.ndarray representing ground-truth mask.
        mask_prediction: 2d np.ndarray representing prediction mask.

    Returns:
        iou: Float value between 0 and 1. None if union is equal to 0.
    """
    _validate_mask(mask=mask_true)
    _validate_mask(mask=mask_prediction)

    if mask_true.shape != mask_prediction.shape:
        raise ValueError(f"mask_true and mask_prediction should have equal shapes.")

    area_intersection = (mask_true & mask_prediction).astype('uint8')
    area_union = (mask_true | mask_prediction).astype('uint8')

    if np.sum(area_union) == 0:
        return None

    return np.sum(area_intersection) / np.sum(area_union)


def _validate_box(box: Tuple[float, float, float, float]):
    if type(box) != tuple or len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(
            f"Bounding box must be defined as four elements tuple: (x_min, y_min, x_max, y_max), "
            f"where x_min < x_max and y_min < y_max. {box} given."
        )


def _validate_mask(mask: np.ndarray):
    if type(mask) != np.ndarray or mask.dtype != 'uint8' or len(mask.shape) != 2 or mask.min() < 0 or mask.max() > 1:
        raise ValueError(
            f"Mask must be defined as 2d np.array with np.uint8 type and binary values (0/1). {mask} given."
        )