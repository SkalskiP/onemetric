from typing import Tuple

import numpy as np


def box_iou(box_true: Tuple[float, float, float, float], box_prediction: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union of two bounding boxes.

    Args:
        box_true: Ground-truth bounding boxes. (x_min, y_min, x_max, y_max)
        box_prediction: Prediction bounding boxes. (x_min, y_min, x_max, y_max)

    Returns:
        float: Float value between 0 and 1.
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

    return area_intersection / (area_true + area_prediction - area_intersection)


def mask_iou(mask_true: np.array, mask_prediction: np.array) -> float:
    """
    Compute Intersection over Union of two masks.

    Args:
        mask_true: Ground-truth mask. 2d np.array
        mask_prediction: Prediction mask. 2d np.array

    Returns:
        float: Float value between 0 and 1.
    """
    pass


def _validate_box(box: Tuple[float, float, float, float]):
    if type(box) != tuple or len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(
            f"Bounding box must be defined as four elements tuple: (x_min, y_min, x_max, y_max), "
            f"where x_min < x_max and y_min < y_max. {box} given."
        )
