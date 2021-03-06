from typing import Tuple

import numpy as np


def box_iou(box_true: Tuple[float, float, float, float], box_prediction: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union of two bounding boxes.
    :param box_true: Ground-truth bounding boxes. [x_min, y_min, x_max, y_max]
    :param box_prediction: Prediction bounding boxes. [x_min, y_min, x_max, y_max]
    :return: Float between 0 and 1.
    """
    pass


def mask_iou(mask_true: np.array, mask_prediction: np.array) -> float:
    """
    Compute Intersection over Union of two masks.
    :param mask_true: Ground-truth mask.
    :param mask_prediction: Prediction mask.
    :return: Float between 0 and 1.
    """
    pass
