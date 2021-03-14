import numpy as np
from typing import Tuple

from onemetric.const import EPSILON

# Updated version of compute_ap from https://github.com/ultralytics/yolov3


def compute_ap(
    recall: np.ndarray,
    precision: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    recall_values = np.concatenate(([0.], recall, [recall[-1] + EPSILON]))
    precision_values = np.concatenate(([1.], precision, [0.]))
    precision_values = np.flip(np.maximum.accumulate(np.flip(precision_values)))

    i = np.where(recall_values[1:] != recall_values[:-1])[0]
    ap = np.sum((recall_values[i + 1] - recall_values[i]) * precision_values[i + 1])

    return ap, recall_values, precision_values
