from typing import List

import numpy as np


def validate_true_batch(true_batch: np.ndarray):
    if type(true_batch) != np.ndarray or len(true_batch.shape) != 2 or true_batch.shape[1] != 5:
        raise ValueError(
            f"True batch must be defined as 2d np.array with (N, 5) shape, where N is number of is number of "
            f"annotated objects and each row is in (x_min, y_min, x_max, y_max, class) format. {true_batch} given."
        )


def validate_detection_batch(detection_batch: np.ndarray):
    if type(detection_batch) != np.ndarray or len(detection_batch.shape) != 2 or detection_batch.shape[1] != 6:
        raise ValueError(
            f"Detected batch must be defined as 2d np.array with (M, 6) shape, where M is number of is number of "
            f"detected objects and each row is in (x_min, y_min, x_max, y_max, class, conf) format. "
            f"{detection_batch} given."
        )


def validate_detections(true_batches: List[np.ndarray], detection_batches: List[np.ndarray]):
    if type(true_batches) != list or type(detection_batches) != list or len(true_batches) != len(detection_batches):
        raise ValueError('true_batches and detection_batches must be lists and their lengths must be equal.')


def validate_precision_recall(recall: np.ndarray, precision: np.ndarray):
    if type(recall) != np.ndarray or type(precision) != np.ndarray or recall.shape != precision.shape:
        raise ValueError("recall and precision must be 1d np.array with (N, ) shape.")