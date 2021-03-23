from typing import Tuple, Optional

import numpy as np


def box_iou(
    box_true: Tuple[float, float, float, float],
    box_detection: Tuple[float, float, float, float]
) -> Optional[float]:
    """
    Compute Intersection over Union of two bounding boxes - `box_true` and `box_detection`. Both boxes are expected to be
    tuples in `(x_min, y_min, x_max, y_max)` format.

    Args:
        box_true: `tuple` representing ground-truth bounding boxes.
        box_detection: `tuple` representing detection bounding boxes.

    Returns:
        iou: `float` value between 0 and 1. `None` if union is equal to 0.

    Example:
    ```
    >>> from onemetric.cv.utils.iou import box_iou

    >>> iou = box_iou_batch(
    ...     boxes_true=(0., 0., 1., 1.),
    ...     boxes_detection=(0.25, 0., 1.25, 1.)
    ... )

    >>> iou
    ... 0.6
    ```
    """
    _validate_box(box=box_true)
    _validate_box(box=box_detection)

    x_min_true, y_min_true, x_max_true, y_max_true = box_true
    x_min_detection, y_min_detection, x_max_detection, y_max_detection = box_detection

    x_min = max(x_min_true, x_min_detection)
    y_min = max(y_min_true, y_min_detection)
    x_max = min(x_max_true, x_max_detection)
    y_max = min(y_max_true, y_max_detection)

    area_intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    area_true = (x_max_true - x_min_true) * (y_max_true - y_min_true)
    area_detection = (x_max_detection - x_min_detection) * (y_max_detection - y_min_detection)
    area_union = area_true + area_detection - area_intersection

    if area_union == 0:
        return None

    return area_intersection / area_union


# Updated version of box_iou_batch from https://github.com/kaanakan/object_detection_confusion_matrix


def box_iou_batch(boxes_true: np.ndarray, boxes_detection: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union of two sets of bounding boxes - `boxes_true` and `boxes_detection`. Both sets of
    boxes are expected to be in `(x_min, y_min, x_max, y_max)` format.

    Args:
        boxes_true: 2d `np.ndarray` representing ground-truth boxes. `shape = (N, 4)` where N is number of true objects.
        boxes_detection: 2d `np.ndarray` representing detection boxes. `shape = (M, 4)` where M is number of detected objects.

    Returns:
        iou: 2d `np.ndarray` representing pairwise IoU of boxes from `boxes_true` and `boxes_detection`. `shape = (N, M)` where N is number of true objects and M is number of detected objects.

    Example:
    ```
    >>> import numpy as np

    >>> from onemetric.cv.utils.iou import box_iou_batch

    >>> boxes_true = np.array([
    ...     [0., 0., 1., 1.],
    ...     [2., 2., 2.5, 2.5]
    ... ])
    >>> boxes_detection = np.array([
    ...     [0., 0., 1., 1.],
    ...     [2., 2., 2.5, 2.5]
    ... ])
    >>> iou = box_iou_batch(boxes_true=boxes_true, boxes_detection=boxes_detection)

    >>> iou
    ... np.array([
    ...     [1., 0.],
    ...     [0., 1.]
    ... ])
    ```
    """

    _validate_boxes_batch(boxes_batch=boxes_true)
    _validate_boxes_batch(boxes_batch=boxes_detection)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_true = box_area(boxes_true.T)
    area_detection = box_area(boxes_detection.T)

    top_left = np.maximum(boxes_true[:, None, :2], boxes_detection[:, :2])
    bottom_right = np.minimum(boxes_true[:, None, 2:], boxes_detection[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    return area_inter / (area_true[:, None] + area_detection - area_inter)


def mask_iou(mask_true: np.ndarray, mask_detection: np.ndarray) -> Optional[float]:
    """
    Compute Intersection over Union of two masks - mask_true and mask_detection. Shapes of mask_true and
    mask_detection should be identical. Both arrays are expected to be `np.uint8` type and contain binary values (0 or 1).

    Args:
        mask_true: 2d `np.ndarray` representing ground-truth mask.
        mask_detection: 2d `np.ndarray` representing detection mask.

    Returns:
        iou: `float` value between 0 and 1. `None` if union is equal to 0.

    Example:
    ```
    >>> import numpy as np

    >>> from onemetric.cv.utils.iou import mask_iou

    >>> full_mask = np.ones((10, 10)).astype('uint8')
    >>> quarter_mask = np.zeros((10, 10)).astype('uint8')
    >>> quarter_mask[0:5, 0:5] = 1

    >>> iou = mask_iou(mask_true=full_mask, mask_detection=quarter_mask)

    >>> iou
    ... 0.25
    ```
    """
    _validate_mask(mask=mask_true)
    _validate_mask(mask=mask_detection)

    if mask_true.shape != mask_detection.shape:
        raise ValueError(f"mask_true and mask_detection should have equal shapes.")

    area_intersection = (mask_true & mask_detection).astype('uint8')
    area_union = (mask_true | mask_detection).astype('uint8')

    if np.sum(area_union) == 0:
        return None

    return np.sum(area_intersection) / np.sum(area_union)


def _validate_box(box: Tuple[float, float, float, float]):
    if type(box) != tuple or len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(
            f"Bounding box must be defined as four elements tuple: (x_min, y_min, x_max, y_max), "
            f"where x_min < x_max and y_min < y_max. {box} given."
        )


def _validate_boxes_batch(boxes_batch: np.ndarray):
    if type(boxes_batch) != np.ndarray or len(boxes_batch.shape) != 2 or boxes_batch.shape[1] != 4:
        raise ValueError(
            f"Bounding boxes batch must be defined as 2d np.array with (N, 4) shape, {boxes_batch} given"
        )


def _validate_mask(mask: np.ndarray):
    if type(mask) != np.ndarray or mask.dtype != 'uint8' or len(mask.shape) != 2 or mask.min() < 0 or mask.max() > 1:
        raise ValueError(
            f"Mask must be defined as 2d np.array with np.uint8 type and binary values (0/1). {mask} given."
        )
