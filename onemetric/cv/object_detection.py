from typing import Optional, List

import numpy as np

from onemetric.cv.utils import box_iou_batch

# Updated version of ConfusionMatrix from https://github.com/kaanakan/object_detection_confusion_matrix


class ConfusionMatrix:
    """
    Calculate and visualize confusion matrix of Object Detection model.
    """
    def __init__(self, num_classes: int, conf_threshold: float = 0.3, iou_threshold: float = 0.5) -> None:
        """
        Initialize new ConfusionMatrix instance.

        Args:
            num_classes: `int` number of classes detected by model.
            conf_threshold: `float` detection confidence threshold between 0 and 1. Detections with lower confidence will be excluded.
            iou_threshold: `float` detection iou  threshold between 0 and 1. Detections with lower iou will be excluded.
        """
        self.__matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.__num_classes = num_classes
        self.__conf_threshold = conf_threshold
        self.__iou_threshold = iou_threshold

    def submit_batch(self, true_batch: np.ndarray, detection_batch: np.ndarray) -> None:
        """
        Update ConfusionMatrix instance with next batch of detections. This method should be triggered fo each image separately.

        Args:
            true_batch: 2d `np.ndarray` representing ground-truth objects. `shape = (N, 5)` where `N` is number of annotated objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class)`.
            detection_batch: `2d np.ndarray` representing detected objects. `shape = (M, 6)` where `M` is number of detected objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class, conf)`.
        """
        self.__validate_true_batch(true_batch=true_batch)
        self.__validate_detection_batch(detection_batch=detection_batch)
        self.__matrix += self.__process_batch(true_batch=true_batch, detection_batch=detection_batch)

    @property
    def matrix(self) -> np.ndarray:
        """
        Return raw confusion matrix.

        Returns:
            confusion_matrix: 2d `np.ndarray` raw confusion matrix. `shape = (num_classes + 1, num_classes + 1)` where additional row and column represents background class.
        """
        return self.__matrix

    def plot(self, target_path: str, class_names: Optional[List[str]] = None) -> None:
        pass

    def __validate_true_batch(self, true_batch: np.ndarray):
        if type(true_batch) != np.ndarray or len(true_batch.shape) != 2 or true_batch.shape[1] != 5:
            raise ValueError(
                f"True batch must be defined as 2d np.array with (N, 5) shape, where N is number of is number of "
                f"annotated objects and each row is in (x_min, y_min, x_max, y_max, class) format. {true_batch} given."
            )

    def __validate_detection_batch(self, detection_batch: np.ndarray):
        if type(detection_batch) != np.ndarray or len(detection_batch.shape) != 2 or detection_batch.shape[1] != 6:
            raise ValueError(
                f"Detected batch must be defined as 2d np.array with (N, 6) shape, where N is number of is number of "
                f"annotated objects and each row is in (x_min, y_min, x_max, y_max, class, conf) format. "
                f"{detection_batch} given."
            )

    def __process_batch(self, true_batch: np.ndarray, detection_batch: np.ndarray) -> np.ndarray:
        matrix = np.zeros((self.__num_classes + 1, self.__num_classes + 1))

        filtered_detection_batch = detection_batch[detection_batch[:, 5] > self.__conf_threshold]

        classes_true = true_batch[:, 4].astype(np.int16)
        classes_detection = filtered_detection_batch[:, 4].astype(np.int16)
        boxes_true = true_batch[:, :4]
        boxes_detection = filtered_detection_batch[:, :4]
        iou_batch = box_iou_batch(boxes_true=boxes_true, boxes_detection=boxes_detection)
        matched_idx = np.asarray(iou_batch > self.__iou_threshold).nonzero()
        matches = np.stack((matched_idx[0], matched_idx[1], iou_batch[matched_idx]), axis=1)
        matches = self._drop_extra_matches(matches=matches)

        for i, label in enumerate(true_batch):
            if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1:
                gt_class = classes_true[i]
                detection_class = classes_detection[int(matches[matches[:, 0] == i, 1][0])]
                matrix[gt_class, detection_class] += 1  # TP (correct)
            else:
                gt_class = classes_true[i]
                matrix[self.__num_classes, gt_class] += 1  # FP (background)

        for i, detection in enumerate(filtered_detection_batch):
            if matches.shape[0] and matches[matches[:, 1] == i].shape[0] == 0:
                detection_class = classes_detection[i]
                matrix[detection_class, self.__num_classes] += 1  # FN (background)

        return matrix

    def _drop_extra_matches(self, matches: np.ndarray) -> np.ndarray:
        if matches.shape[0] > 0:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        return matches


class MeanAveragePrecision:
    """
    To be added soon. Calculate and visualize mean average precision (mAP) of Object Detection model.
    """
    pass
