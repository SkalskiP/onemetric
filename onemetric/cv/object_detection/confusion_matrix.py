from typing import Optional, List

import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from onemetric.cv.utils.iou import box_iou_batch


class ConfusionMatrix:
    """
    Calculate and visualize confusion matrix for Object Detection model.
    """
    def __init__(self, num_classes: int, conf_threshold: float = 0.3, iou_threshold: float = 0.5) -> None:
        """
        `ConfusionMatrix` object stores aggregated confusion matrix, that gets updated after each call of `submit_batch` method. Objects and detections are classified as TP, FP or FN according to rules listed below:

        - **[TP] True Positive** Detection is classified as TP when its `iou >= iou_threshold`.
        - **[FP] False Positive** Detection is classified as FP when its `iou < iou_threshold`. Additionally any duplicated bounding-boxes (multiple detections pointing to the same ground-truth object) are also considered FP.
        - **[FN] False Negative** Objects that are present in the image but failed to be detected are considered FN.
        - **[TN] True Negative** Every part of the image where we, correctly, did not detected an object is considered TN. This metrics is not useful for object detection, hence we ignore it.

        Args:
            num_classes: `int` number of classes detected by model.
            conf_threshold: `float` detection confidence threshold between 0 and 1. Detections with lower confidence will be excluded.
            iou_threshold: `float` detection iou  threshold between 0 and 1. Detections with lower iou will be classified as FP.
        """
        self._matrix = np.zeros((num_classes + 1, num_classes + 1))
        self._num_classes = num_classes
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold

    def submit_batch(self, true_batch: np.ndarray, detection_batch: np.ndarray) -> None:
        """
        Update aggregated confusion matrix with next batch of detections. **This method should be triggered for each image separately.**

        Args:
            true_batch: 2d `np.ndarray` representing ground-truth objects. `shape = (N, 5)` where `N` is number of annotated objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class)`.
            detection_batch: `2d np.ndarray` representing detected objects. `shape = (M, 6)` where `M` is number of detected objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class, conf)`.
        """
        _validate_true_batch(true_batch=true_batch)
        _validate_detection_batch(detection_batch=detection_batch)
        self._matrix += self._process_batch(true_batch=true_batch, detection_batch=detection_batch)

    @property
    def matrix(self) -> np.ndarray:
        """
        Property, aggregated confusion matrix.

        Returns:
            confusion_matrix: 2d `np.ndarray` raw confusion matrix. `shape = (num_classes + 1, num_classes + 1)` where additional row and column represent background class, FP and FN respectively.
        """
        return self._matrix

    def plot(self, target_path: str, title: Optional[str] = None, class_names: Optional[List[str]] = None) -> None:
        """
        Create a plot of confusion matrix and save it at selected location.

        Args:
            target_path: `str` selected target location of confusion matrix plot.
            title: `Optional[str]` title displayed at the top of the confusion matrix plot. Default `None`.
            class_names: `Optional[List[str]]` list of class names detected my model. If non given class indexes will be used. Default `None`.
        """
        array = self.matrix.copy()

        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        sn.set(font_scale=1.0 if self._num_classes < 50 else 0.8)

        labels = class_names is not None and (0 < len(class_names) < 99) and len(class_names) == self._num_classes
        x_tick_labels = class_names + ['FN'] if labels else "auto"
        y_tick_labels = class_names + ['FP'] if labels else "auto"
        sn.heatmap(array, annot=self._num_classes < 30, annot_kws={"size": 8}, fmt='.2f', square=True, vmin=0,
                   cmap='Blues', xticklabels=x_tick_labels, yticklabels=y_tick_labels).set_facecolor((1, 1, 1))

        if title:
            fig.axes[0].set_title(title)
        fig.axes[0].set_xlabel('Predicted label')
        fig.axes[0].set_ylabel('True label')
        fig.savefig(target_path, dpi=250)

    def _process_batch(self, true_batch: np.ndarray, detection_batch: np.ndarray) -> np.ndarray:
        result_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1))
        detection_batch_filtered = detection_batch[detection_batch[:, 5] > self._conf_threshold]
        true_classes = true_batch[:, 4].astype(np.int16)
        detection_classes = detection_batch_filtered[:, 4].astype(np.int16)
        true_boxes = true_batch[:, :4]
        detection_boxes = detection_batch_filtered[:, :4]

        self._validate_classes_idx_values(classes=true_classes)
        self._validate_classes_idx_values(classes=detection_classes)

        iou_batch = box_iou_batch(boxes_true=true_boxes, boxes_detection=detection_boxes)
        matched_idx = np.asarray(iou_batch > self._iou_threshold).nonzero()

        if matched_idx[0].shape[0]:
            matches = np.stack((matched_idx[0], matched_idx[1], iou_batch[matched_idx]), axis=1)
            matches = self._drop_extra_matches(matches=matches)
        else:
            matches = np.zeros((0, 3))

        matched_true_idx, matched_detection_idx, _ = matches.transpose().astype(np.int16)

        for i, true_class_value in enumerate(true_classes):
            j = matched_true_idx == i
            if matches.shape[0] > 0 and sum(j) == 1:
                result_matrix[true_class_value, detection_classes[matched_detection_idx[j]]] += 1  # TP
            else:
                result_matrix[true_class_value, self._num_classes] += 1  # FN

        for i, detection_class_value in enumerate(detection_classes):
            if not any(matched_detection_idx == i):
                result_matrix[self._num_classes, detection_class_value] += 1  # FP

        return result_matrix

    def _drop_extra_matches(self, matches: np.ndarray) -> np.ndarray:
        if matches.shape[0] > 0:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        return matches

    def _validate_classes_idx_values(self, classes: np.ndarray):
        if classes.size > 0 and (classes.min() < 0 or classes.max() > self._num_classes - 1):
            raise ValueError(
                f"Class index values must be between 0 and ${self._num_classes - 1}. {classes} given."
            )


def _validate_true_batch(true_batch: np.ndarray):
    if type(true_batch) != np.ndarray or len(true_batch.shape) != 2 or true_batch.shape[1] != 5:
        raise ValueError(
            f"True batch must be defined as 2d np.array with (N, 5) shape, where N is number of is number of "
            f"annotated objects and each row is in (x_min, y_min, x_max, y_max, class) format. {true_batch} given."
        )


def _validate_detection_batch(detection_batch: np.ndarray):
    if type(detection_batch) != np.ndarray or len(detection_batch.shape) != 2 or detection_batch.shape[1] != 6:
        raise ValueError(
            f"Detected batch must be defined as 2d np.array with (M, 6) shape, where M is number of is number of "
            f"detected objects and each row is in (x_min, y_min, x_max, y_max, class, conf) format. "
            f"{detection_batch} given."
        )
