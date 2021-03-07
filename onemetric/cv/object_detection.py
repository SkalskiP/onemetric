import numpy as np

from onemetric.cv.utils import box_iou_batch


class ConfusionMatrix:
    """
    Calculate and visualize confusion matrix of Object Detection model. Updated version of
    https://github.com/kaanakan/object_detection_confusion_matrix
    """
    def __init__(self, num_classes: int, conf_threshold: float = 0.3, iou_threshold: float = 0.5) -> None:
        """
        Initialize new ConfusionMatrix instance.

        Args:
            num_classes: `int` number of classes detected by model.
            conf_threshold: `float` detection confidence threshold between 0 and 1. Detections with lower confidence will
            be excluded.
            iou_threshold: `float` detection iou  threshold between 0 and 1. Detections with lower iou will be excluded.
        """
        self.__matrix = np.zeros((num_classes, num_classes))
        self.__num_classes = num_classes
        self.__conf_threshold = conf_threshold
        self.__iou_threshold = iou_threshold

    def submit_batch(self, objects_true: np.ndarray, objects_detection: np.ndarray) -> None:
        """
        Update ConfusionMatrix instance with next batch of detections. This method should be triggered fo each image.

        Args:
            objects_true: 2d `np.ndarray` representing ground-truth objects. `shape = (N, 6)` where N is number of
            objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class, conf)`.
            objects_detection: `2d np.ndarray` representing detected objects. `shape = (M, 5)` where M is number of
            objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class)`.
        """
        objects_detection = objects_detection[objects_detection[:, 4] > self.__conf_threshold]

        classes_true = objects_true[:, 4].astype(np.int16)
        classes_detection = objects_detection[:, 4].astype(np.int16)
        boxes_true = objects_true[:, :4]
        boxes_detection = objects_detection[:, :4]
        iou_batch = box_iou_batch(boxes_true=boxes_true, boxes_detection=boxes_detection)
        matched_idx = np.where(iou_batch > self.__iou_threshold)

        all_matches = []
        for i in range(matched_idx[0].shape[0]):
            all_matches.append([matched_idx[0][i], matched_idx[1][i], iou_batch[matched_idx[0][i], matched_idx[1][i]]])
        all_matches = np.array(all_matches)
        print(all_matches)

    @property
    def matrix(self) -> np.ndarray:
        """
        Returns:
            confusion_matrix: 2d `np.ndarray` raw confusion matrix.
        """
        return self.__matrix

    def visualize(self) -> None:
        pass
