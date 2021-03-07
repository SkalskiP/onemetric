import numpy as np


class ConfusionMatrix:
    """
    Calculate and visualize confusion matrix of Object Detection model.
    """
    def __init__(self, num_classes: int, conf_threshold: float = 0.3, iou_threshold: float = 0.5) -> None:
        """
        Initialize new ConfusionMatrix instance.

        Args:
            num_classes: int
            conf_threshold: float
            iou_threshold: float
        """
        self.__matrix = np.zeros((num_classes, num_classes))
        self.__num_classes = num_classes
        self.__conf_threshold = conf_threshold
        self.__iou_threshold = iou_threshold
