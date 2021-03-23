from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

from onemetric.cv.object_detection.average_precision import AveragePrecision


@dataclass(frozen=True)
class MeanAveragePrecision:
    value: float
    per_class: List[AveragePrecision]
    num_classes: int
    iou_threshold: float

    @classmethod
    def from_detections(
        cls,
        true_batches: List[np.ndarray],
        detection_batches: List[np.ndarray],
        num_classes: int,
        iou_threshold: float = 0.5
    ) -> MeanAveragePrecision:
        """
        Calculate mean average precision (mAP) metric for selected `iou_threshold` based on `true_batches` and `detection_batches`.

        Args:
            true_batches: `List[np.ndarray]` representing ground-truth objects across all images in concerned dataset. Each element of `true_batches` list describe single image and has `shape = (N, 5)` where `N` is number of ground-truth objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class)`.
            detection_batches: `List[np.ndarray]` representing detected objects across all images in concerned dataset. Each element of `detection_batches` list describe single image and has `shape = (M, 6)` where `M` is number of detected objects. Each row is expected to be in `(x_min, y_min, x_max, y_max, class, conf)`.
            num_classes: `int` number of classes detected by model.
            iou_threshold: `float` detection iou  threshold between 0 and 1. Detections with lower iou will be classified as FP.

        Returns:
            mean_average_precision: `MeanAveragePrecision` object containing mAP `value` calculated for provided `iou_threshold` as well as `AveragePrecision` object calculated for each individual class.

        Example:
        ```
        >>> import numpy as np

        >>> from onemetric.cv.object_detection import MeanAveragePrecision

        >>> true_batches = [
        ...     np.array([
        ...         [0.0, 0.0, 3.0, 3.0, 1],
        ...         [2.0, 2.0, 5.0, 5.0, 1],
        ...         [6.0, 1.0, 8.0, 3.0, 2],
        ...     ]),
        ...     np.array([
        ...         [1.0, 1.0, 2.0, 2.0, 2],
        ...     ]),
        ... ]

        >>> detection_batches = [
        ...     np.array([
        ...         [0.0, 0.0, 3.0, 3.0, 1, 0.9],
        ...         [0.1, 0.1, 3.0, 3.0, 0, 0.9],
        ...         [6.0, 1.0, 8.0, 3.0, 1, 0.8],
        ...         [1.0, 6.0, 2.0, 7.0, 1, 0.8],
        ...     ]),
        ...     np.array([
        ...         [1.0, 1.0, 2.0, 2.0, 2, 0.8],
        ...     ]),
        ... ]

        >>> mean_average_precision = MeanAveragePrecision.from_detections(
        ...     true_batches=true_batches,
        ...     detection_batches=detection_batches,
        ...     num_classes=3
        ... )

        >>> mean_average_precision.value
        ... 0.4444444444444444
        ```
        """
        per_class = [
            AveragePrecision.from_detections(
                true_batches=true_batches,
                detection_batches=detection_batches,
                class_idx=class_idx,
                iou_threshold=iou_threshold
            )
            for class_idx
            in range(num_classes)
        ]
        values = [ap.value for ap in per_class]
        return cls(value=sum(values) / num_classes, per_class=per_class, num_classes=num_classes, iou_threshold=iou_threshold)

    def plot(self, target_path: str, title: Optional[str] = None, class_names: Optional[List[str]] = None) -> None:
        """
        Create mean average precision plot and save it at selected location.

        Args:
            target_path: `str` selected target location of confusion matrix plot.
            title: `Optional[str]` title displayed at the top of the confusion matrix plot. Default `None`.
            class_names: `Optional[List[str]]` list of class names detected my model. If non given class indexes will be used. Default `None`.
        """
        text_labels = class_names is not None and len(class_names) == self.num_classes
        labels = class_names if text_labels else list(range(self.num_classes))

        fig = plt.figure(figsize=(12, 9), tight_layout=True, facecolor='white')
        ax = fig.add_subplot(111)

        for label, ap in zip(labels, self.per_class):
            ax.plot(ap.recall_values, ap.precision_values, label=label, linewidth=2.0,)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        # axis style
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
            ax.spines[s].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1, 1), facecolor='white',
                  framealpha=1, frameon=False, fontsize=10)
        ax.set_facecolor('white')
        ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.5)

        if title:
            plt.title(title, fontsize=20, pad=20)

        fig.savefig(target_path, dpi=250, facecolor=fig.get_facecolor(), transparent=True)

