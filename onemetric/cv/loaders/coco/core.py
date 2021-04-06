from typing import Optional

from onemetric.cv.loaders.base import DataSetLoader, DataSetElement

import numpy as np


class COCOElement(DataSetElement):
    """
    TODO:
    """

    def __init__(self, image_path: str, annotations: np.ndarray) -> None:
        """
        TODO:
        """
        self._image: Optional[np.ndarray] = None
        self._image_path = image_path
        self._annotations = annotations

    def get_image(self) -> np.ndarray:
        """
        TODO:
        """
        pass

    def get_image_path(self) -> str:
        """
        TODO:
        """
        return self._image_path

    def get_annotations(self) -> np.ndarray:
        """
        TODO:
        """
        return self._annotations


class COCOLoader(DataSetLoader):
    """
    TODO:
    """

    def __init__(self, images_dir_path: str, annotations_file_path) -> None:
        """
        TODO:
        """
        self._images_dir_path = images_dir_path
        self._annotations_file_path = annotations_file_path
