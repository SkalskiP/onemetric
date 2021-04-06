import os
from typing import Generator, List

import numpy as np
from PIL import Image

from onemetric.cv.loaders.base import DataSetElement, DataSetLoader
from onemetric.cv.loaders.yolo.dataclasses import DataSetEntry
from onemetric.utils.general import list_files_with_extension, read_text_file_lines

IMAGES_EXT = ('.png', '.jpg', '.jpeg')
ANNOTATION_EXT = '.txt'


class YOLOElement(DataSetElement):
    """
    TODO:
    """

    def __init__(self, image: np.ndarray, image_path: str, annotations: np.ndarray) -> None:
        """
        TODO:
        """
        self._image = image
        self._image_path = image_path
        self._annotations = annotations

    def get_image(self) -> np.ndarray:
        """
        TODO:
        """
        return self._image

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


class YOLOLoader(DataSetLoader):
    """
    TODO:
    """

    def __init__(self, images_dir_path: str, annotations_dir_path) -> None:
        """
        TODO:
        """
        self._images_dir_path = images_dir_path
        self._annotations_dir_path = annotations_dir_path

    def load(self) -> Generator[YOLOElement, None, None]:
        """
        TODO:
        """
        for entry in self._get_entries():
            yield YOLOLoader._load_element(entry=entry)

    def _get_entries(self) -> List[DataSetEntry]:
        image_names = list_files_with_extension(root_path=self._images_dir_path, extensions=IMAGES_EXT)
        annotation_names = list_files_with_extension(root_path=self._annotations_dir_path, extensions=ANNOTATION_EXT)
        images = {
            os.path.splitext(image_name)[0]: image_name
            for image_name
            in image_names
        }
        annotations = {
            os.path.splitext(annotation_name)[0]: annotation_name
            for annotation_name
            in annotation_names
        }
        common_tokens = set(images.keys()).intersection(set(annotations.keys()))
        return [
            DataSetEntry(
                image_path=os.path.join(self._images_dir_path, images.get(token)),
                annotation_path=os.path.join(self._annotations_dir_path, annotations.get(token))
            )
            for token
            in sorted(common_tokens)
        ]

    @staticmethod
    def _load_element(entry: DataSetEntry) -> YOLOElement:
        image = np.asarray(Image.open(entry.image_path))
        image_height, image_width, _ = image.shape
        annotations = YOLOLoader._load_annotations(
            annotation_path=entry.annotation_path,
            image_height=image_height,
            image_width=image_width
        )
        return YOLOElement(image=image, image_path=entry.image_path, annotations=annotations)

    @staticmethod
    def _load_annotations(annotation_path: str, image_height: int, image_width: int) -> np.ndarray:
        raw_annotations = read_text_file_lines(file_path=annotation_path)
        annotation = np.stack([
            YOLOLoader._load_annotation(raw_annotation=raw_annotation)
            for raw_annotation
            in raw_annotations
        ])
        annotation[:, 0] *= image_width
        annotation[:, 2] *= image_width
        annotation[:, 1] *= image_height
        annotation[:, 3] *= image_height
        return annotation

    @staticmethod
    def _load_annotation(raw_annotation: str) -> np.ndarray:
        class_idx, x, y, width, height = raw_annotation.split(' ')
        return np.array([
            float(x) - float(width) / 2,
            float(y) - float(height) / 2,
            float(x) + float(width) / 2,
            float(y) + float(height) / 2,
            int(class_idx)
        ])
