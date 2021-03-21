import os
from dataclasses import dataclass
from typing import Generator, List

import numpy as np

from onemetric.cv.loaders.base import DataSetElement, DataSetLoader
from onemetric.utils.general import list_files_with_extension

IMAGES_EXT = ('.png', '.jpg', '.jpeg')
ANNOTATION_EXT = '.txt'


@dataclass(frozen=True)
class DataSetEntry:
    image_path: str
    annotation_path: str


class YOLOElement(DataSetElement):

    def __init__(self, image: np.ndarray, annotations: np.ndarray) -> None:
        self._image = image
        self._annotations = annotations

    def get_image(self) -> np.ndarray:
        return self._image

    def get_annotations(self) -> np.ndarray:
        return self._annotations


class YOLOLoader(DataSetLoader):

    def __init__(self, images_dir_path: str, annotations_dir_path) -> None:
        self._images_dir_path = images_dir_path
        self._annotations_dir_path = annotations_dir_path

    def load(self) -> Generator[YOLOElement, None, None]:
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
        pass
