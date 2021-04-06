from abc import ABC, abstractmethod
from typing import Generator

import numpy as np


class DataSetElement(ABC):

    @abstractmethod
    def get_image(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_image_path(self) -> str:
        pass

    @abstractmethod
    def get_annotations(self) -> np.ndarray:
        pass


class DataSetLoader(ABC):

    @abstractmethod
    def load(self) -> Generator[DataSetElement, None, None]:
        pass
