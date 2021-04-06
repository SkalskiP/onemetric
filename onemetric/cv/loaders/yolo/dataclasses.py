from dataclasses import dataclass


@dataclass(frozen=True)
class DataSetEntry:
    image_path: str
    annotation_path: str
