import os
from typing import Tuple, List, Union


def list_files_with_extension(root_path: str, extensions: Union[str, Tuple[str, ...]]) -> List[str]:
    return [file for file in os.listdir(root_path) if file.endswith(extensions)]


def read_text_file_lines(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        lines = [line.strip('\n') for line in file.readlines()]
        return [line for line in lines if len(line) > 0]
