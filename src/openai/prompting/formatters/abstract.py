from abc import ABC
from typing import List


class Formatter(ABC):
    def __call__(self, sample: dict, demonstrations: List[dict] = None) -> str:
        raise NotImplementedError("Not implemented")