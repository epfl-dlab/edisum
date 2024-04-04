from abc import ABC
from typing import List


class Retriever(ABC):
    def __call__(self, sample:dict) -> List[dict]:
        
        raise NotImplementedError("Not implemented")