from abc import ABC
import hydra

class Prompting(ABC):
    def __init__(self, formatter, retriever):
        self.formatter = formatter
        self.retriever = retriever

    def __call__(self, batch):
        """Batch is an iterable of dictionaries with keys "id" and "text" (and possibly other)."""
        raise NotImplementedError("Not implemented")