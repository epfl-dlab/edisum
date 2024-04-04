from src.openai.prompting.abstract import Prompting

class SimplePrompting(Prompting):
    def __init__(self, formatter, retriever):
        super().__init__(formatter, retriever)
        
    def parse_output(self, text):
        return text

    def __call__(self, batch):
        """Batch is an iterable of dictionaries with keys "id" and "text" (and possibly other)."""
        # if batch is not an iterable wrap it in a list
        if not hasattr(batch, "__iter__"):
            batch = [batch]
        demonstrations = self.retriever(batch)
        batch_prompts = self.formatter(batch, demonstrations)

        return batch_prompts