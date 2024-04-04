from src.openai.prompting.formatters.abstract import Formatter
from typing import List

class SimpleFormatter(Formatter):
    def __init__(
        self,
        global_format=None,
        query_format=None,
        demonstration_format=None,
        instruction=None
    ):
        self.global_format = global_format if global_format \
            else "{instruction}\n\n{demonstration}\n\n{query}\ncomment:"
        self.query_format = query_format if query_format \
            else "text removed:\n{text1}\ntext added:\n{text2}"
        self.demonstration_format = demonstration_format if demonstration_format \
            else "text removed:\n{text1}\ntext added:\n{text2}\ncomment:{target}"  
        self.instruction = instruction if instruction \
            else "You are presented with the edit of a Wikipedia article in a form of text that was removed and added. Give a brief summary of the edit in a form of a comment no longer than 15 words."

    def __query_formatting(self, text1, text2, target):
        return self.query_format.replace("{text1}", text1).replace("{text2}", text2).replace("{target}", target)

    def __demonstration_formatting(self, demonstrations: List):
        formatted_demonstrations = []
        for i, demonstration in enumerate(demonstrations):
            prev_texts = sorted(demonstration["prev_texts"])
            cur_texts = sorted(demonstration["cur_texts"])
            target = demonstration["summary"]

            formatted = self.demonstration_format\
                .replace("{text1}", "\n".join(prev_texts))\
                .replace("{text2}", "\n".join(cur_texts))\
                .replace("{target}", target)
            formatted_demonstrations.append(f"\n"+formatted)  # .append(f"Example{i+1}:\n"+formatted)
        
        return "\n".join(formatted_demonstrations)
    
    def __global_formatting(self, instruction, demonstration, query):
        return self.global_format.replace("{instruction}", instruction)\
            .replace("{demonstration}", demonstration)\
            .replace("{query}", query)

    def __call__(self, batch_samples, batch_demonstrations):
        batch_prompts = []
        for sample in batch_samples:
            text1 = "\n".join(sample["prev_texts"])
            text2 = "\n".join(sample["cur_texts"])
            target = sample["summary"]
            query = self.__query_formatting(text1, text2, target)
            demonstration = batch_demonstrations[sample["id"]]
            demonstration = self.__demonstration_formatting(demonstration)
            prompt = self.__global_formatting(self.instruction, demonstration, query)
            batch_prompts.append({"id":sample["id"],"prompt":prompt})
            
        return batch_prompts