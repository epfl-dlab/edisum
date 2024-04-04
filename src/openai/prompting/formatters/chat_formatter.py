from src.openai.prompting.formatters.abstract import Formatter
from typing import List

class ChatFormatter(Formatter):
    def __init__(
        self,
        query_format=None,
        demonstration_format=None,
        user_instruction=None,
        system_instruction=None,
        system_instruction_only=False
    ):
        self.query_format = query_format if query_format \
            else "text removed:\n{text1}\ntext added:\n{text2}"
        self.demonstration_format = demonstration_format if demonstration_format \
            else "text removed:\n{text1}\ntext added:\n{text2}"  
        self.user_instruction = user_instruction if user_instruction \
            else "You are presented with the edit of a Wikipedia article in a form of text that was removed and added. Give a brief summary of the edit in a form of a comment no longer than 15 words."
        self.system_instruction = system_instruction if system_instruction \
            else "You are an assistant that writes comments about the change for Wikipedia edits"
        self.system_instruction_only = system_instruction_only

    def __query_formatting(self, text1, text2, target):
        query_msgs = []
        content = self.query_format.replace("{text1}", text1).replace("{text2}", text2).replace("{target}", target)
        query_msgs.append({'role':'user', 'content':content})
        return query_msgs

    def __demonstration_formatting(self, demonstrations: List):
        demonstrations_msgs = []
        for i, demonstration in enumerate(demonstrations):
            prev_texts = sorted(demonstration["prev_texts"])
            cur_texts = sorted(demonstration["cur_texts"])
            target = demonstration["summary"]

            formatted = self.demonstration_format\
                .replace("{text1}", "\n".join(prev_texts))\
                .replace("{text2}", "\n".join(cur_texts))
                
            demonstration_msg = [{'role':'user', 'content': formatted},{'role':'assistant', 'content':target}]
            demonstrations_msgs.extend(demonstration_msg) 
        
        return demonstrations_msgs
    
    def __global_formatting(self, demonstration_msgs, query_msgs):
        msgs = []
        msgs.append({'role': 'system', 'content': self.system_instruction})
        if not self.system_instruction_only:
            msgs.append({'role': 'user', 'content': self.user_instruction})
            msgs.append({'role': 'assistant', 'content': "Understood"})
        msgs.extend(demonstration_msgs)
        msgs.extend(query_msgs)
        return msgs

    def __call__(self, batch_samples, batch_demonstrations):
        batch_msgs = []
        for sample in batch_samples:
            text1 = "\n".join(sample["prev_texts"])
            text2 = "\n".join(sample["cur_texts"])
            target = sample["summary"]
            query_msgs = self.__query_formatting(text1, text2, target)
            demonstration = batch_demonstrations[sample["id"]]
            demonstrations_msgs = self.__demonstration_formatting(demonstration)
            msgs = self.__global_formatting(demonstrations_msgs, query_msgs)
            batch_msgs.append({"id":sample["id"], "prompt":msgs})
            
        return batch_msgs