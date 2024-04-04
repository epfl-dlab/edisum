import jsonlines
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time
import math
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm


class OpenAIAPI():
    def __init__(
        self,
        model,
        api_keys,
        prompting,
        n_workers_per_key,
        generation_parameters,
        dry_run=False,
        gen_dataset=False,
        no_gen=False
    ):
        self.model = model
        self.api_keys = api_keys.split(",")
        self.prompting = prompting
        self.n_workers_per_key = n_workers_per_key

        self.generation_parameters = generation_parameters

        self.dry_run = dry_run
        # if generate the formal dataset, will select certain columns
        self.gen_dataset = gen_dataset

        self.__waittime_per_key = int(1)  # 10 requests per minutes

        # Initialize to now - waittime_per_key to make the class know we haven't called it recently
        self.__last_call_per_key = [time.time() - self.__waittime_per_key] * len(self.api_keys)
        
        # whether or not to generate the data (if True, this class would be used to format data only)
        self.no_gen = no_gen  

    def _choose_next_api_key(self):
        """
        It chooses the next API key to use, by:
        - finding the one that has been used the least recently
        - check whether we need to wait for using it or not
        - if we don't need to wait, we use this key
        - if we need to wait, we wait the appropriate amount of time and retry to find a key
        Why retry instead of using the key we were waiting for after waiting?
        Because another thread might have taken this key and another one might have become available in the meantime.
        Returns: api_key_index, the index of the key to using next
        """
        api_key_idx = self.__last_call_per_key.index(min(self.__last_call_per_key))
        last_call_on_key = time.time() - self.__last_call_per_key[api_key_idx]
        good_to_go = last_call_on_key > self.__waittime_per_key

        if not good_to_go:
            time.sleep(self.__waittime_per_key - last_call_on_key)
            return self._choose_next_api_key()

        self.__last_call_per_key[api_key_idx] = time.time()
        return api_key_idx

    # @retry(wait=wait_random_exponential(min=5, max=180), stop=stop_after_attempt(15))
    def _get_text_completion(
        self,
        prompts,
        api_key_idx,
    ):
        """
        Calls the OpenAI API with the chosen API key for the selected batch of prompts
        In case an error occurs, the library tenacity will retry the call with exponentially increasing sleep times,
        to a maximum of 6 attempts and for a maximum waiting time of 60s.
        When calling the API, it uses the self.generation_parameters passed in __init__
        Args:
            prompts: list of prompts to send to the API
            api_key_idx: the index of the API key to use (api_keys are stored in self.api_keys)
        Returns: JSON-object from OpenAI API with the model completions.
        """
        import openai

        openai.api_key = self.api_keys[api_key_idx]

        # We re-update the last call here, in case we have retries.
        self.__last_call_per_key[api_key_idx] = time.time()
        if self.model in ['text-curie-001', 'text-davinci-003']:
            return openai.Completion.create(prompt=prompts, model=self.model, **self.generation_parameters)
        elif self.model in ['gpt-3.5-turbo', 'gpt-4']:
            return [openai.ChatCompletion.create(messages=prompt, model=self.model, request_timeout=800, **self.generation_parameters) for prompt in prompts]
        
 
    def clean_completions(self, completions):
        """
        Parses the JSON object from OpenAI to extract the relevant 'text' field.
        The prompting-strategy-specific parsing is called on each completion.
        This might be necessary if the prompting strategy expects some specific structure in the answer from which it
        can extract the answer to the query.
        Args:
            completions: JSON-object returned by OpenAI API
        Returns: a dictionary where each index (individual prompt) is associated with the list of model completion.
        Depending on the choice of generation parameters, there might be several completions per prompt.
        """
        collected_answers = defaultdict(list)
        if self.model in ['text-curie-001', 'text-davinci-003']:
            for k in completions["choices"]:
                idx = k["index"]  / self.generation_parameters.get("best_of", 1)
                collected_answers[math.floor(idx)].append(self.prompting.parse_output(k["text"]))
        elif self.model in ['gpt-3.5-turbo', 'gpt-4']:
            for i, completion in enumerate(completions):
                idx = i
                collected_answers[math.floor(idx)].append(self.prompting.parse_output(completion["choices"][0]["message"]["content"]))
        return collected_answers

    def predict(self, batch):
        # Transforms the batch into prompts according to the prompting strategy
        prompts = [p['prompt'] for p in self.prompting(batch)]
        if self.dry_run:
            for prompt in prompts:
                print(str(prompt) + "\n")
            return
        api_key_idx = self._choose_next_api_key() if len(self.api_keys) > 1 else 0
        max_attempts = 3
        cnt = 0
        while cnt < max_attempts:
            try:
                cnt += 1
                completions = self._get_text_completion(prompts, api_key_idx)
                break
            except Exception as e:
                sec_wait = 5
                print(e)
                print(f"Error when querying OpenAI, try again in {sec_wait} sec")
                time.sleep(sec_wait)

        cleaned_completions = self.clean_completions(completions) # TODO fix the bug with referencing
        

        # Format the output to get: the query, the formatted prompt, and the list of model completions
        for key, sample in enumerate(batch):
            sample["prompt"] = prompts[key]
            sample["model_completions"] = cleaned_completions[key] if not self.gen_dataset else cleaned_completions[key][0]

        if hasattr(self, "outputs"):  # outputs is defined in "predict_datamodule", doing this for multiproccessing
            self.outputs.extend(batch)
        return batch

    def predict_datamodule(self, dataloader, output_file=None, output_type='json', output_cols=['']):
        if output_file:
            assert output_type in {'jsonl', 'json', 'csv'}, "only support output type in csv or json"
            assert output_file.split(".")[-1] == output_type, "output filename does not match output type"
        self.outputs = []
        self.has_header = False  # for csv output
        
        for batch in tqdm(dataloader):
            if not self.no_gen:
                self.predict(batch)
                if not self.dry_run and output_file is not None:
                    self.write_outputs(output_file, self.outputs, output_type, output_cols)
                self.outputs = [] 
            else:
                for _, sample in enumerate(batch):
                    sample["model_completions"] = []
                if not self.dry_run and output_file is not None:
                    self.write_outputs(output_file, self.outputs, output_type, output_cols)
                self.outputs = []
        
        # n_workers = self.n_workers_per_key * len(self.api_keys)
        # with ThreadPoolExecutor(max_workers=n_workers) as executor:
        #     executor.map(self.predict, dataloader)
        
            

    def write_outputs(self, output_file, summary, output_type, output_cols, url=True):
        if output_type in {'json', 'jsonl'}:
            with open(output_file, "a") as fp:
                json_writer = jsonlines.Writer(fp)
                for obj in summary:
                    if url:
                        obj['url'] = f"https://en.wikipedia.org/w/index.php?&diff=prev&oldid={obj['rev_id']}"
                    if self.gen_dataset:
                        obj['summary'] = obj['model_completions']
                        del obj['model_completions']
                        del obj['url']
                        obj['prev_texts'] = "\n".join(obj['prev_texts'])
                        obj['cur_texts'] = "\n".join(obj['cur_texts'])
                    json_writer.write(dict([(k, v) for k,v in obj.items() if k in output_cols]))
        elif output_type == 'csv':
            if not self.has_header:
                with open(output_file, "w") as fp:
                    csv_writer = csv.DictWriter(fp, fieldnames=output_cols)
                    csv_writer.writeheader()
                    self.has_header = True
            with open(output_file, "a") as fp:
                csv_writer = csv.DictWriter(fp, fieldnames=output_cols)
                csv_writer.writeheader()
                for obj in summary:
                    if url:
                        obj['url'] = f"https://en.wikipedia.org/w/index.php?&diff=prev&oldid={obj['rev_id']}"
                    if self.gen_dataset:
                        obj['summary'] = obj['model_completions']
                        del obj['model_completions']
                        del obj['url']
                        obj['prev_texts'] = "\n".join(obj['prev_texts'])
                        obj['cur_texts'] = "\n".join(obj['cur_texts'])
                    csv_writer.writerow(dict([(k, v) for k,v in obj.items() if k in output_cols]))