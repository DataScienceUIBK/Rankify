import os
import json
from typing import Optional, Tuple, List, Dict, Union,Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch
import numpy as np
from ftfy import fix_text
from transformers.generation import GenerationConfig
from vllm import LLM, SamplingParams, RequestOutput
import copy
import random
import re
import unicodedata
from abc import ABC, abstractmethod
from enum import Enum

ALPH_START_IDX = ord('A') - 1


class PromptMode(Enum):
    UNSPECIFIED = "unspecified"
    RANK_GPT = "rank_GPT"
    LRL = "LRL"

    def __str__(self):
        return self.value


class RankLLM(ABC):
    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode,
        num_few_shot_examples: int,
    ) -> None:
        self._model = model
        self._context_size = context_size
        self._prompt_mode = prompt_mode
        self._num_few_shot_examples = num_few_shot_examples

    def max_tokens(self) -> int:
        """
        Returns the maximum number of tokens for a given model

        Returns:
            int: The maximum token count.
        """
        return self._context_size

    @abstractmethod
    def run_llm(self, prompt: Union[str, List[Dict[str, str]]]) -> Tuple[str, int]:
        """
        Abstract method to run the target language model with a passed in prompt.

        Args:
            prompt (Union[str, List[Dict[str, str]]]): The prompt to be processed by the model.

        Returns:
            Tuple[str, int]: A tuple object containing the text response and the number of tokens in the response.
        """
        pass

    @abstractmethod
    def create_prompt_batched(
        self, results, rank_start: int, rank_end: int, batch_size: int
    ) -> List[Tuple[Union[str, List[Dict[str, str]]], int]]:
        """
        Abstract method to create a batch of prompts based on the results and given ranking range.

        Args:
            results (List[Result]): The list of result objects containing data for prompt generation.
            rank_start (int): The starting rank for prompt generation.
            rank_end (int): The ending rank for prompt generation.

        Returns:
            Tuple[List[Union[str, List[Dict[str, str]]], List[int]]: A tuple object containing the list of generated prompts and the list of number of tokens in the generated prompts.
        """
        pass

    @abstractmethod
    def create_prompt(
        self, result, rank_start: int, rank_end: int
    ) -> Tuple[Union[str, List[Dict[str, str]]], int]:
        """
        Abstract method to create a prompt based on the result and given ranking range.

        Args:
            result (Result): The result object containing data for prompt generation.
            rank_start (int): The starting rank for prompt generation.
            rank_end (int): The ending rank for prompt generation.

        Returns:
            Tuple[Union[str, List[Dict[str, str]]], int]: A tuple object containing the generated prompt and the number of tokens in the generated prompt.
        """
        pass

    @abstractmethod
    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """
        Abstract method to calculate the number of tokens contained in the given prompt.

        Args:
            prompt (Union[str, List[Dict[str, str]]]): The prompt for which to compute the token count for.

        Returns:
            int: The number of tokens in the given prompt.
        """
        pass

    @abstractmethod
    def cost_per_1k_token(self, input_token: bool) -> float:
        """
        Abstract method to calculate the cost per 1,000 tokens for the target language model.

        Args:
            input_token (bool): Flag to indicate if the cost is for input tokens or output tokens.

        Returns:
            float: The cost per 1,000 tokens.
        """
        pass

    @abstractmethod
    def num_output_tokens(self) -> int:
        """
        Abstract method to estimate the number of tokens in the model's output, constrained by max tokens for the target language model.

        Returns:
            int: The estimated number of output tokens.
        """
        pass

    def permutation_pipeline(
        self,
        result,
        use_logits: bool,
        use_alpha: bool,
        rank_start: int,
        rank_end: int,
        logging: bool = False,
    ):
        """
        Runs the permutation pipeline on the passed in result set within the passed in rank range.

        Args:
            result (Result): The result object to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            Result: The processed result object after applying permutation.
        """
        prompt, in_token_count = self.create_prompt(result, use_alpha, rank_start, rank_end)
        if logging:
            print(f"prompt: {prompt}\n")
        permutation, out_token_count = self.run_llm(
            prompt, use_logits=use_logits, use_alpha=use_alpha, current_window_size=rank_end - rank_start
        )
        if logging:
            print(f"output: {permutation}")
        ranking_exec_info = RankingExecInfo(
            prompt, permutation, in_token_count, out_token_count
        )
        if result.ranking_exec_summary == None:
            result.ranking_exec_summary = []
        result.ranking_exec_summary.append(ranking_exec_info)
        result = self.receive_permutation(result, permutation, rank_start, rank_end, use_alpha)

        
        return result
    
    def permutation_pipeline_batched(
        self,
        results,
        use_logits: bool,
        use_alpha: bool,
        rank_start: int,
        rank_end: int,
        logging: bool = False,
    ):
        """
        Runs the permutation pipeline on the passed in result set within the passed in rank range for a batch of results.
        Args:
            results (List[Result]): The list of result objects to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.
        Returns:
            List[Result]: The processed list of result objects after applying permutation.
        """
        prompts = []
        prompts = self.create_prompt_batched(results, use_alpha, rank_start, rank_end, batch_size=32)
        batched_results = self.run_llm_batched([prompt for prompt, _ in prompts], use_logits=use_logits, use_alpha=use_alpha, current_window_size=rank_end - rank_start)
        #---------------------------------
        for index, (result, (prompt, in_token_count)) in enumerate(zip(results, prompts)):
            permutation, out_token_count = batched_results[index]
            if logging:
                print(f"output: {permutation}")
            ranking_exec_info = RankingExecInfo(
                prompt, permutation, in_token_count, out_token_count
            )
            if result.ranking_exec_summary is None:
                result.ranking_exec_summary = []
            result.ranking_exec_summary.append(ranking_exec_info)
            result = self.receive_permutation(result, permutation, rank_start, rank_end, use_alpha)
        #print(results , "aaaaaaaaaaaa")
        
        #aaaaaa
        return results

    def sliding_windows(
        self,
        retrieved_result,
        use_logits: bool,
        use_alpha: bool,
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
        logging: bool = False,
    ):
        """
        Applies the sliding window algorithm to the reranking process.

        Args:
            retrieved_result (Result): The result object to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            Result: The result object after applying the sliding window technique.
        """
        rerank_result = copy.deepcopy(retrieved_result)
        end_pos = rank_end
        start_pos = rank_end - window_size
        # end_pos > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
        # start_pos + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
        while end_pos > rank_start and start_pos + step != rank_start:
            start_pos = max(start_pos, rank_start)
            rerank_result = self.permutation_pipeline(
                rerank_result, use_logits, use_alpha, start_pos, end_pos, logging
            )
            end_pos = end_pos - step
            start_pos = start_pos - step
        #print("====================================")
        #print(rerank_result)
        #print("====================================")
        return rerank_result
    
    def sliding_windows_batched(
        self,
        retrieved_results,
        use_logits: bool,
        use_alpha: bool,
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
        logging: bool = False,
    ):
        """
        Applies the sliding window algorithm to the reranking process for a batch of result objects.
        Args:
            retrieved_results (List[Result]): The list of result objects to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.
        Returns:
            List[Result]: The list of result objects after applying the sliding window technique.
        """
        rerank_results = [copy.deepcopy(result) for result in retrieved_results]

        end_pos = rank_end
        start_pos = rank_end - window_size
        
        # end_pos > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
        # start_pos + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
        while end_pos > rank_start and start_pos + step != rank_start:
            start_pos = max(start_pos, rank_start)
            rerank_results = self.permutation_pipeline_batched(
                rerank_results, use_logits, use_alpha, start_pos, end_pos, logging
            )
            end_pos = end_pos - step
            start_pos = start_pos - step
        return rerank_results

    def get_ranking_cost_upperbound(
        self, num_q: int, rank_start: int, rank_end: int, window_size: int, step: int
    ) -> Tuple[float, int]:
        """
        Calculates the upper bound of the ranking cost for a given set of parameters.

        Args:
            num_q (int): The number of queries.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.

        Returns:
            Tuple[float, int]: A tuple object containing the cost and the total number of tokens used (input tokens + output tokens).
        """
        # For every prompt generated for every query assume the max context size is used.
        num_promt = (rank_end - rank_start - window_size) / step + 1
        input_token_count = (
            num_q * num_promt * (self._context_size - self.num_output_tokens())
        )
        output_token_count = num_q * num_promt * self.num_output_tokens()
        cost = (
            input_token_count * self.cost_per_1k_token(input_token=True)
            + output_token_count * self.cost_per_1k_token(input_token=False)
        ) / 1000.0
        return (cost, input_token_count + output_token_count)

    def get_ranking_cost(
        self,
        retrieved_results: List[Dict[str, Any]],
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
    ) -> Tuple[float, int]:
        """
        Calculates the ranking cost based on actual token counts from generated prompts.

        Args:
            retrieved_results (List[Dict[str, Any]]): A list of retrieved results for processing.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.

        Returns:
            Tuple[float, int]: A tuple object containing the calculated cost and the total number of tokens used (input tokens + output tokens).
        """
        input_token_count = 0
        output_token_count = 0
        # Go through the retrieval result using the sliding window and count the number of tokens for generated prompts.
        # This is an estimated cost analysis since the actual prompts' length will depend on the ranking.
        for result in retrieved_results:
            end_pos = rank_end
            start_pos = rank_end - window_size
            while start_pos >= rank_start:
                start_pos = max(start_pos, rank_start)
                prompt, _ = self.create_prompt(result, start_pos, end_pos)
                input_token_count += self.get_num_tokens(prompt)
                end_pos = end_pos - step
                start_pos = start_pos - step
                output_token_count += self.num_output_tokens()
        cost = (
            input_token_count * self.cost_per_1k_token(input_token=True)
            + output_token_count * self.cost_per_1k_token(input_token=False)
        ) / 1000.0
        return (cost, input_token_count + output_token_count)

    def _clean_response(self, response: str, use_alpha: bool) -> str:
        new_response = ""
        if use_alpha:
            for c in response:
                if not c.isalpha():
                    new_response += " "
                else:
                    new_response += str(ord(c) - ALPH_START_IDX)
            new_response = new_response.strip()
        else:
            for c in response:
                if not c.isdigit():
                    new_response += " "
                else:
                    new_response += c
            new_response = new_response.strip()
            
        return new_response

    def _remove_duplicate(self, response: List[int]) -> List[int]:
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    def receive_permutation(
        self, result, permutation: str, rank_start: int, rank_end: int, use_alpha: bool
    ):
        """
        Processes and applies a permutation to the ranking results.

        This function takes a permutation string, representing the new order of items,
        and applies it to a subset of the ranking results. It adjusts the ranks and scores in the
        'result' object based on this permutation.

        Args:
            result (Result): The result object containing the initial ranking results.
            permutation (str): A string representing the new order of items.
                            Each item in the string should correspond to a rank in the results.
            rank_start (int): The starting index of the range in the results to which the permutation is applied.
            rank_end (int): The ending index of the range in the results to which the permutation is applied.

        Returns:
            Result: The updated result object with the new ranking order applied.

        Note:
            This function assumes that the permutation string is a sequence of integers separated by spaces.
            Each integer in the permutation string corresponds to a 1-based index in the ranking results.
            The function first normalizes these to 0-based indices, removes duplicates, and then reorders
            the items in the specified range of the 'result.hits' list according to the permutation.
            Items not mentioned in the permutation string remain in their original sequence but are moved after
            the permuted items.
        """
        response = self._clean_response(permutation, use_alpha)
        response = [int(x) - 1 for x in response.split()]
        response = self._remove_duplicate(response)
        cut_range = copy.deepcopy(result.hits[rank_start:rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response]
        #print("==================")
        #print(response)
        #print("==================")
        #asdasdada
        
        for j, x in enumerate(response):
            result.hits[j + rank_start] = copy.deepcopy(cut_range[x])
            #print(result.hits[j + rank_start])
            #asd
            if "rank" in result.hits[j + rank_start]:
                result.hits[j + rank_start]["rank"] = cut_range[j]["rank"]
            if "score" in result.hits[j + rank_start]:
                result.hits[j + rank_start]["score"] = cut_range[j]["score"]
        #print("result: ", result.hits)
        return result

    def _replace_number(self, s: str, use_alpha) -> str:
        if use_alpha:
            return re.sub(r"\[([A-z]+)\]", r"(\1)", s)
        else:
            return re.sub(r"\[(\d+)\]", r"(\1)", s)
        
class RankListwiseOSLLM(RankLLM):
    def __init__(
        self,
        model: str,
        context_size: int = 4096,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = False,
        window_size: int = 20,
        system_message: str = None,
        batched: bool = False,
        gpu_memory_utilization: float = 0.9,  # Increased GPU memory usage
        max_model_len: int = 29168  # Limit max model length within cache limit
    ) -> None:
        super().__init__(model, context_size, prompt_mode, num_few_shot_examples)
        self._device = device
        if self._device == "cuda":
            assert torch.cuda.is_available(), "CUDA is not available on this device"
        if prompt_mode != PromptMode.RANK_GPT:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. Only RANK_GPT is supported."
            )

        # Initialize the LLM model with custom memory and length configurations
        self._llm = LLM(
            model=model, 
            max_logprobs=30, 
            enforce_eager=False,
            gpu_memory_utilization=gpu_memory_utilization,  # Set memory utilization
            max_model_len=max_model_len  # Set max sequence length
        )
        
        self._tokenizer = self._llm.get_tokenizer()
        self.system_message_supported = "system" in self._tokenizer.chat_template
        self._batched = batched
        self._variable_passages = variable_passages
        self._window_size = window_size
        self._system_message = system_message
        self._output_token_estimate = None

        if num_few_shot_examples > 0:
            with open("data/output_v2_aug_filtered.jsonl", "r") as json_file:
                self._examples = list(json_file)[1:-1]

    def _evaluate_logits(self, logits: Dict[str, 'Logit'], use_alpha: bool, total: Tuple[int, int]) -> Tuple[str, Dict[int, float]]:
        if use_alpha:
            evaluations = {
                ord(logit.decoded_token): logit.logprob
                for logit in logits.values()
                if len(logit.decoded_token) == 1 and 
                   logit.decoded_token.isalpha() and 
                   ALPH_START_IDX + 1 <= ord(logit.decoded_token) <= ALPH_START_IDX + self._window_size
            }
            sorted_evaluations = sorted(evaluations.items(), key=lambda x: -x[1])
            result_string = ">".join([f"[{chr(x)}]" for x, y in sorted_evaluations])
        else:
            evaluations = {
                int(logit.decoded_token): logit.logprob
                for logit in logits.values()
                if logit.decoded_token.isnumeric() and
                   not unicodedata.name(logit.decoded_token).startswith(('SUPERSCRIPT', 'VULGAR FRACTION', 'SUBSCRIPT')) and
                   total[0] <= int(logit.decoded_token) <= total[1]
            }
            sorted_evaluations = sorted(evaluations.items(), key=lambda x: -x[1])
            result_string = ">".join([f"[{x}]" for x, y in sorted_evaluations])

        return result_string, evaluations

    def _get_logits_single_digit(self, output: RequestOutput, use_alpha: bool = False, effective_location: int = 1, total: Tuple[int, int] = (1, 9)):
        logits = output.outputs[0].logprobs[effective_location]
        return self._evaluate_logits(logits, use_alpha, total)

    def run_llm_batched(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        current_window_size: Optional[int] = None,
        use_logits: bool = False,
        use_alpha: bool = False,
    ) -> List[Tuple[str, int]]:
        if current_window_size is None:
            current_window_size = self._window_size

        temp = 0.0
        if use_logits:
            params = SamplingParams(
                min_tokens=2,
                max_tokens=2, 
                temperature=temp,
                logprobs=30,
            )
            outputs = self._llm.generate(prompts, sampling_params=params, use_tqdm=True)
            arr = [self._get_logits_single_digit(output, use_alpha=use_alpha) for output in outputs]
            return [(s, len(s)) for s, __ in arr]
        else:
            params = SamplingParams(
                temperature=temp,
                max_tokens=self.num_output_tokens(use_alpha, current_window_size),
                min_tokens=self.num_output_tokens(use_alpha, current_window_size),
            )
            outputs = self._llm.generate(prompts, sampling_params=params, use_tqdm=True)
            return [
                (output.outputs[0].text, len(output.outputs[0].token_ids))
                for output in outputs
            ]

    def run_llm(
        self, prompt: str, current_window_size: Optional[int] = None, use_logits: bool = False, use_alpha: bool = False
    ) -> Tuple[str, int]:
        if current_window_size is None:
            current_window_size = self._window_size

        temp = 0.0
        if use_logits:
            params = SamplingParams(min_tokens=1, max_tokens=1, temperature=temp, logprobs=30)
            output = self._llm.generate([prompt+"["], sampling_params=params, use_tqdm=False)[0]
            s, _ = self._get_logits_single_digit(output, effective_location=0, use_alpha=use_alpha)
            return s, len(s)
        else:
            max_new_tokens = self.num_output_tokens(use_alpha, current_window_size)
            params = SamplingParams(min_tokens=max_new_tokens, max_tokens=max_new_tokens, temperature=temp)
            output = self._llm.generate([prompt], sampling_params=params, use_tqdm=False)[0]
            return output.outputs[0].text, len(output.outputs[0].text)

    def num_output_tokens(self, use_alpha: bool, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size

        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate

        if use_alpha:
            token_str = " > ".join([f"[{i+1}]" for i in range(current_window_size)])
        else:
            token_str = " > ".join([f"[{chr(ALPH_START_IDX+i+1)}]" for i in range(current_window_size)])

        _output_token_estimate = len(self._tokenizer.encode(token_str)) - 1

        if self._window_size == current_window_size:
            self._output_token_estimate = _output_token_estimate

        return _output_token_estimate

    def _add_prefix_prompt(self, use_alpha, query: str, num: int) -> str:
        if use_alpha:
            return f"I will provide you with {num} passages, each indicated by a alphabetical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"
        else:
            return f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"

    def _add_post_prompt(self, use_alpha, query: str, num: int) -> str:
        if use_alpha:
            example_ordering = "[B] > [A]" if self._variable_passages else "[D] > [B]"
        else:
            example_ordering = "[2] > [1]" if self._variable_passages else "[4] > [2]"
        return f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}, Only respond with the ranking results, do not say any word or explain."

    def _add_few_shot_examples(self, conv):
        for _ in range(self._num_few_shot_examples):
            ex = random.choice(self._examples)
            obj = json.loads(ex)
            prompt = obj["conversations"][0]["value"]
            response = obj["conversations"][1]["value"]
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], response)
        return conv
    
    def _add_few_shot_examples_messages(self, messages):
        for _ in range(self._num_few_shot_examples):
            ex = random.choice(self._examples)
            obj = json.loads(ex)
            prompt = obj["conversations"][0]["value"]
            response = obj["conversations"][1]["value"]
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": response})
        return messages

    def create_prompt(self, result, use_alpha: bool, rank_start: int, rank_end: int) -> Tuple[str, int]:
        query = result.query
        num = len(result.hits[rank_start:rank_end])
        max_length = 300
        while True:
            messages = list()
            if self._system_message and self.system_message_supported:
                messages.append({"role": "system", "content": self._system_message})
            messages = self._add_few_shot_examples_messages(messages)
            prefix = self._add_prefix_prompt(use_alpha, query, num)
            rank = 0
            input_context = f"{prefix}\n"
            for hit in result.hits[rank_start:rank_end]:
                rank += 1
                content = hit["content"].replace("Title: Content: ", "").strip()
                content = " ".join(content.split()[:max_length])
                identifier = chr(ALPH_START_IDX + rank) if use_alpha else str(rank)
                input_context += f"[{identifier}] {self._replace_number(content, use_alpha)}\n"
            input_context += self._add_post_prompt(use_alpha, query, num)
            messages.append({"role": "user", "content": input_context})
            if self._system_message and not self.system_message_supported:
                messages[0]["content"] = self._system_message + "\n " + messages[0]["content"]
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt = fix_text(prompt)
            num_tokens = self.get_num_tokens(prompt)
            if num_tokens <= self.max_tokens() - self.num_output_tokens(rank_end - rank_start):
                break
            else:
                max_length -= max(
                    1,
                    (
                        num_tokens - self.max_tokens() + self.num_output_tokens(rank_end - rank_start)
                    ) // ((rank_end - rank_start) * 4),
                )
        return prompt, num_tokens

    def create_prompt_batched(
        self,
        results,
        use_alpha: bool,
        rank_start: int,
        rank_end: int,
        batch_size: int = 32,
    ) -> List[Tuple[str, int]]:
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        all_completed_prompts = []

        with ThreadPoolExecutor() as executor:
            for batch in chunks(results, batch_size):
                completed_prompts = list(
                    executor.map(
                        lambda result: self.create_prompt(result, use_alpha, rank_start, rank_end),
                        batch,
                    )
                )
                all_completed_prompts.extend(completed_prompts)
        return all_completed_prompts

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0
    

class RankingExecInfo:
    def __init__(
        self, prompt, response: str, input_token_count: int, output_token_count: int
    ):
        self.prompt = prompt
        self.response = response
        self.input_token_count = input_token_count
        self.output_token_count = output_token_count

    def __repr__(self):
        return str(self.__dict__)


class Result:
    def __init__(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        ranking_exec_summary: List[RankingExecInfo] = None,
    ):
        self.query = query
        self.hits = hits
        self.ranking_exec_summary = ranking_exec_summary

    def __repr__(self):
        return str(self.__dict__)


class ResultsWriter:
    def __init__(self, results, append: bool = False):
        self._results = results
        self._append = append

    def write_in_json_format(self, filename: str):
        results = []
        for result in self._results:
            results.append({"query": result.query, "hits": result.hits})
        with open(filename, "a" if self._append else "w") as f:
            json.dump(results, f, indent=2)

class ResultsLoader:
    def __init__(self, filename: str):
        data = json.load(open(filename, 'r'))
        self._results = []
        for item in data:
            hits = []
            for hit in item['hits']:
                hits.append({'qid': hit['qid'], 'docid': hit['docid'], 'score': float(hit['score']), 'content': hit['content']})
            self._results.append(Result(query=item['query'], hits=hits))
    
    def get_results(self, with_context: bool):
        if with_context:
            return self._results
        else:
            results = dict()
            for result in self._results:
                query = result.query
                hits = result.hits
                qid = hits[0]['qid']
                results[qid] = dict()
                for hit in hits:
                    pid = hit['docid']
                    score = hit['score']
                    results[qid][pid] = score
            return results
        


class FirstReranker:
    def __init__(self, agent: RankListwiseOSLLM):
        self.agent = agent

    def rerank(self, retrieved_result: Result, use_logits, use_alpha, rank_start, rank_end, window_size, step, logging=False, batched=False) -> Result:
        if batched:
            return self.agent.sliding_windows_batched(
                retrieved_results=[retrieved_result],
                use_logits=use_logits,
                use_alpha=use_alpha,
                rank_start=rank_start,
                rank_end=rank_end,
                window_size=window_size,
                step=step,
                logging=logging
            )[0]

        return self.agent.sliding_windows(
            retrieved_result=retrieved_result,
            use_logits=use_logits,
            use_alpha=use_alpha,
            rank_start=rank_start,
            rank_end=rank_end,
            window_size=window_size,
            step=step,
            logging=logging
        )

