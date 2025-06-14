import re
import numpy as np
import collections
import math
from functools import lru_cache
from collections import Counter
import tempfile
import subprocess
import os 
def normalize_answer(s):
    """
    Normalizes an answer string by **removing punctuation, articles,** and **extra whitespace**.

    Args:
        s (str): The answer string.

    Returns:
        str: Normalized answer string.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punctuation(text):
        return re.sub(r'[^a-z0-9 ]', '', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    return white_space_fix(remove_articles(remove_punctuation(s.lower())))


class BaseMetric:
    """
    Base class for **evaluation metrics**.

    Attributes:
        metric_name (str): Name of the metric.
        config (dict): Configuration dictionary.
        dataset_name (str): Name of the dataset used for evaluation.

    Methods:
        calculate_metric(data): Computes the evaluation metric.
        get_dataset_answer(data): Extracts ground-truth answers from dataset.
    """
    metric_name = "base"

    def __init__(self, config):
        """
        Initializes the base metric.

        Args:
            config (dict): Configuration dictionary.
        """
        self.config = config
        self.dataset_name = config.get("dataset_name", "default")

    def calculate_metric(self, data):
        """
        Calculates the metric.

        Args:
            data: Data object containing predictions and ground truth.

        Returns:
            tuple: Metric scores and individual scores for each instance.
        """
        return {}, []

    # def get_dataset_answer(self, data):
    #     """
    #     Extracts ground-truth answers from dataset.

    #     Args:
    #         data: Data object containing documents.

    #     Returns:
    #         list: List of ground-truth answers.
    #     """
    #     answers_list = []
    #     for doc in data.documents:
    #         ans = doc.answers
    #         if hasattr(ans, "answers"):
    #             answers_list.append(ans.answers)
    #         elif isinstance(ans, (list, tuple)):
    #             answers_list.append(list(ans))
    #         elif isinstance(ans, str):
    #             answers_list.append([ans])
    #         else:
    #             answers_list.append([str(ans)])
    #     return answers_list
    def get_dataset_answer(self, data):
        """
        Extracts ground-truth answers from dataset.

        Args:
            data: Data object containing documents.

        Returns:
            list: List of ground-truth answers.
        """
        #print(data.doc)
        return [doc.answers.answers for doc in data.documents]
### **Generation Metrics (Exact Match, F1, Precision, Recall, BLEU)**

class ExactMatch(BaseMetric):
    """
    Computes **Exact Match (EM) Score**.

    The **Exact Match** metric checks whether the **predicted answer exactly matches** one of the ground-truth answers.
    """
    metric_name = "exact_match"

    def calculate_em(self, prediction, golden_answers):
        """
        Computes Exact Match score.

        Args:
            prediction (str): The predicted answer.
            golden_answers (list): List of reference answers.

        Returns:
            float: **1.0** if there is an exact match, else **0.0**.
        """

        normalized_prediction = normalize_answer(prediction)
        for answer in golden_answers:
            if normalize_answer(answer) == normalized_prediction:
                return 1.0
        return 0.0

    def calculate_metric(self, data):
        """
        Computes the **Exact Match (EM) score**.

        Args:
            data: Data object containing **predictions** and **ground-truth answers**.

        Returns:
            dict: **{"exact_match": score}**
            list: List of EM scores per prediction.
        """
        pred_list = data.predictions
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.calculate_em(pred, golden_answers)
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"exact_match": score *100 }, metric_score_list


class F1Score(BaseMetric):
    """
    Computes **F1 Score**, **Precision**, and **Recall**.

    The **F1 Score** is the harmonic mean of **precision** and **recall**, commonly used in QA evaluation.
    """
    metric_name = "f1_score"

    def token_level_scores(self, prediction, ground_truths):
        """
        Computes **F1, Precision, and Recall** scores for a **prediction-ground-truth pair**.

        Args:
            prediction (str): The predicted answer.
            ground_truths (list): List of reference answers.

        Returns:
            dict: **{"f1": score, "precision": score, "recall": score}**
        """
        final_metric = {"f1": 0, "precision": 0, "recall": 0}
        for ground_truth in ground_truths:
            pred_tokens = normalize_answer(prediction).split()
            truth_tokens = normalize_answer(ground_truth).split()
            common = Counter(pred_tokens) & Counter(truth_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                continue

            precision = num_same / len(pred_tokens)
            recall = num_same / len(truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

            final_metric["f1"] = max(f1, final_metric["f1"])
            final_metric["precision"] = max(precision, final_metric["precision"])
            final_metric["recall"] = max(recall, final_metric["recall"])

        return final_metric

    def calculate_metric(self, data):
        pred_list = data.predictions
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["f1"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"f1_score": score*100 }, metric_score_list


class PrecisionScore(F1Score):
    """
    Computes **Precision Score**.

    Precision is the fraction of retrieved documents that are **relevant**.
    """
    metric_name = "precision"

    def calculate_metric(self, data):
        pred_list = data.predictions
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["precision"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"precision": score*100}, metric_score_list


class RecallScore(F1Score):
    """
    Computes **Recall Score**.

    Recall is the fraction of relevant documents that are **retrieved**.
    """
    metric_name = "recall"

    def calculate_metric(self, data):
        pred_list = data.predictions
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["recall"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"recall": score*100}, metric_score_list

class ContainsMatch(BaseMetric):
    """
    Computes **Contains Match** metric.

    This metric checks whether any reference answer is **contained within** the predicted answer.
    """
    metric_name = "contains_match"

    def calculate_contains(self, prediction, golden_answers):
        normalized_prediction = normalize_answer(prediction)
        for answer in golden_answers:
            if normalize_answer(answer) in normalized_prediction:
                return 1.0  # The prediction contains the answer
        return 0.0

    def calculate_metric(self, data):
        pred_list = data.predictions
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.calculate_contains(pred, golden_answers)
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"contains_match": score*100}, metric_score_list
    

class BLEUScore(BaseMetric):
    """
    Computes **BLEU Score** for evaluating text generation.

    BLEU (**Bilingual Evaluation Understudy**) measures the similarity between **machine-generated text**
    and **reference translations** by analyzing overlapping **n-grams**.

    References
    ----------
    - Papineni et al. (2002): BLEU: A Method for Automatic Evaluation of Machine Translation.

    Attributes:
        metric_name (str): The metric name (`"bleu_score"`).
        max_order (int): Maximum **n-gram order** to consider (default: `4`).
        smooth (bool): Whether to apply **smoothing** to prevent zero precision (default: `False`).

    Methods:
        compute_bleu(reference_corpus, translation_corpus): Computes BLEU score.
        _get_ngrams(segment, max_order): Extracts **n-grams** from text.
        calculate_metric(data): Computes BLEU score for **generated text**.
    """
    metric_name = "bleu_score"

    def __init__(self, config):
        """
        Initializes the **BLEU Score** metric.

        Args:
            config (dict): Configuration dictionary containing:
                - `"bleu_max_order"` (int, optional): Maximum **n-gram order** (default: `4`).
                - `"bleu_smooth"` (bool, optional): Whether to apply **smoothing** (default: `False`).
        """
        super().__init__(config)
        self.max_order = config.get("bleu_max_order", 4)
        self.smooth = config.get("bleu_smooth", False)

    def compute_bleu(self, reference_corpus, translation_corpus):
        """
        Computes **BLEU Score** by comparing **generated translations** with **reference translations**.

        Args:
            reference_corpus (List[List[List[str]]]): 
                A **list of reference translations**, where each reference is a list of tokenized reference sentences.
            translation_corpus (List[List[str]]): 
                A **list of model-generated translations**, tokenized into words.

        Returns:
            float: BLEU score (**0 to 1** range, higher is better).
        """
        matches_by_order = [0] * self.max_order
        possible_matches_by_order = [0] * self.max_order
        reference_length = 0
        translation_length = 0

        for references, translation in zip(reference_corpus, translation_corpus):
            reference_length += min(len(r) for r in references)
            translation_length += len(translation)

            merged_ref_ngram_counts = collections.Counter()
            for reference in references:
                merged_ref_ngram_counts |= self._get_ngrams(reference, self.max_order)
            translation_ngram_counts = self._get_ngrams(translation, self.max_order)
            overlap = translation_ngram_counts & merged_ref_ngram_counts
            for ngram in overlap:
                matches_by_order[len(ngram) - 1] += overlap[ngram]
            for order in range(1, self.max_order + 1):
                possible_matches = len(translation) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order - 1] += possible_matches

        precisions = [0] * self.max_order
        for i in range(self.max_order):
            if self.smooth:
                precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
            else:
                precisions[i] = matches_by_order[i] / possible_matches_by_order[i] if possible_matches_by_order[i] > 0 else 0.0

        bleu = math.exp(sum(math.log(p) for p in precisions) / self.max_order) if min(precisions) > 0 else 0.0
        return bleu
    
    def _get_ngrams(self, segment, max_order):
        """
        Extracts **n-grams** from a given text segment up to a specified **n-gram order**.

        Args:
            segment (List[str]): Tokenized text segment.
            max_order (int): Maximum **n-gram** length.

        Returns:
            collections.Counter: A counter containing **n-grams** and their occurrences.
        """
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i : i + order])
                ngram_counts[ngram] += 1
        return ngram_counts

    def calculate_metric(self, data):
        """
        Computes **BLEU Score** for generated text compared to reference answers.

        Args:
            data: A **data object** containing:
                - `data.predictions` (List[str]): List of **generated answers**.
                - `data.documents` (List[Document]): List of **documents** with ground-truth answers.

        Returns:
            Tuple:
                - dict: **{"bleu_score": score}**
                - List[float]: BLEU scores for each instance.
        """
        pred_list = [normalize_answer(pred).split() for pred in data.predictions]
        golden_answers_list = [[normalize_answer(ans).split() for ans in ans_list] for ans_list in self.get_dataset_answer(data)]

        bleu_score = self.compute_bleu(golden_answers_list, pred_list)
        return {"bleu_score": bleu_score}, [bleu_score] * len(pred_list)



class Metrics:
    """
    Computes **Retrieval & Generation Metrics**.

    Attributes:
        documents (list): The list of **Document** instances.
        config (dict): Configuration dictionary.

    Methods:
        top_k_accuracy(k, use_reordered): Computes **Top-K Accuracy**.
        calculate_retrieval_metrics(ks, use_reordered): Computes **retrieval performance at multiple K**.
        calculate_generation_metrics(predictions): Computes **generation evaluation scores**.
    """
    QREL_MAPPING = {
        'dl19': 'dl19-passage',
        'dl20': 'dl20-passage',
        'covid': 'beir-v1.0.0-trec-covid-test',
        'arguana': 'beir-v1.0.0-arguana-test',
        'touche': 'beir-v1.0.0-webis-touche2020-test',
        'news': 'beir-v1.0.0-trec-news-test',
        'scifact': 'beir-v1.0.0-scifact-test',
        'fiqa': 'beir-v1.0.0-fiqa-test',
        'scidocs': 'beir-v1.0.0-scidocs-test',
        'nfc': 'beir-v1.0.0-nfcorpus-test',
        'quora': 'beir-v1.0.0-quora-test',
        'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
        'fever': 'beir-v1.0.0-fever-test',
        'robust04': 'beir-v1.0.0-robust04-test',
        'signal': 'beir-v1.0.0-signal1m-test',
    }
    def __init__(self, documents):
        """
        Initializes the **Metrics** class.

        Args:
            documents (list): List of **Document** instances.
        """
        self.documents = documents
        print()
        self.config = {"dataset_name": "QA_Evaluation"}

    def top_k_accuracy(self, k, use_reordered=False):
        """
        Computes **Top-K retrieval accuracy**.

        Args:
            k (int): The **K value** (e.g., Top-5, Top-10).
            use_reordered (bool, optional): Whether to use **reranked contexts**.

        Returns:
            float: **Top-K accuracy** (percentage).
        """
        hits, total = 0, 0
        for document in self.documents:
            contexts_to_use = document.reorder_contexts if (use_reordered and document.reorder_contexts) else document.contexts
            if any(context.has_answer for context in contexts_to_use[:k]):
                hits += 1
            total += 1
        return (hits / total) * 100 if total > 0 else 0

    def calculate_retrieval_metrics(self, ks=[1, 5, 10, 20, 50, 100], use_reordered=False):
        """
        Computes **Top-K retrieval metrics** for multiple values of **K**.

        Args:
            ks (list, optional): List of K values (default: `[1, 5, 10, 20, 50, 100]`).
            use_reordered (bool, optional): Whether to use **reranked results**.

        Returns:
            dict: Dictionary with **Top-K accuracy scores**.
        """
        results = {}
        for k in ks:
            results[f'top_{k}'] = self.top_k_accuracy(k, use_reordered)
        return results

    def calculate_generation_metrics(self, predictions):
        """
        Computes **generation evaluation metrics**.

        Args:
            predictions (list): List of model predictions.

        Returns:
            dict: Dictionary with **Exact Match, F1, Precision, Recall, Contains Match** scores.
        """
        data = type("Data", (object,), {"documents": self.documents, "predictions": predictions})()
        metric_classes = [ExactMatch, F1Score, PrecisionScore, RecallScore, ContainsMatch]

        results = {}
        for metric_class in metric_classes:
            metric = metric_class(self.config)
            score, _ = metric.calculate_metric(data)
            results.update(score)

        return results
    def generate_trec_format(self, use_reordered= False):
        """
        Converts **Documents** into **TREC format** for evaluation.

        Returns:
            str: String formatted in **TREC format**.
        """
        trec_results = []
        if use_reordered:
            name= "rankify"
        else:
            name="bm25"
        for doc in self.documents:
            
            contexts_to_use = doc.reorder_contexts if (use_reordered and doc.reorder_contexts) else doc.contexts

            for rank, context in enumerate(contexts_to_use):
                trec_results.append(f"{doc.id} Q0 {context.id} {rank + 1} {context.score} {name}")
        
        return "\n".join(trec_results)

    def calculate_trec_metrics(self, ndcg_cuts=[10], map_cuts=[100], mrr_cuts=[10], qrel='dl19', use_reordered=False):
        """
        Computes **NDCG, MAP, and MRR** using Pyserini's `trec_eval` command-line tool.

        Args:
            ndcg_cuts (list, optional): List of **NDCG@k** values (default: `[10]`).
            map_cuts (list, optional): List of **MAP@k** values (default: `[100]`).
            mrr_cuts (list, optional): List of **MRR@k** values (default: `[10]`).
            qrel (str, optional): The **dataset key** (default: `'dl19'`).
            use_reordered (bool, optional): Whether to use **reranked contexts**.

        Returns:
            dict: Dictionary containing **NDCG@k, MAP@k, and MRR@k** scores.
        """
        if os.path.exists(qrel):
            qrel_path = qrel  # user gave a custom path like "./qrels.txt"
        elif qrel in self.QREL_MAPPING:
            qrel_path = self.QREL_MAPPING[qrel]
        else:
            raise ValueError(f"Invalid qrel: {qrel}. Must be a known key or a path to qrels.txt.")

        results = {}

        with tempfile.NamedTemporaryFile(delete=False, mode="w") as trec_file:
            trec_file.write(self.generate_trec_format(use_reordered))
            trec_file_path = trec_file.name

        # Path to the TREC qrels file (ground-truth relevance file)
        # Ensure `trec_eval` is installed and accessible
        trec_eval_cmd = "python -m pyserini.eval.trec_eval"

        for k in ndcg_cuts:
            #cmd = f"{trec_eval_cmd} -c -m ndcg_cut.{k} {self.QREL_MAPPING[qrel]} {trec_file_path}"
            cmd = f"{trec_eval_cmd} -c -m ndcg_cut.{k} {qrel_path} {trec_file_path}"

            output = self.run_trec_eval(cmd)
            results[f"ndcg@{k}"] = output

        for k in map_cuts:
            #cmd = f"{trec_eval_cmd} -c -m map_cut.{k} {self.QREL_MAPPING[qrel]} {trec_file_path}"
            cmd = f"{trec_eval_cmd} -c -m map_cut.{k} {qrel_path} {trec_file_path}"

            output = self.run_trec_eval(cmd)
            results[f"map@{k}"] = output

        for k in mrr_cuts:
            #cmd = f"{trec_eval_cmd} -c -m recip_rank {self.QREL_MAPPING[qrel]} {trec_file_path}"
            cmd = f"{trec_eval_cmd} -c -m recip_rank {qrel_path} {trec_file_path}"

            output = self.run_trec_eval(cmd)
            results[f"mrr@{k}"] = output

        # Clean up temporary file
        os.remove(trec_file_path)

        return results

    def run_trec_eval(self, command):
        """
        Runs `trec_eval` and extracts the metric score.

        Args:
            command (str): The `trec_eval` command.

        Returns:
            float: Extracted metric score.
        """
        try:
            output = subprocess.run(command.split(), capture_output=True, text=True, check=True)
            return self.parse_trec_output(output.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running `trec_eval`: {e.stderr}")
            return 0.0  # Return 0 if the command fails

    def parse_trec_output(self, output):
        """
        Parses the **output** from `trec_eval`.

        Args:
            output (str): Raw text output from `trec_eval`.

        Returns:
            float: Extracted metric score.
        """
        lines = output.strip().split("\n")
        if not lines:
            print("Warning: `trec_eval` output is empty. Check input files.")
            return 0.0

        last_line = lines[-1].split()
        if len(last_line) < 3:
            print(f"Warning: Unexpected `trec_eval` output format:\n{output}")
            return 0.0

        try:
            return float(last_line[2])  # Extract the metric score
        except ValueError:
            print(f"Error parsing metric score from:\n{output}")
            return 0.0