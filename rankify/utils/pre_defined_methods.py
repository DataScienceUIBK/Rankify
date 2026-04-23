
import warnings

# Attempt to import vllm, but do not raise an error if it's missing
try:
    from vllm import LLM, SamplingParams, RequestOutput
    VLLM_AVAILABLE = True
except ModuleNotFoundError:
    warnings.warn("vLLM is not installed. `FirstModelReranker`, `LiT5ScoreReranker`, `LiT5DistillReranker`, `VicunaReranker`, and `ZephyrReranker` will not be available.", UserWarning)
    VLLM_AVAILABLE = False


from rankify.models.upr import UPR
from rankify.models.rankgpt import RankGPT
from rankify.models.flashrank import FlashRanker
from rankify.models.rankt5 import RankT5
from rankify.models.listt5 import ListT5
from rankify.models.inranker import InRanker
from rankify.models.apiranker import APIRanker
from rankify.models.transformer_ranker import TransformerRanker
from rankify.models.llm_layerwise_ranker import LLMLayerWiseRanker
from rankify.models.monot5 import MonoT5
#from rankify.models.first_reranker import FirstModelReranker
#from rankify.models.lit5_reranker import LiT5DistillReranker
#from rankify.models.lit5_reranker import LiT5ScoreReranker
#from rankify.models.vicuna_reranker import VicunaReranker
#from rankify.models.zephyr_reranker import ZephyrReranker
from rankify.models.blender_reranker import BlenderReranker
from rankify.models.splade_reranker import SpladeReranker
from rankify.models.sentence_transformer_reranker import SentenceTransformerReranker
from rankify.models.colbert_ranker import ColBERTReranker
from rankify.models.monobert import MonoBERT
from rankify.models.llm2vec_reranker import LLM2VecReranker
from rankify.models.twolar import TWOLAR
from rankify.models.duot5 import DuoT5
from rankify.models.echorank import EchoRankReranker
from rankify.models.incontext_reranker import InContextReranker


# Conditionally import FirstModelReranker only if vLLM is available
if VLLM_AVAILABLE:
    from rankify.models.first_reranker import FirstModelReranker
    from rankify.models.lit5_reranker import LiT5DistillReranker
    from rankify.models.lit5_reranker import LiT5ScoreReranker
    from rankify.models.vicuna_reranker import VicunaReranker
    from rankify.models.zephyr_reranker import ZephyrReranker

METHOD_MAP ={
    # Existing reranking methods
    'upr': UPR,
    'rankgpt-api':RankGPT,
    'rankgpt':RankGPT,
    'flashrank' : FlashRanker,
    'monot5': MonoT5,
    'rankt5': RankT5,
    'listt5': ListT5,
    'inranker': InRanker,
    'apiranker': APIRanker,
    'transformer_ranker': TransformerRanker,
    'llm_layerwise_ranker':LLMLayerWiseRanker,
    #'first_ranker':FirstModelReranker,
    #'lit5dist': LiT5DistillReranker,
    #'lit5score': LiT5ScoreReranker,
    #'vicuna_reranker': VicunaReranker,
    #'zephyr_reranker': ZephyrReranker,
    'blender_reranker': BlenderReranker,
    'splade_reranker': SpladeReranker,
    'sentence_transformer_reranker': SentenceTransformerReranker,
    'colbert_ranker': ColBERTReranker,
    'monobert': MonoBERT,
    'llm2vec': LLM2VecReranker,
    'twolar':TWOLAR,
    'echorank':EchoRankReranker,
    'duot5': DuoT5,
    'incontext_reranker': InContextReranker,
}


# Only add `first_ranker` if `FirstModelReranker` is available
if VLLM_AVAILABLE:
    METHOD_MAP['first_ranker'] = FirstModelReranker
    METHOD_MAP['lit5dist']= LiT5DistillReranker
    METHOD_MAP['lit5score']= LiT5ScoreReranker
    METHOD_MAP['vicuna_reranker']=  VicunaReranker
    METHOD_MAP['zephyr_reranker']=  ZephyrReranker