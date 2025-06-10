from rankify.utils.api.openaiclient import OpenaiClient
from rankify.utils.api.litellmclient import LitellmClient
from rankify.utils.api.claudeclient import ClaudeClient

HF_PRE_DEFIND_MODELS ={
    'upr':{
        't5-small':'google/t5-small-lm-adapt',
        't5-base':'google/t5-base-lm-adapt',
        't5-large':'google/t5-large-lm-adapt',
        't0-3b':'bigscience/T0_3B',
        't0-11b':'bigscience/T0',
        'gpt-neo-2.7b':'EleutherAI/gpt-neo-2.7B',
        'gpt-j-6b':'EleutherAI/gpt-j-6b',
        'gpt2':'openai-community/gpt2',
        'gpt2-medium':'openai-community/gpt2-medium',
        'gpt2-large':'openai-community/gpt2-large',
        'gpt2-xl':'openai-community/gpt2-xl',
        'flan-t5-xl': "google/flan-t5-xl"
        
    },
    'rankgpt-api':{
        'gpt-3.5':'gpt-3.5',
        'gpt-4': 'gpt-4o',
        'gpt-4-mini':'gpt-4o-mini',
        'llamav3.1-8b':'llamav3.1-8b',
        'llamav3.1-70b':'llamav3.1-70b',
        'claude-3-5' : 'claude-3-5'
    },
    'rankgpt':{
        'llamav3.1-8b':'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'llamav3.1-70b':'meta-llama/Llama-3.1-70B-Instruct',
        'Llama-3.2-1B':"meta-llama/Llama-3.2-1B-Instruct",
        'Llama-3.2-3B':"meta-llama/Llama-3.2-3B-Instruct",
        'Qwen2.5-7B':"Qwen/Qwen2.5-7B",
        'Mistral-7B-Instruct-v0.2': 'mistralai/Mistral-7B-Instruct-v0.2',
        "Mistral-7B-Instruct-v0.3":"mistralai/Mistral-7B-Instruct-v0.3",
        
    },

    'flashrank':{
        "ms-marco-TinyBERT-L-2-v2": "ms-marco-TinyBERT-L-2-v2",
        "ms-marco-MiniLM-L-12-v2": "ms-marco-MiniLM-L-12-v2",
        "ms-marco-MultiBERT-L-12": "ms-marco-MultiBERT-L-12",
        "rank-T5-flan": "rank-T5-flan",
        "ce-esci-MiniLM-L12-v2": "ce-esci-MiniLM-L12-v2",
        "rank_zephyr_7b_v1_full": "rank_zephyr_7b_v1_full",
        "miniReranker_arabic_v1": "miniReranker_arabic_v1",
        
    },
    'flashrank-model-file':{
        "ms-marco-TinyBERT-L-2-v2": "flashrank-TinyBERT-L-2-v2.onnx",
        "ms-marco-MiniLM-L-12-v2": "flashrank-MiniLM-L-12-v2_Q.onnx",
        "ms-marco-MultiBERT-L-12": "flashrank-MultiBERT-L12_Q.onnx",
        "rank-T5-flan": "flashrank-rankt5_Q.onnx",
        "ce-esci-MiniLM-L12-v2": "flashrank-ce-esci-MiniLM-L12-v2_Q.onnx",
        "rank_zephyr_7b_v1_full": "rank_zephyr_7b_v1_full.Q4_K_M.gguf",
        "miniReranker_arabic_v1": "miniReranker_arabic_v1.onnx",
        
    },
    'monot5':{
        "monot5-base-msmarco": "castorini/monot5-base-msmarco",
        "monot5-base-msmarco-10k":"castorini/monot5-base-msmarco-10k",
        "monot5-large-msmarco": "castorini/monot5-large-msmarco",
        "monot5-large-msmarco-10k": "castorini/monot5-large-msmarco-10k",
        "monot5-base-med-msmarco": "castorini/monot5-base-med-msmarco",
        "monot5-3b-med-msmarco": "castorini/monot5-3b-med-msmarco",
        "monot5-3b-msmarco-10k": "castorini/monot5-3b-msmarco-10k",
        "mt5-base-en-msmarco": "unicamp-dl/mt5-base-en-msmarco",
        "ptt5-base-pt-msmarco-10k-v2": "unicamp-dl/ptt5-base-pt-msmarco-10k-v2",
        "ptt5-base-pt-msmarco-100k-v2": "unicamp-dl/ptt5-base-pt-msmarco-100k-v2",
        "ptt5-base-en-pt-msmarco-100k-v2": "unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2",
        "mt5-base-en-pt-msmarco-v2": "unicamp-dl/mt5-base-en-pt-msmarco-v2",
        "mt5-base-mmarco-v2": "unicamp-dl/mt5-base-mmarco-v2",
        "mt5-base-en-pt-msmarco-v1": "unicamp-dl/mt5-base-en-pt-msmarco-v1",
        "mt5-base-mmarco-v1": "unicamp-dl/mt5-base-mmarco-v1",
        "ptt5-base-pt-msmarco-10k-v1": "unicamp-dl/ptt5-base-pt-msmarco-10k-v1",
        "ptt5-base-pt-msmarco-100k-v1": "unicamp-dl/ptt5-base-pt-msmarco-100k-v1",
        "ptt5-base-en-pt-msmarco-10k-v1": "unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1",
        "mt5-3B-mmarco-en-pt": "unicamp-dl/mt5-3B-mmarco-en-pt",
        "mt5-13b-mmarco-100k": "unicamp-dl/mt5-13b-mmarco-100k",
        "monoptt5-small": "unicamp-dl/monoptt5-small",
        "monoptt5-base": "unicamp-dl/monoptt5-base",
        "monoptt5-large": "unicamp-dl/monoptt5-large",
        "monoptt5-3b": "unicamp-dl/monoptt5-3b",
        
    },
    'rankt5':{
        'rankt5-base': 'Soyoung97/RankT5-base',
        'rankt5-large': 'Soyoung97/RankT5-large',
        'rankt5-3b': 'Soyoung97/RankT5-3b',
        
    },
    'listt5':{
        'listt5-base':'Soyoung97/ListT5-base',
        'listt5-3b': 'Soyoung97/ListT5-3b',
       
    },
    'inranker':{
        'inranker-small': 'unicamp-dl/InRanker-small',
        'inranker-base' :'unicamp-dl/InRanker-base',
        'inranker-3b':'unicamp-dl/InRanker-3B',
        
    },
    'apiranker':{
        "cohere":"cohere",
        "jina":"jina",
        "voyage":"voyage",
        "mixedbread.ai":"mixedbread.ai",
    },
    'transformer_ranker': {
        "mxbai-rerank-xsmall":"mixedbread-ai/mxbai-rerank-xsmall-v1",
        "mxbai-rerank-base": "mixedbread-ai/mxbai-rerank-base-v1",
        "mxbai-rerank-large": "mixedbread-ai/mxbai-rerank-large-v1",
        "bge-reranker-base":"BAAI/bge-reranker-base",
        "bge-reranker-large":"BAAI/bge-reranker-large",
        "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
        "bce-reranker-base" :"maidalun1020/bce-reranker-base_v1",
        "jina-reranker-tiny":'jinaai/jina-reranker-v1-tiny-en',
        "jina-reranker-turbo":"jinaai/jina-reranker-v1-turbo-en",
        "jina-reranker-base-multilingual":"jinaai/jina-reranker-v2-base-multilingual",
        "gte-multilingual-reranker-base":"Alibaba-NLP/gte-multilingual-reranker-base",
        "camembert-base-mmarcoFR":"antoinelouis/crossencoder-camembert-base-mmarcoFR",
        "camembert-large-mmarcoFR":"antoinelouis/crossencoder-camembert-large-mmarcoFR",
        "camemberta-base-mmarcoFR":"antoinelouis/crossencoder-camemberta-base-mmarcoFR",
        "distilcamembert-mmarcoFR":"antoinelouis/crossencoder-distilcamembert-mmarcoFR",
        "cross-encoder-mmarco-mMiniLMv2-L12-H384-v1":"corrius/cross-encoder-mmarco-mMiniLMv2-L12-H384-v1",
        "nli-deberta-v3-large":"cross-encoder/nli-deberta-v3-large",
        "ms-marco-MiniLM-L-12-v2":"cross-encoder/ms-marco-MiniLM-L-12-v2",
        "ms-marco-MiniLM-L-6-v2":"cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L-4-v2":"cross-encoder/ms-marco-MiniLM-L-4-v2",
        "ms-marco-MiniLM-L-2-v2":"cross-encoder/ms-marco-MiniLM-L-2-v2",
        "ms-marco-TinyBERT-L-2-v2":"cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "ms-marco-electra-base":"cross-encoder/ms-marco-electra-base",
        "ms-marco-TinyBERT-L-6":"cross-encoder/ms-marco-TinyBERT-L-6",
        "ms-marco-TinyBERT-L-4":"cross-encoder/ms-marco-TinyBERT-L-4",
        "ms-marco-TinyBERT-L-2":"cross-encoder/ms-marco-TinyBERT-L-2",
        "msmarco-MiniLM-L12-en-de-v1":"cross-encoder/msmarco-MiniLM-L12-en-de-v1",
        "msmarco-MiniLM-L6-en-de-v1":"cross-encoder/msmarco-MiniLM-L6-en-de-v1",
        
    },
    'llm_layerwise_ranker':{
        "bge-multilingual-gemma2":"BAAI/bge-multilingual-gemma2",
        "bge-reranker-v2-gemma":"BAAI/bge-reranker-v2-gemma",
        "bge-reranker-v2-minicpm-layerwise":"BAAI/bge-reranker-v2-minicpm-layerwise",
        "bge-reranker-v2.5-gemma2-lightweight":"BAAI/bge-reranker-v2.5-gemma2-lightweight",
        
    },
    'first_ranker':{
        "First-Model":"rryisthebest/First_Model",
        "Llama-3-8B":"meta-llama/Meta-Llama-3-8B-Instruct",
        
    },
    'lit5dist':{
        "LiT5-Distill-base": "castorini/LiT5-Distill-base",
        "LiT5-Distill-large":	"castorini/LiT5-Distill-large",
        "LiT5-Distill-xl":	"castorini/LiT5-Distill-xl",
        "LiT5-Distill-base-v2":	"castorini/LiT5-Distill-base-v2",
        "LiT5-Distill-large-v2":	"castorini/LiT5-Distill-large-v2",
        "LiT5-Distill-xl-v2":	"castorini/LiT5-Distill-xl-v2",
        
    },
    'lit5score':{
        "LiT5-Score-base":	"castorini/LiT5-Score-base",
        "LiT5-Score-large":	"castorini/LiT5-Score-large",
        "LiT5-Score-xl":	"castorini/LiT5-Score-xl",    
    },
    'vicuna_reranker':{
        "rank_vicuna_7b_v1": "castorini/rank_vicuna_7b_v1",
        "rank_vicuna_7b_v1_noda":	"castorini/rank_vicuna_7b_v1_noda",
        "rank_vicuna_7b_v1_fp16":	"castorini/rank_vicuna_7b_v1_fp16",
        "rank_vicuna_7b_v1_noda_fp16":	"castorini/rank_vicuna_7b_v1_noda_fp16",
        
    },
    'zephyr_reranker':{
        "rank_zephyr_7b_v1_full":"castorini/rank_zephyr_7b_v1_full",
        
    },
    'blender_reranker':{
        "PairRM":"llm-blender/PairRM",
        
    },
    'splade_reranker':{
        "splade-cocondenser":"naver/splade-cocondenser-ensembledistil",
       
    },
    'sentence_transformer_reranker': {
        "all-MiniLM-L6-v2":"all-MiniLM-L6-v2",
        "gtr-t5-base":"sentence-transformers/gtr-t5-base",
        "gtr-t5-large":"sentence-transformers/gtr-t5-large",
        "gtr-t5-xl":"sentence-transformers/gtr-t5-xl",
        "gtr-t5-xxl":"sentence-transformers/gtr-t5-xxl",
        "sentence-t5-base":"sentence-transformers/sentence-t5-base",
        "sentence-t5-xl":"sentence-transformers/sentence-t5-xl",
        "sentence-t5-xxl":"sentence-transformers/sentence-t5-xxl",
        "sentence-t5-large":"sentence-transformers/sentence-t5-large",
        "distilbert-multilingual-nli-stsb-quora-ranking":"sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking",
        "msmarco-bert-co-condensor":"sentence-transformers/msmarco-bert-co-condensor",
        "msmarco-roberta-base-v2":"sentence-transformers/msmarco-roberta-base-v2",
        
    },
    'colbert_ranker':{
        "colbertv2.0": "colbert-ir/colbertv2.0",
        "FranchColBERT": "bclavie/FraColBERTv2",
        "JapanColBERT": "bclavie/JaColBERTv2",
        "SpanishColBERT": "AdrienB134/ColBERTv2.0-spanish-mmarcoES",
        'jina-colbert-v1-en': 'jinaai/jina-colbert-v1-en',
        'ArabicColBERT-250k':'akhooli/arabic-colbertv2-250k-norm',
        'ArabicColBERT-711k':'akhooli/arabic-colbertv2-711k-norm',
        'BengaliColBERT':'turjo4nis/colbertv2.0-bn',
        'mxbai-colbert-large-v1':'mixedbread-ai/mxbai-colbert-large-v1',
        
    },
    'monobert':{
        "monobert-large": "castorini/monobert-large-msmarco"
    },
    'llm2vec': {
        "Meta-Llama-31-8B": "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
        "Meta-Llama-3-8B": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        "Mistral-7B": "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        "Llama-2-7B": "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
        "Sheared-LLaMA": "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"
    },
    'twolar':{
        'twolar-xl':"Dundalia/TWOLAR-xl",
        'twolar-large':"Dundalia/TWOLAR-large"
    },
    'echorank':{
        'flan-t5-large' : 'google/flan-t5-large',
        'flan-t5-xl' : 'google/flan-t5-xl'
    },
    'incontext_reranker':{
        'llamav3.1-8b':'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'llamav3.1-70b':'meta-llama/Llama-3.1-70B-Instruct',
        'Mistral-7B-Instruct-v0.2':'mistralai/Mistral-7B-Instruct-v0.2',
        
    }
}


URL ={
    'default': {'url': "https://api.openai.com/v1" , 'model_name': 'gpt-3.5-turbo-0125', 'class': OpenaiClient},
    'gpt-3.5': {'url': "https://api.openai.com/v1" , 'model_name': 'gpt-3.5-turbo-0125', 'class': OpenaiClient},
    'gpt-4':  {'url':'https://api.openai.com/v1', 'model_name': 'gpt-4o', 'class': OpenaiClient},
    'gpt-4-mini': {'url':'https://api.openai.com/v1', 'model_name':'gpt-4o-mini', 'class': OpenaiClient},
    'llamav3.1-8b' : {'url':'https://api.together.xyz/v1', 'model_name':'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', 'class':OpenaiClient },
    'llamav3.1-70b' :{'url': 'https://api.together.xyz/v1', 'model_name':'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', 'class':OpenaiClient },
    'claude-3-5' :{'url': 'https://api.anthropic.com', 'model_name':'claude-3-5-sonnet-20241022', 'class': ClaudeClient},
    "cohere": {'url': "https://api.cohere.ai/v2/rerank" , 'model_name':"rerank-english-v3.0"},
    "jina": {'url': "https://api.jina.ai/v1/rerank" , 'model_name':"jina-reranker-v1-base-en"},
    "voyage": {'url': "https://api.voyageai.com/v1/rerank" , 'model_name':"rerank-lite-1"},
    "mixedbread.ai": {'url': "https://api.mixedbread.ai/v1/reranking" , 'model_name':"mixedbread-ai/mxbai-rerank-large-v1"},
}



API_DOCUMENT_KEY_MAPPING = {
    "mixedbread.ai": "input",
    "text-embeddings-inference": "texts"
}
API_RETURN_DOCUMENTS_KEY_MAPPING = {
    "mixedbread.ai": "return_input",
    "text-embeddings-inference": "return_text"
}
API_RESULTS_KEY_MAPPING = {
    "voyage": "data",
    "mixedbread.ai": "data",
    "text-embeddings-inference": None
}
API_SCORE_KEY_MAPPING = {
    "mixedbread.ai": "score",
    "text-embeddings-inference": "score"
}


PREDICTION_TOKENS = {
    "default": ["▁false", "▁true"],
    "castorini/monot5-base-msmarco": ["▁false", "▁true"],
    "castorini/monot5-base-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-base-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-msmarco-10k": ["▁false", "▁true"],
    "unicamp-dl/InRanker-small": ["▁false", "▁true"],
    "unicamp-dl/InRanker-base": ["▁false", "▁true"],
    "unicamp-dl/InRanker-3B": ["▁false", "▁true"],
    "unicamp-dl/mt5-base-en-msmarco": ["▁no", "▁yes"],
    "unicamp-dl/ptt5-base-pt-msmarco-10k-v2": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-pt-msmarco-100k-v2": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2": ["▁não", "▁sim"],
    "unicamp-dl/mt5-base-en-pt-msmarco-v2": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v2": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-en-pt-msmarco-v1": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v1": ["▁no", "▁yes"],
    "unicamp-dl/ptt5-base-pt-msmarco-10k-v1": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-pt-msmarco-100k-v1": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1": ["▁não", "▁sim"],
    "unicamp-dl/mt5-3B-mmarco-en-pt": ["▁", "▁true"],
    "unicamp-dl/mt5-13b-mmarco-100k": ["▁", "▁true"],
    "unicamp-dl/monoptt5-small": ["▁Não", "▁Sim"],
    "unicamp-dl/monoptt5-base": ["▁Não", "▁Sim"],
    "unicamp-dl/monoptt5-large": ["▁Não", "▁Sim"],
    "unicamp-dl/monoptt5-3b": ["▁Não", "▁Sim"],
}

INDEX_TYPE = {
    'bm25':{
        'wiki': {
            'url' : 'https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/bm25.zip?download=true'
        },
        'msmarco':{
            'url': "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/bm25_index_msmarco.zip?download=true"
        }
    },
    'contriever':{
         'wiki': {
            'url': 'https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar', #https://dl.fbaipublicfiles.com/contriever/embeddings/contriever/wikipedia_embeddings.tar 
            'passages_url': 'https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/psgs_w100/psgs_w100.tsv?download=true',
        },
        'msmarco': {
            'url': 'https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_embeddings_contriever.zip?download=true',
            'passages_url': 'https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/msmarco-passage-corpus/msmarco-passage-corpus.tsv?download=true',
        },
    },
    'bge': {
        'wiki': {
            'urls': [
                "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/bgb_index.tar.gz.part1?download=true",
                "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/bgb_index.tar.gz.part2?download=true",
            ],
            'passages_url': "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/psgs_w100/psgs_w100.tsv?download=true",
        },
        'msmarco': {
            'urls': "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_embeddings_bgb.zip?download=true",
            'passages_url': "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/msmarco-passage-corpus/msmarco-passage-corpus.tsv?download=true",
        },
    },
    'colbert': {
        'wiki': {
            'urls': [
                "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/wikipedia_embeddings_colbert/wiki.zip.001?download=true",
                "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/wikipedia_embeddings_colbert/wiki.zip.002?download=true",
                "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/wikipedia_embeddings_colbert/wiki.zip.003?download=true",
                "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/wikipedia_embeddings_colbert/wiki.zip.004?download=true",
                "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/wikipedia_embeddings_colbert/wiki.zip.005?download=true"
            ],
            'passages_url': 'https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/psgs_w100/psgs_w100.tsv?download=true'
        },
        'msmarco': {
            'urls': "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/index/msmarco_embeddings_colbert.zip?download=true",
            'passages_url': "https://huggingface.co/datasets/abdoelsayed/reranking-datasets/resolve/main/msmarco-passage-corpus/msmarco-passage-corpus.tsv?download=true",
        },
    },
}