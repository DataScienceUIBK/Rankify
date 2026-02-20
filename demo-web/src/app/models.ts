// ─── Retrievers ────────────────────────────────────────────────────────────────
export const RETRIEVERS = [
    { value: "bm25", label: "BM25 (Sparse)", description: "Lexical matching, fast" },
    { value: "dpr", label: "DPR (Bi-Encoder)", description: "Dense Passage Retrieval" },
    { value: "ance", label: "ANCE", description: "Approximate Nearest Neighbor" },
    { value: "contriever", label: "Contriever", description: "Unsupervised dense retriever" },
    { value: "colbert", label: "ColBERTv2", description: "Late interaction model" },
    { value: "bge", label: "BGE-v1.5 (Default)", description: "State-of-the-art bi-encoder" },
] as const;

export type RetrieverValue = typeof RETRIEVERS[number]["value"];

// ─── Reranker Categories + Models ─────────────────────────────────────────────
export interface RerankerModel { value: string; label: string; }
export interface RerankerCategory {
    value: string;
    label: string;
    description: string;
    models: RerankerModel[];
}

export const RERANKER_CATEGORIES: RerankerCategory[] = [
    {
        value: "none",
        label: "None",
        description: "Skip reranking step",
        models: [],
    },
    {
        value: "flashrank",
        label: "FlashRank",
        description: "Ultra-fast ONNX reranker",
        models: [
            { value: "ms-marco-TinyBERT-L-2-v2", label: "TinyBERT-L-2-v2" },
            { value: "ms-marco-MiniLM-L-12-v2", label: "MiniLM-L-12-v2" },
            { value: "ms-marco-MultiBERT-L-12", label: "MultiBERT-L-12" },
            { value: "rank-T5-flan", label: "rank-T5-flan" },
            { value: "ce-esci-MiniLM-L12-v2", label: "ce-esci-MiniLM" },
            { value: "rank_zephyr_7b_v1_full", label: "Rank Zephyr 7B" },
        ],
    },
    {
        value: "transformer_ranker",
        label: "Cross-Encoder",
        description: "HuggingFace cross-encoders (BGE, mxbai, Jina, etc.)",
        models: [
            { value: "bge-reranker-base", label: "BGE Reranker Base" },
            { value: "bge-reranker-large", label: "BGE Reranker Large" },
            { value: "bge-reranker-v2-m3", label: "BGE Reranker v2-m3" },
            { value: "mxbai-rerank-xsmall", label: "mxbai Rerank xSmall" },
            { value: "mxbai-rerank-base", label: "mxbai Rerank Base" },
            { value: "mxbai-rerank-large", label: "mxbai Rerank Large" },
            { value: "jina-reranker-tiny", label: "Jina Reranker Tiny" },
            { value: "jina-reranker-turbo", label: "Jina Reranker Turbo" },
            { value: "jina-reranker-base-multilingual", label: "Jina Reranker Multilingual" },
            { value: "ms-marco-MiniLM-L-6-v2", label: "MS-MARCO MiniLM-L-6" },
            { value: "ms-marco-MiniLM-L-12-v2", label: "MS-MARCO MiniLM-L-12" },
            { value: "ms-marco-electra-base", label: "MS-MARCO ELECTRA Base" },
        ],
    },
    {
        value: "monot5",
        label: "MonoT5",
        description: "T5-based pointwise reranker",
        models: [
            { value: "monot5-base-msmarco", label: "MonoT5 Base" },
            { value: "monot5-large-msmarco", label: "MonoT5 Large" },
            { value: "monot5-base-msmarco-10k", label: "MonoT5 Base 10k" },
            { value: "monot5-large-msmarco-10k", label: "MonoT5 Large 10k" },
            { value: "monot5-3b-msmarco-10k", label: "MonoT5 3B" },
            { value: "monot5-base-med-msmarco", label: "MonoT5 Base Med" },
            { value: "monot5-3b-med-msmarco", label: "MonoT5 3B Med" },
        ],
    },
    {
        value: "rankgpt",
        label: "RankGPT (Local LLM)",
        description: "LLM-based listwise reranker (vLLM)",
        models: [
            { value: "Llama-3.2-1B", label: "Llama 3.2 1B Instruct" },
            { value: "Llama-3.2-3B", label: "Llama 3.2 3B Instruct" },
            { value: "llamav3.1-8b", label: "Llama 3.1 8B Instruct" },
            { value: "llamav3.1-70b", label: "Llama 3.1 70B Instruct" },
            { value: "Qwen2.5-7B", label: "Qwen 2.5 7B" },
            { value: "Mistral-7B-Instruct-v0.3", label: "Mistral 7B v0.3" },
        ],
    },
    {
        value: "rankgpt-api",
        label: "RankGPT (API)",
        description: "Cloud LLM reranker (OpenAI, Anthropic)",
        models: [
            { value: "gpt-3.5", label: "GPT-3.5 Turbo" },
            { value: "gpt-4", label: "GPT-4o" },
            { value: "gpt-4-mini", label: "GPT-4o mini" },
            { value: "claude-3-5", label: "Claude 3.5 Sonnet" },
            { value: "llamav3.1-8b", label: "Llama 3.1 8B (Together)" },
            { value: "llamav3.1-70b", label: "Llama 3.1 70B (Together)" },
        ],
    },
    {
        value: "colbert_ranker",
        label: "ColBERT Reranker",
        description: "Late-interaction reranker",
        models: [
            { value: "colbertv2.0", label: "ColBERTv2.0" },
            { value: "jina-colbert-v1-en", label: "Jina ColBERT v1" },
            { value: "mxbai-colbert-large-v1", label: "mxbai ColBERT Large" },
        ],
    },
    {
        value: "monobert",
        label: "MonoBERT",
        description: "BERT-based pointwise reranker",
        models: [
            { value: "monobert-large", label: "MonoBERT Large" },
        ],
    },
    {
        value: "upr",
        label: "UPR",
        description: "Unsupervised Passage Reranker (generative)",
        models: [
            { value: "t5-small", label: "T5 Small" },
            { value: "t5-base", label: "T5 Base" },
            { value: "t5-large", label: "T5 Large" },
            { value: "gpt2", label: "GPT-2" },
            { value: "gpt2-xl", label: "GPT-2 XL" },
            { value: "flan-t5-xl", label: "Flan-T5 XL" },
        ],
    },
    {
        value: "inranker",
        label: "InRanker",
        description: "T5-based reranker from Unicamp-DL",
        models: [
            { value: "inranker-small", label: "InRanker Small" },
            { value: "inranker-base", label: "InRanker Base" },
            { value: "inranker-3b", label: "InRanker 3B" },
        ],
    },
    {
        value: "rankt5",
        label: "RankT5",
        description: "T5-based learning-to-rank",
        models: [
            { value: "rankt5-base", label: "RankT5 Base" },
            { value: "rankt5-large", label: "RankT5 Large" },
            { value: "rankt5-3b", label: "RankT5 3B" },
        ],
    },
    {
        value: "llm_layerwise_ranker",
        label: "LLM Layerwise (BGE)",
        description: "Layerwise LLM reranker (Gemma2, MiniCPM)",
        models: [
            { value: "bge-reranker-v2-gemma", label: "BGE Reranker v2 Gemma" },
            { value: "bge-reranker-v2-minicpm-layerwise", label: "BGE MiniCPM Layerwise" },
            { value: "bge-reranker-v2.5-gemma2-lightweight", label: "BGE Gemma2 Lightweight" },
            { value: "bge-multilingual-gemma2", label: "BGE Multilingual Gemma2" },
        ],
    },
    {
        value: "sentence_transformer_reranker",
        label: "Sentence Transformer Reranker",
        description: "Bi-encoder reranker (GTR-T5, ST models)",
        models: [
            { value: "gtr-t5-base", label: "GTR-T5 Base" },
            { value: "gtr-t5-large", label: "GTR-T5 Large" },
            { value: "gtr-t5-xl", label: "GTR-T5 XL" },
        ],
    },
    {
        value: "apiranker",
        label: "API Reranker",
        description: "External API rerankers (Cohere, Jina, Voyage)",
        models: [
            { value: "cohere", label: "Cohere Rerank" },
            { value: "jina", label: "Jina Rerank" },
            { value: "voyage", label: "Voyage Rerank" },
            { value: "mixedbread.ai", label: "Mixedbread.ai Rerank" },
        ],
    },
];

// ─── Generators ────────────────────────────────────────────────────────────────
export const GENERATORS = [
    { value: "openai", label: "GPT-4o (OpenAI)", description: "OpenAI cloud API" },
    { value: "claude", label: "Claude 3.5 Sonnet", description: "Anthropic cloud API" },
    { value: "llama-3", label: "Llama 3.1 8B", description: "Local vLLM endpoint" },
    { value: "qwen", label: "Qwen 2.5 7B", description: "Local vLLM endpoint" },
    { value: "mistral", label: "Mistral 7B", description: "Local vLLM endpoint" },
    { value: "litellm", label: "LiteLLM (any)", description: "Universal LLM proxy" },
] as const;

export type GeneratorValue = typeof GENERATORS[number]["value"];
