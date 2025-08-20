# API Reference

Below is an overview of the modules, classes, and functions available in **Rankify**.

## Dataset Module
- [Dataset](dataset.md)

## Metrics Module
- [Metrics](metrics.md)

## Retrievers
- [Retriever](retrievers/retriever.md)
- [BM25 Retriever](retrievers/bm25.md)
- [Dense Retriever](retrievers/dense.md)
- [BGE Retriever](retrievers/bge.md)
- [ColBERT Retriever](retrievers/colbert.md)
- [Contriever Retriever](retrievers/contriever.md)

## Rerankers
- [Base](rerankings/base.md)
- [Reranking](rerankings/reranking.md)
- [UPR](rerankings/upr.md)
- [FlashRank](rerankings/flashrank.md)
- [RankGPT](rerankings/rankgpt.md)
- [Blender Reranker](rerankings/blender.md)
- [ColBERT Reranker](rerankings/colbert_ranker.md)
- [EchoRank](rerankings/echo_rank.md)
- [First Reranker](rerankings/first_ranker.md)
- [Incontext Reranker](rerankings/incontext_reranker.md)
- [InRank Reranker](rerankings/inrank.md)
- [ListT5 Reranker](rerankings/listt5.md)
- [Lit5 Reranker](rerankings/lit5.md)
- [LLM Layerwise Reranker](rerankings/llm_layerwise_ranker.md)
- [LLM2vec Reranker](rerankings/llm2vec_reranker.md)
- [MonoBERT Reranker](rerankings/monobert.md)
- [MonoT5 Reranker](rerankings/monot5.md)
- [Rank Fid Reranker](rerankings/rank_fid.md)
- [RankT5 Reranker](rerankings/rankt5.md)
- [Sentence Transformer Reranker](rerankings/sentence_transformer_reranker.md)
- [SPLADE Reranker](rerankings/splade_reranker.md)
- [Transformer Reranker](rerankings/transformer_reranker.md)
- [TwoLAR Reranker](rerankings/twolar.md)
- [Vicuna Reranker](rerankings/vicuna_reranker.md)
- [Zephyr Reranker](rerankings/zephyr_reranker.md)

## Generator Module
- [Generator](generators/generator.md)
- [Prompt Generator](generators/prompt_generator.md)
- [Prompt Template](generators/prompt_template.md)

### LLM endpoints
- [Base RAG Model](generators/models/base_rag_model.md)
- [Model Factory](generators/models/model_factory.md)
- [Hugging Face Model](generators/models/huggingface_model.md)
- [LiteLLM Model](generators/models/litellm_model.md)
- [OpenAI Model](generators/models/openai_model.md)
- [vLLM Model](generators/models/vllm_model.md)
- [FiD Model](generators/models/fid_model.md)

### RAG methods
- [Basic RAG Method](generators/rag_methods/basic_rag_method.md)
- [Zero-Shot Generation](generators/rag_methods/zero_shot.md)
- [Basic RAG](generators/rag_methods/basic_rag.md)
- [Chain of Thought RAG](generators/rag_methods/chain_of_thought_rag.md)
- [Self Consistency RAG](generators/rag_methods/self_consistency_rag.md)
- [ReAct RAG](generators/rag_methods/react_rag.md)
- [FiD](generators/rag_methods/fid_rag_method.md)
- [In Context RALM](generators/rag_methods/in_context_ralm_rag.md)