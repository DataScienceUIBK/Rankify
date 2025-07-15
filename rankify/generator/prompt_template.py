from typing import List, Optional
from enum import Enum

class PromptTemplate(Enum):
    BASIC_RAG = "basic-rag"
    CHAIN_OF_THOUGHT_RAG = "chain-of-thought-rag"
    ZERO_SHOT = "zero-shot"
    SELF_CONSISTENCY_RAG = "self-consistency-rag"

DEFAULT_PROMPTS = {
    PromptTemplate.BASIC_RAG: (
        "You are a helpful assistant. Give a single, concise answer to the question.\n"
        #"Since I want to use this for evaluation, only the single answer helps me a lot more and there is no point in explaining it.\n"
        "Question: {question}\n"
        "Contexts:\n{contexts}\n"
        "Answer:"
    ),
    PromptTemplate.CHAIN_OF_THOUGHT_RAG: (
        "You are a helpful assistant. Think step by step and then answer concisely with only the answer.\n"
        "Question: {question}\n"
        "Contexts:\n{contexts}\n"
        "Let's think step by step.\nAnswer:"
    ),
    PromptTemplate.SELF_CONSISTENCY_RAG: (
        "You are a helpful assistant. Think step by step and then answer concisely with only the answer as a fact.\n"
        "Question: {question}\n"
        "Contexts:\n{contexts}\n"
        "Let's think step by step.\n"
        "Answer:"
    ),
    PromptTemplate.ZERO_SHOT: (
        "Answer the following question concisely.\n"
        "Question: {question}\n"
        "Answer:"
    ),
}
