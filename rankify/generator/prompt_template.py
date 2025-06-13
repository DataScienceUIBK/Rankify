from typing import List, Optional
from enum import Enum

class PromptTemplate(Enum):
    BASIC_RAG = "basic-rag"
    CHAIN_OF_THOUGHT_RAG = "chain-of-thought-rag"
    FID = "fid"
    IN_CONTEXT_RALM = "in-context-ralm"
    ZERO_SHOT = "zero-shot"

DEFAULT_PROMPTS = {
    PromptTemplate.BASIC_RAG: (
        "You are a helpful assistant. Give a single, concise answer to the question.\n"
        "Give only the answer, not the question  or context.\n"
        "Question: {question}\n"
        "Contexts:\n{contexts}\n"
        "Answer:"
    ),
    PromptTemplate.CHAIN_OF_THOUGHT_RAG: (
        "You are a helpful assistant. Think step by step and then answer concisely.\n"
        "Question: {question}\n"
        "Contexts:\n{contexts}\n"
        "Let's think step by step.\nAnswer:"
    ),
    PromptTemplate.FID: (
        "You are a helpful assistant. Use the following passages to answer the question.\n"
        "Question: {question}\n"
        "Passages:\n{contexts}\n"
        "Answer:"
    ),
    PromptTemplate.IN_CONTEXT_RALM: (
        "{contexts}\n\nBased on these texts, answer these questions:\nQ: {question}\nA:"
    ),
    PromptTemplate.ZERO_SHOT: (
        "Answer the following question using only the provided context.\n"
        "Question: {question}\n"
        "Context:\n{contexts}\n"
        "Answer:"
    ),
}
