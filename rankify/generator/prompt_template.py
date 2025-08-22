from typing import List, Optional
from enum import Enum

class PromptTemplate(Enum):
    """
    **PromptTemplate Enum** for RAG Prompt Strategies.

    Enumerates the supported prompt templates for different RAG methods in Rankify.
    Each value corresponds to a specific prompt formatting strategy, used by the PromptGenerator to construct prompts for the model.

    Members:
        BASIC_RAG: Standard prompt for naive RAG (context + question + answer).
        CHAIN_OF_THOUGHT_RAG: Prompt for chain-of-thought reasoning (step-by-step thinking).
        ZERO_SHOT: Prompt for zero-shot generation (question only, no context).
        SELF_CONSISTENCY_RAG: Prompt for self-consistency reasoning (multiple step-by-step answers).
        REACT_RAG: Prompt for ReAct reasoning and action (interleaved reasoning and retrieval actions).

    Notes:
        - Used by PromptGenerator to select and format prompts for each RAG method.
        - Ensures consistent prompt structure across different techniques and models.
        - The corresponding template strings are defined in DEFAULT_PROMPTS.
    """

    BASIC_RAG = "basic-rag"
    CHAIN_OF_THOUGHT_RAG = "chain-of-thought-rag"
    ZERO_SHOT = "zero-shot"
    SELF_CONSISTENCY_RAG = "self-consistency-rag"
    REACT_RAG = "react-rag"
    IN_CONTEXT_RALM="in-context-ralm"

DEFAULT_PROMPTS = {
    PromptTemplate.BASIC_RAG: (
        "You are a helpful assistant. Give a single, concise answer to the question using the provided contexts.\n"
        "If the provided contexts are not sufficient, you may also use your own knowledge"
        "Question: {question}\n"
        "Contexts:\n{contexts}\n"
        "Answer:"
    ),
    PromptTemplate.CHAIN_OF_THOUGHT_RAG: (
        "You are a helpful assistant. Think step by step and then answer concisely with only the answer as a fact.\n"
        "Use the provided contexts if possible; if they are insuffiecient, you may use your own knowledge."
        "Question: {question}\n"
        "Contexts:\n{contexts}\n"
        "Let's think step by step.\n"
        "Answer:"
    ),
    PromptTemplate.SELF_CONSISTENCY_RAG: (
        "You are a helpful assistant. Think step by step and then answer concisely with only the answer as a fact.\n"
        "Use the provided contexts if possible; if they are insuffiecient, you may use your own knowledge."
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
    PromptTemplate.REACT_RAG: (
        "You are an intelligent assistant that solves tasks by interleaving reasoning with actions.\n\n"
        "You must follow this format strictly:\n"
        "Thought: [your reasoning about the problem]\n"
        "Action: [the action you want to take, in the form Search[argument]]\n"
        "Observation: [the result returned by the environment]\n\n"
        "Repeat Thought → Action → Observation as many times as needed.\n"
        "If you are confident you know the answer from your own knowledge, you may proceed directly to the Final Answer without searching.\n"
        "When you have enough information, provide:\n"
        "Final Answer: [your final solution to the user’s query]\n\n"
        "Guidelines:\n"
        "- Be concise in each Thought, just explain what is necessary for your next step.\n"
        "- Only use the defined Actions available to you (Search[query]).\n"
        "- If no action is required or you are sure of the answer, proceed to the Final Answer.\n"
        "- Do not hallucinate Observations; they are always provided externally.\n"
        "- End with \"Final Answer:\" when the task is solved.\n\n"
        "- The final Answer should be as precise and concise as possible.\n"
        "Question: {question}\n"
        "Contexts:\n{contexts}\n"
        "Thought:"
    ),
    PromptTemplate.IN_CONTEXT_RALM: (
        "You are a helpful assistant. Give a single, concise answer to the question using the provided contexts.\n"
        "If the provided contexts are not sufficient, you may also use your own knowledge"
        "Question: {question}\n"
        "Contexts:\n{contexts}\n"
        "Answer:"
    )
}
