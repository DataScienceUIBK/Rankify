from typing import List, Optional
from enum import Enum

class PromptTemplate(Enum):
    BASIC_RAG = "basic-rag"
    CHAIN_OF_THOUGHT_RAG = "chain-of-thought-rag"
    ZERO_SHOT = "zero-shot"
    SELF_CONSISTENCY_RAG = "self-consistency-rag"
    REACT_RAG = "react-rag"

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
        "Question: {question}\n"
        "Contexts:\n{contexts}\n"
        "Thought:"
    )
}
