from rankify.generator.rag_methods.basic_rag import BasicRAG
from rankify.generator.rag_methods.chain_of_thought_rag import ChainOfThoughtRAG
from rankify.generator.rag_methods.fid_rag_method import FiDRAGMethod
from rankify.generator.rag_methods.in_context_ralm_rag import InContextRALMRAG
from rankify.generator.rag_methods.self_consistency_rag import SelfConsistencyRAG
from rankify.generator.rag_methods.zero_shot import ZeroShotRAG
from rankify.generator.rag_methods.react_rag import ReActRAG 



RAG_METHODS = {
    "in-context-ralm": InContextRALMRAG,
    "fid": FiDRAGMethod,
    "zero-shot": ZeroShotRAG,
    "basic-rag": BasicRAG,
    "chain-of-thought-rag": ChainOfThoughtRAG,
    "self-consistency-rag": SelfConsistencyRAG,
    "react-rag": ReActRAG
}