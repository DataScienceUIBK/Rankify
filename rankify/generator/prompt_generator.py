from typing import List


class PromptGenerator:
    def __init__(self, model_type: str, method: str):
        self.model_type = model_type
        self.method = method

    def generate_system_prompt(self) -> str:
        """Generate a system-level prompt."""
        return f"System prompt for model type: {self.model_type}, method: {self.method}"

    def generate_user_prompt(self, question: str, contexts: List[str]) -> str:
        """Generate a user-level prompt for universal RAG."""
        context_str = "\n".join(contexts)
        # Use proper [INST] format for Llama-style models and concise instructions
        return (
            "[INST] <<SYS>>\n"
            "You are a helpful assistant. Answer the question as concisely as possible. "
            "Give only the answer, not the question or context. If the answer cannot be determined from the information, say \"I don't know\".\n"
            "<</SYS>>\n\n"
            f"Question: {question}\n"
            f"Contexts:\n{context_str}\n"
            "[/INST]"
        )