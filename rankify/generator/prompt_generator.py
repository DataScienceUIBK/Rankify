from typing import List


class PromptGenerator:
    def __init__(self, model_type: str, method: str):
        self.model_type = model_type
        self.method = method

    def generate_system_prompt(self) -> str:
        """Generate a system-level prompt."""
        return f"System prompt for model type: {self.model_type}, method: {self.method}"

    def generate_user_prompt(self, question: str, contexts: List[str]) -> str:
        """Generate a user-level prompt."""
        context_str = "\n".join(contexts)
        return f"Question: {question}\nContexts:\n{context_str}"