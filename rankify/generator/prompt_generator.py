from typing import List, Optional

from rankify.generator.prompt_template import DEFAULT_PROMPTS, PromptTemplate

class PromptGenerator:
    def __init__(self, method: str, model_type: str, prompt_template: Optional[PromptTemplate] = None):
        self.method = method
        if prompt_template is not None:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = self._select_template(method)

    def _select_template(self, method: str) -> PromptTemplate:
        method = method.lower()
        if method in PromptTemplate._value2member_map_:
            return PromptTemplate(method)
        return PromptTemplate.BASIC_RAG

    def generate_user_prompt(self, question: str, contexts: List[str], custom_prompt: Optional[str] = None) -> str:
        if contexts is None:
            contexts = []
        context_str = "\n".join(contexts)
        if custom_prompt:
            print(custom_prompt.format(question=question, contexts=context_str))
            return custom_prompt.format(question=question, contexts=context_str)
        template = DEFAULT_PROMPTS[self.prompt_template]
        print(template.format(question=question, contexts=context_str))
        return template.format(question=question, contexts=context_str)