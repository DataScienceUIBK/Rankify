from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.generator.generator import Generator
from rankify.utils.models.rank_llm.rerank.api_keys import get_litellm_api_key

# Define question and answer
question = Question("What is the capital of France?")
#answers = Answer(["Paris"])
answers = Answer([""])
contexts = [
    Context(id=1, title="France", text="The capital of France is Paris.", score=0.9),
    Context(id=2, title="Germany", text="Berlin is the capital of Germany.", score=0.5)
]

# Construct document
doc = Document(question=question, answers=answers, contexts=contexts)

#load api-key
api_key = get_litellm_api_key()

# Initialize Generator (e.g., Meta Llama)
generator = Generator(method="basic-rag", model_name='ollama/mistral', backend="litellm", api_key=api_key)

# Generate answer
generated_answers = generator.generate([doc])
print(generated_answers)  # Output: ["Paris"]
