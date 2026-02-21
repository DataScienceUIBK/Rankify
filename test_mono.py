import os
os.environ["RERANKING_CACHE_DIR"] = "/home/aa17626/.cache/rankify"
from rankify.models.reranking import Reranking
from rankify.dataset.dataset import Document, Question, Answer, Context
import traceback

print("Initializing MonoBERT...")
try:
    reranker = Reranking(method="monobert", model_name="monobert-large")
    print("MoboBERT initialized.")
    doc = Document(question=Question("Test"), answers=Answer([]), contexts=[Context(text="Context 1", id="1"), Context(text="Context 2", id="2")])
    res = reranker.rank([doc])
    
    print("Reorder Contexts Length:", len(res[0].reorder_contexts) if res[0].reorder_contexts else 0)
    print("Contexts Length:", len(res[0].contexts))
except Exception as e:
    print("CRASH!")
    traceback.print_exc()
