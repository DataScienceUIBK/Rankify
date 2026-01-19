import streamlit as st
from pathlib import Path
import tempfile
import io
from contextlib import redirect_stdout, redirect_stderr

# =========================================================
# Import your indexers
# =========================================================
from rankify.indexing.lucene_indexer import LuceneIndexer
from rankify.indexing.dpr_indexer import DPRIndexer
from rankify.indexing.ance_indexer import ANCEIndexer
from rankify.indexing.contriever_indexer import ContrieverIndexer
from rankify.indexing.colbert_indexer import ColBERTIndexer
from rankify.indexing.bge_indexer import BGEIndexer


RETRIEVERS = {
    "bm25": LuceneIndexer,
    "dpr": DPRIndexer,
    "ance": ANCEIndexer,
    "contriever": ContrieverIndexer,
    "colbert": ColBERTIndexer,
    "bge": BGEIndexer,
}

# ============== UI Start ==============
st.title("📦 Rankify Indexing UI Demo")

# ==============  1) Corpus Selection (Upload OR Path) ==============
st.header("1) Select Corpus")

method = st.radio(
    "Choose corpus input method:",
    ["⬆️ Upload (small files)", "📁 Path to file (large files)"],
    key="method"
)

corpus_path = None

# --- OPTION A: UPLOAD SMALL FILES ---
if method == "⬆️ Upload (small files)":
    uploaded = st.file_uploader("Upload JSONL corpus", type=["jsonl"], key="uploader")

    if uploaded:
        # Save file to temporary path
        temp_path = Path(tempfile.gettempdir()) / uploaded.name
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())

        st.success(f"Uploaded and saved to: {temp_path}")
        corpus_path = temp_path


# --- OPTION B: PATH FOR LARGE FILES ---
else:
    corpus_input = st.text_input("Enter absolute path to corpus JSONL:", key="path_input")

    if corpus_input:
        path = Path(corpus_input)
        if not path.exists():
            st.error("❌ File does not exist.")
        else:
            corpus_path = path


# Stop if we don't yet have a valid file path
if corpus_path is None:
    st.stop()

# ==============  2) Indexing Configuration ==============
st.header("2) Indexing Configuration")

retriever = st.selectbox("Retriever", list(RETRIEVERS.keys()), key="retriever")

output_dir = st.text_input("Output Directory", "rankify_indices", key="output_dir")
index_type = st.text_input("Index Type", "wiki", key="index_type")
threads = st.number_input("Threads", min_value=1, value=32, key="threads")

encoder_name = None
batch_size = None
device = None
embedding_batch_size = None
chunk_size = None

dense = ["dpr", "ance", "contriever", "colbert", "bge"]

if retriever in dense:
    encoder_name = st.text_input("Encoder Model (optional)", key="encoder")
    batch_size = st.number_input("Batch Size", value=32, key="batch_size")
    device = st.selectbox("Device", ["cpu", "cuda"], index=1, key="device")
    chunk_size = st.number_input("Chunk Size", min_value=1, value=1024, key="chunk_size")

if retriever == "contriever":
    embedding_batch_size = st.number_input(
        "Embedding Batch Size", value=0, key="embedding_batch"
    )

# ==============  3) Run Indexing ==============
if st.button("🚀 Build Index", key="build"):
    st.write("### 🏗️ Building index...")

    kwargs = {
        "corpus_path": str(corpus_path),
        "retriever_name": retriever,
        "output_dir": output_dir,
        "index_type": index_type,
        "threads": threads,
        "chunk_size": chunk_size
    }

    # Dense-specific
    if encoder_name:
        kwargs["encoder_name"] = encoder_name
    if batch_size:
        kwargs["batch_size"] = batch_size
    if device:
        kwargs["device"] = device
    if retriever == "contriever" and embedding_batch_size is not None:
        kwargs["embedding_batch_size"] = embedding_batch_size

    # Run indexer
    IndexerClass = RETRIEVERS[retriever]
    indexer = IndexerClass(**kwargs)

    try:
        indexer.build_index()
        indexer.load_index()
        st.success(f"🎉 Index built successfully at: {indexer.index_dir}")
    except Exception as e:
        st.error(f"❌ Indexing failed:\n{e}")
