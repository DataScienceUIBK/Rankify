import logging
import pickle
from pathlib import Path

import numpy as np
import torch

from rankify.retrievers.utils import process_chunk, generate_chunks, count_file_lines, process_chunk_tabbed
from multiprocessing.dummy import Pool
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import json
logging.basicConfig(level=logging.INFO)

def to_tsv(input_path, output_dir, chunk_size, threads) -> str:
    """
    Convert a corpus file to TSV format.
    :param input_path: Path to the input corpus file.
    :param output_dir: Directory to save the output TSV file.
    :param chunk_size: Number of lines to process in each chunk.
    :param threads: Number of threads to use for processing.
    :return: Path to the output TSV file.
    """
    output_path = output_dir / "passages.tsv"

    total_lines = count_file_lines(input_path)
    logging.info("Streaming and processing corpus with " + str(threads) + " threads...")
    with open(input_path, "r", encoding="utf-8") as f_in, \
            open(output_path, "w", encoding="utf-8") as f_out, \
            Pool(processes=threads) as pool:

            pbar = tqdm(total=total_lines, desc="Processing")

            for result in pool.imap_unordered(lambda args: process_chunk_tabbed(*args), generate_chunks(f_in, chunk_size)):
                for i, doc_json in enumerate(result):
                        f_out.write(doc_json + "\n")
                pbar.update(len(result))

            pbar.close()

    logging.info(f"Done saving corpus to {output_path}")
    return output_path

def to_pyserini_jsonl(input_path, output_dir, chunk_size, threads) -> str:
    """
    Convert a corpus file to Pyserini JSONL format.
    :param input_path: Path to the input corpus file.
    :param output_dir: Directory to save the output JSONL file.
    :param chunk_size: Number of lines to process in each chunk.
    :param threads: Number of threads to use for processing.
    :return: Path to the output JSONL file.
    """
    output_path = output_dir / "corpus.jsonl"

    total_lines = count_file_lines(input_path)
    mapping_file = output_dir / "id_mapping.json"
    if mapping_file.exists():
        import json
        with open(mapping_file, "r", encoding="utf-8") as f:
            id_mapping = json.load(f)
        # Attach mapping to function for multiprocessing access
        process_chunk.id_mapping = id_mapping
        logging.info(f"Loaded ID mapping with {len(id_mapping)} entries")
    
    logging.info("Streaming and processing corpus with " + str(threads) + " threads...")
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out, \
         Pool(processes=threads) as pool:

        pbar = tqdm(total=total_lines, desc="Processing")

        for result in pool.imap_unordered(lambda args: process_chunk(*args), generate_chunks(f_in, chunk_size)):
            for doc_json in result:
                #print(doc_json)
                f_out.write(doc_json + "\n")
            pbar.update(len(result))

        pbar.close()

    logging.info(f"✅ Done saving corpus to {output_path}")
    return output_path

def to_pyserini_jsonl_dense(input_path, output_dir, chunk_size, threads) -> str:
    """
    Convert a dense corpus file to Pyserini JSONL format.
    :param input_path: Path to the input dense corpus file.
    :param output_dir: Directory to save the output JSONL file.
    :param chunk_size: Number of lines to process in each chunk.
    :param threads: Number of threads to use for processing.
    :return: Path to the output JSONL file.
    """
    output_path = output_dir / "corpus.jsonl"

    total_lines = count_file_lines(input_path)
    logging.info("Streaming and processing dense corpus with " + str(threads) + " threads...")

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out, \
         Pool(processes=threads) as pool:

        pbar = tqdm(total=total_lines, desc="Processing")

        for result in pool.imap_unordered(lambda args: process_chunk(*args, dense_index=True), generate_chunks(f_in, chunk_size)):
            for doc_json in result:
                f_out.write(doc_json + "\n")
            pbar.update(len(result))

        pbar.close()

    logging.info(f"Done saving dense corpus to {output_path}")
    return output_path

def to_contriever_embedding_chunks(jsonl_path, output_dir, chunk_size, model_name, batch_size, device):
    """
    Converts a corpus to Contriever embeddings in .pkl chunks.
    :param jsonl_path: Path to the input JSONL file containing documents.
    :param output_dir: Directory to save the output .pkl files.
    :param chunk_size: Number of documents to process in each chunk.
    :param model_name: Name of the Contriever model to use.
    :param batch_size: Number of documents to process in each batch for encoding.
    :param device: Device to run the model on (e.g., "cpu" or "cuda").
    Returns:
        List[Path]: Paths to all .pkl shards.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    shard_paths = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        chunk = []
        shard_id = 0

        for idx, line in enumerate(f):
            doc = json.loads(line.strip())
            text = doc.get("contents", doc.get("text", "")).strip()
            if not text:
                continue
            doc_id = doc.get("id", f"doc{idx}")
            chunk.append((doc_id, text))

            if len(chunk) >= chunk_size:
                ids, embeddings = encode_contriever_chunk(chunk, tokenizer, model, batch_size, device)
                path = output_dir / f"embeddings_{shard_id}.pkl"
                with open(path, "wb") as fout:
                    pickle.dump((ids, embeddings), fout)
                shard_paths.append(path)
                shard_id += 1
                chunk = []

        if chunk:
            ids, embeddings = encode_contriever_chunk(chunk, tokenizer, model, batch_size, device)
            path = output_dir / f"embeddings_{shard_id}.pkl"
            with open(path, "wb") as fout:
                pickle.dump((ids, embeddings), fout)
            shard_paths.append(path)

    return shard_paths


def encode_contriever_chunk(chunk, tokenizer, model, batch_size, device) -> tuple:
    """
    Encodes a chunk of text using Contriever.
    :param chunk: List of tuples (doc_id, text) to encode.
    :param tokenizer: Tokenizer for the Contriever model.
    :param model: Contriever model to use for encoding.
    :param batch_size: Number of documents to process in each batch.
    :param device: Device to run the model on (e.g., "cpu" or "cuda").
    :return: Tuple of lists (ids, embeddings) where ids are document IDs and embeddings are their corresponding embeddings.
    """
    ids = []
    all_embeddings = []

    for i in tqdm(range(0, len(chunk), batch_size), desc="Encoding chunk"):
        batch = chunk[i:i+batch_size]
        batch_ids = [doc_id for doc_id, _ in batch]
        texts = [text for _, text in batch]

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()

        all_embeddings.append(embeddings)
        ids.extend(batch_ids)

    embeddings = np.vstack(all_embeddings)
    return ids, embeddings
