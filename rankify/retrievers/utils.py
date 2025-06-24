#utils.py
import json
import logging
import torch

def count_file_lines(path) -> int:
    """
    Efficiently count the number of lines in a file.
    :param path: str: Path to the file.
    :return: int: Number of lines in the file.
    """
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count

def sanitize(text):
    """
    Sanitize text by replacing tabs and newlines with spaces and stripping leading/trailing whitespace.
    :param text: str: The text to sanitize.
    :return: str: Sanitized text.
    """
    return text.replace("\t", " ").replace("\n", " ").strip()

def process_chunk(chunk_lines, start_idx, dense_index=False):
    """
    Process a chunk of lines from the corpus file and convert them to the required format.

    Args:
        chunk_lines (list): Lines from the corpus file.
        start_idx (int): Fallback starting index for IDs.
        dense_index (bool): True for dense format with separate title field.

    Returns:
        list: List of JSON strings with the required fields.
    """
    results = []
    for i, line in enumerate(chunk_lines):
        try:
            doc = json.loads(line.strip())

            raw_id = doc.get("id")
            #Todo: replace doc can be removed when str is supported as type
            # doc_id = (
            #     int(raw_id.replace("doc", "")) if isinstance(raw_id, str) and raw_id.startswith("doc")
            #     else int(raw_id) if raw_id
            #     else start_idx + i
            # )
            # WITH THIS:
            # Use global mapping if available, otherwise fallback
            if hasattr(process_chunk, 'id_mapping') and str(raw_id) in process_chunk.id_mapping:
                doc_id = process_chunk.id_mapping[str(raw_id)]
            elif isinstance(raw_id, str) and raw_id.startswith("doc"):
                try:
                    doc_id = int(raw_id.replace("doc", ""))
                except ValueError:
                    doc_id = start_idx + i
            elif raw_id:
                try:
                    doc_id = int(raw_id)
                except (ValueError, TypeError):
                    doc_id = start_idx + i
            else:
                doc_id = start_idx + i

            contents = (doc.get("contents") or doc.get("text", "")).strip()
            title = doc.get("title", contents[:100] if contents else "").strip()

            if dense_index:
                # Keep title and contents separate
                doc_dict = {"id": doc_id, "contents": contents}
                if title:
                    doc_dict["title"] = title
                results.append(json.dumps(doc_dict, ensure_ascii=False))
            else:
                # For sparse, combine title and contents
                if title:
                    full_contents = f"{title}\n{contents}"
                else:
                    full_contents = contents
                results.append(json.dumps({"id": doc_id, "contents": full_contents}, ensure_ascii=False))

        except (json.JSONDecodeError, ValueError):
            continue
    return results

def process_chunk_tabbed(chunk_lines, start_idx) -> list:
    """
    Process a chunk of lines from the corpus file and convert them to tab-separated format.
    Args:
        chunk_lines (list): Lines from the corpus file.
        start_idx (int): Fallback starting index for IDs.
    Returns:
        list: List of strings formatted as "id\tcontents\ttitle".
    """
    results = []
    for i, line in enumerate(chunk_lines):
        try:
            doc = json.loads(line.replace("\t", " ").strip())
            doc_id = doc.get("id", "docid").replace("doc", "")
            doc_id = doc_id if doc_id else start_idx + i
            contents = (doc.get("contents") or doc.get("text", "")).strip()
            title = doc.get("title", contents[:100] if contents else "No Title").strip()

            results.append(f"{sanitize(str(doc_id))}\t{sanitize(contents)}\t{sanitize(title)}")
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Skipping line due to error: {e}")
    return results

def bge_embed_chunk(chunk_lines, tokenizer, model, device):
    """
    Process a chunk of lines: tokenize, encode, and return embeddings with doc IDs.

    Returns:
        List[dict]: List of dicts with 'id' and 'embedding'.
    """
    ids = []
    texts = []

    for line in chunk_lines:
        obj = json.loads(line)
        doc_id = obj.get("id")
        title = obj.get("title", "")
        text = obj.get("text", "")
        full_text = f"{title} {text}".strip()
        texts.append(full_text)
        ids.append(doc_id)

    with torch.no_grad():
        tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = model(**tokenized)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()

    return [{"id": doc_id, "embedding": emb} for doc_id, emb in zip(ids, embeddings)]


def generate_chunks(file_obj, chunk_size=512):
    """
    Generator to yield chunks of lines from a file.

    Args:
        file_obj (file object): The file object to read from.
        chunk_size (int): The number of lines per chunk.
    Yields:
        tuple: A tuple containing a list of lines and the starting index of the chunk.
    """
    buffer = []
    start_idx = 0
    for line in file_obj:
        buffer.append(line)
        if len(buffer) >= chunk_size:
            yield buffer[:], start_idx
            buffer = []
            start_idx += chunk_size
    if buffer:
        yield buffer, start_idx
