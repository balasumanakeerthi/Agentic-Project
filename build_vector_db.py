# build_vector_db.py
import os, glob, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

DATA_DIR = "data"
MODEL_NAME = "all-MiniLM-L6-v2"   # small & effective
INDEX_FILE = "faiss_index.index"
DOCS_FILE = "faiss_docs.pkl"

def load_texts(data_dir):
    texts = []
    metadatas = []
    for path in glob.glob(os.path.join(data_dir, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            t = f.read().strip()
        if not t:
            continue
        texts.append(t)
        metadatas.append({"source": os.path.basename(path)})
    return texts, metadatas

def embed_texts(model, texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=True)
        embeddings.append(emb)
    return np.vstack(embeddings).astype("float32")

def normalize(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms==0] = 1
    return v / norms

def main():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"Directory not found: '{DATA_DIR}'. Create it and add .txt files.")
    texts, metadatas = load_texts(DATA_DIR)
    if not texts:
        raise SystemExit("No .txt files found in data/. Add files and rerun.")
    print(f"[info] Loaded {len(texts)} documents from {DATA_DIR}")

    print("[info] loading SentenceTransformer model...")
    model = SentenceTransformer(MODEL_NAME)

    print("[info] embedding documents...")
    vecs = embed_texts(model, texts)
    vecs = normalize(vecs)

    dim = vecs.shape[1]
    print(f"[info] vectors shape: {vecs.shape}, dim={dim}")

    # Use inner product on normalized vectors => cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, INDEX_FILE)

    with open(DOCS_FILE, "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)

    print(f"✅ Saved FAISS index to '{INDEX_FILE}' and docs to '{DOCS_FILE}'")

if __name__ == "__main__":
    main()
