import numpy as np
from sentence_transformers import SentenceTransformer
from embeddings import generate_chunk_embeddings


def retrieve_top_k(query, chunks, embeddings, model, k=5):
    """
    Retrieve top-k most similar chunks for a query using cosine similarity.

    Args:
        query: str
        chunks: List[Document]
        embeddings: np.ndarray
        model: SentenceTransformer
        k: int

    Returns:
        List[dict]
    """
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    scores = embeddings @ query_vec
    top_k_idx = np.argsort(scores)[-k:][::-1]

    results = []
    for idx in top_k_idx:
        results.append(
            {
                "chunk_id": int(idx),
                "score": float(scores[idx]),
                "metadata": chunks[idx].metadata,
                "text": chunks[idx].page_content,
            }
        )

    return results


if __name__ == "__main__":
    chunks, embeddings, model = generate_chunk_embeddings(
        split="train",
        max_samples=10000,
        chunk_size=500,
        chunk_overlap=100,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    query = "Which magazine was started first Arthur's Magazine or First for Women?"
    results = retrieve_top_k(query, chunks, embeddings, model, k=5)

    for rank, item in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print("Score:", round(item["score"], 4))
        print("Metadata:", item["metadata"])
        print("Text:", item["text"][:400])