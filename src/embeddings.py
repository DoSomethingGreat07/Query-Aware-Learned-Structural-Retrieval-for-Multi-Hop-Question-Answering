import numpy as np
from sentence_transformers import SentenceTransformer
from chunking import chunk_documents


def generate_chunk_embeddings(
    split="train",
    max_samples=1000,
    chunk_size=300,
    chunk_overlap=50,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Generate dense embeddings for chunked documents.

    Returns:
        chunks: List[Document]
        embeddings: np.ndarray of shape (num_chunks, embedding_dim)
        model: SentenceTransformer
    """
    chunks = chunk_documents(
        split=split,
        max_samples=max_samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    model = SentenceTransformer(model_name)

    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return chunks, embeddings, model


if __name__ == "__main__":
    chunks, embeddings, _ = generate_chunk_embeddings(
        split="train",
        max_samples=1000,
        chunk_size=300,
        chunk_overlap=50,
    )

    print("Total chunks:", len(chunks))
    print("Embedding matrix shape:", embeddings.shape)

    print("\nSample metadata:")
    print(chunks[0].metadata)

    print("\nSample chunk text:")
    print(chunks[0].page_content[:300])

    print("\nFirst embedding vector shape:")
    print(embeddings[0].shape)

    print("\nEmbedding dtype:")
    print(embeddings.dtype)