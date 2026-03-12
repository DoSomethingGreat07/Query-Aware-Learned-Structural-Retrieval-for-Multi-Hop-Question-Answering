import numpy as np
import networkx as nx
from embeddings import generate_chunk_embeddings


def build_chunk_graph(
    split="train",
    max_samples=1000,
    chunk_size=300,
    chunk_overlap=50,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    top_k_neighbors=5,
    min_similarity=0.40,
):
    """
    Build a chunk graph where:
    - each node = one chunk
    - edges connect semantically similar chunks

    Returns:
        G: networkx.Graph
        chunks: list of LangChain Document chunks
        embeddings: np.ndarray
    """
    chunks, embeddings, _ = generate_chunk_embeddings(
        split=split,
        max_samples=max_samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model_name,
    )

    G = nx.Graph()

    # Add nodes
    for i, chunk in enumerate(chunks):
        G.add_node(
            i,
            title=chunk.metadata.get("title", ""),
            source=chunk.metadata.get("source", ""),
            text=chunk.page_content,
        )

    # Cosine similarity because embeddings are already normalized
    sim_matrix = embeddings @ embeddings.T

    num_chunks = len(chunks)

    for i in range(num_chunks):
        # Exclude self
        sim_scores = sim_matrix[i].copy()
        sim_scores[i] = -1.0

        # Top-k nearest neighbors
        neighbor_indices = np.argsort(sim_scores)[-top_k_neighbors:][::-1]

        for j in neighbor_indices:
            score = float(sim_scores[j])

            if score >= min_similarity:
                G.add_edge(i, j, weight=score)

    return G, chunks, embeddings


if __name__ == "__main__":
    G, chunks, embeddings = build_chunk_graph(
        split="train",
        max_samples=200,
        chunk_size=300,
        chunk_overlap=50,
        top_k_neighbors=5,
        min_similarity=0.40,
    )

    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    sample_node = 0
    print("\nSample node:")
    print("Node ID:", sample_node)
    print("Title:", G.nodes[sample_node]["title"])
    print("Text:", G.nodes[sample_node]["text"][:300])

    print("\nNeighbors of sample node:")
    for neighbor in list(G.neighbors(sample_node))[:5]:
        print(
            f"Neighbor ID: {neighbor}, "
            f"Title: {G.nodes[neighbor]['title']}, "
            f"Weight: {G[sample_node][neighbor]['weight']:.4f}"
        )