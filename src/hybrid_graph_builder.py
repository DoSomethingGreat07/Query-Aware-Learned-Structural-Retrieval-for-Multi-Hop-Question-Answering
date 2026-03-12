import re
import numpy as np
import networkx as nx
from collections import defaultdict

from chunking import chunk_documents
from embeddings import generate_chunk_embeddings


def normalize_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def simple_keywords(text, min_len=4):
    proper_tokens = re.findall(r"[A-Z][A-Za-z0-9\-]{3,}", text)
    stopwords = {
        "This", "That", "With", "From", "Have", "Were", "Which", "Their", "There",
        "About", "After", "Before", "Into", "During", "Than", "Then", "Them",
        "They", "Been", "Being", "Also", "Some", "Such", "Most", "More", "Less",
        "Only", "Other", "Many", "First", "Second", "Third", "Would", "Could",
        "Should", "Where", "When", "What", "Who", "Whose", "While", "Because",
        "These", "Those", "Each", "Very", "Much", "Well", "Used", "Using",
        "American", "Published", "Magazine", "Magazines", "However", "Although",
        "Among", "Between", "Through", "Without", "Another", "Within",
    }
    proper_set = {t for t in proper_tokens if t not in stopwords and len(t) >= min_len}
    if len(proper_set) < 3:
        lower_stopwords = {s.lower() for s in stopwords} | {
            "this", "that", "with", "from", "have", "were", "which", "their",
            "there", "about", "after", "before", "into", "during", "than",
            "then", "them", "they", "been", "being", "also", "some", "such",
            "most", "more", "less", "only", "other", "many", "first", "second",
            "third", "would", "could", "should", "where", "when", "what", "who",
            "whose", "while", "because", "these", "those", "each", "very",
            "much", "well", "used", "using", "magazine", "magazines", "american",
            "published",
        }
        all_tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text.lower())
        proper_set = {t for t in all_tokens if len(t) >= min_len and t not in lower_stopwords}
    return proper_set


def add_hyperlink_edges(G, chunks, title_to_nodes):
    """Fast hyperlink edges via string matching (no regex)."""
    title_lower_map = {
        title.lower(): nodes
        for title, nodes in title_to_nodes.items()
        if title
    }
    hyperlink_count = 0
    for i, chunk in enumerate(chunks):
        text_lower  = chunk.page_content.lower()
        chunk_title = chunk.metadata.get("title", "").lower()
        for other_title_lower, other_nodes in title_lower_map.items():
            if other_title_lower == chunk_title:
                continue
            if other_title_lower in text_lower:
                for j in other_nodes:
                    if i == j:
                        continue
                    if G.has_edge(i, j):
                        G[i][j]["edge_types"].add("hyperlink")
                        G[i][j]["weight"] = max(G[i][j]["weight"], 1.0)
                    else:
                        G.add_edge(i, j, weight=1.0, edge_types={"hyperlink"})
                        hyperlink_count += 1
    return hyperlink_count


def build_hybrid_chunk_graph(
    split="train",
    max_samples=10000,
    chunk_size=500,           # reverted from 500 — more chunks = better recall
    chunk_overlap=100,         # reverted from 100
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    semantic_top_k=3,
    min_similarity=0.60,
    add_same_title_edges=True,
    add_adjacent_chunk_edges=True,
    add_hyperlink_edges_flag=True,
    add_keyword_overlap_edges=True,
    min_keyword_overlap=3,
    max_nodes_for_semantic_knn=10000,  # skip semantic KNN above this threshold
):
    """
    Hybrid graph — chunk_size=300 reverted (500 hurt recall at scale).
    Semantic KNN skipped for graphs > max_nodes_for_semantic_knn (O(N²) too slow).
    """
    chunks, embeddings, _ = generate_chunk_embeddings(
        split=split,
        max_samples=max_samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model_name,
    )

    G = nx.Graph()
    for i, chunk in enumerate(chunks):
        G.add_node(
            i,
            title=chunk.metadata.get("title", ""),
            source=chunk.metadata.get("source", ""),
            text=chunk.page_content,
            chunk_id=i,
        )

    title_to_nodes = defaultdict(list)
    for i, chunk in enumerate(chunks):
        title_to_nodes[chunk.metadata.get("title", "")].append(i)

    # ---------- 1. Same-title edges ----------
    if add_same_title_edges:
        for title, node_ids in title_to_nodes.items():
            if len(node_ids) < 2:
                continue
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    u, v = node_ids[i], node_ids[j]
                    if G.has_edge(u, v):
                        G[u][v]["edge_types"].add("same_title")
                        G[u][v]["weight"] = max(G[u][v]["weight"], 0.95)
                    else:
                        G.add_edge(u, v, weight=0.95, edge_types={"same_title"})

    # ---------- 2. Adjacent chunk edges ----------
    if add_adjacent_chunk_edges:
        for title, node_ids in title_to_nodes.items():
            if len(node_ids) < 2:
                continue
            for i in range(len(sorted(node_ids)) - 1):
                u, v = sorted(node_ids)[i], sorted(node_ids)[i + 1]
                if G.has_edge(u, v):
                    G[u][v]["edge_types"].add("adjacent_chunk")
                    G[u][v]["weight"] = max(G[u][v]["weight"], 0.90)
                else:
                    G.add_edge(u, v, weight=0.90, edge_types={"adjacent_chunk"})

    # ---------- 3. Hyperlink edges ----------
    if add_hyperlink_edges_flag:
        print("Building hyperlink edges...")
        hyperlink_count = add_hyperlink_edges(G, chunks, title_to_nodes)
        print(f"  → {hyperlink_count} hyperlink edges added")

    # ---------- 4. Semantic KNN (skipped for large graphs) ----------
    num_chunks = len(chunks)
    if semantic_top_k > 0 and num_chunks <= max_nodes_for_semantic_knn:
        print(f"Building semantic KNN edges ({num_chunks} nodes)...")
        sim_matrix = embeddings @ embeddings.T
        for i in range(num_chunks):
            sim_scores = sim_matrix[i].copy()
            sim_scores[i] = -1.0
            neighbor_indices = np.argsort(sim_scores)[-semantic_top_k:][::-1]
            for j in neighbor_indices:
                score = float(sim_scores[j])
                if score < min_similarity:
                    continue
                if G.has_edge(i, j):
                    G[i][j]["edge_types"].add("semantic_knn")
                    G[i][j]["weight"] = max(G[i][j]["weight"], score * 0.5)
                else:
                    G.add_edge(i, j, weight=score * 0.5, edge_types={"semantic_knn"})
    else:
        print(f"Skipping semantic KNN (graph has {num_chunks} nodes > {max_nodes_for_semantic_knn} limit)")

    # ---------- 5. Keyword overlap edges ----------
    if add_keyword_overlap_edges:
        print("Building keyword overlap edges...")
        node_keywords    = {}
        keyword_to_nodes = defaultdict(set)
        for i, chunk in enumerate(chunks):
            kws = simple_keywords(chunk.page_content)
            node_keywords[i] = kws
            for kw in kws:
                keyword_to_nodes[kw].add(i)

        candidate_pairs = defaultdict(int)
        for kw, node_set in keyword_to_nodes.items():
            node_list = list(node_set)
            if len(node_list) > 30:
                continue
            for i in range(len(node_list)):
                for j in range(i + 1, len(node_list)):
                    u, v = node_list[i], node_list[j]
                    if chunks[u].metadata.get("title") == chunks[v].metadata.get("title"):
                        continue
                    pair = (u, v) if u < v else (v, u)
                    candidate_pairs[pair] += 1

        for (u, v), overlap_count in candidate_pairs.items():
            if overlap_count < min_keyword_overlap:
                continue
            weight = min(0.85, 0.55 + 0.05 * overlap_count)
            if G.has_edge(u, v):
                G[u][v]["edge_types"].add("keyword_overlap")
                G[u][v]["weight"] = max(G[u][v]["weight"], weight)
            else:
                G.add_edge(u, v, weight=weight, edge_types={"keyword_overlap"})

    return G, chunks, embeddings


def get_bridge_edges_only(G):
    bridge_edge_types = {"hyperlink", "keyword_overlap", "adjacent_chunk", "same_title"}
    bridge_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("edge_types", set()) & bridge_edge_types
    ]
    return G.edge_subgraph(bridge_edges)


if __name__ == "__main__":
    G, chunks, embeddings = build_hybrid_chunk_graph(
        split="train",
        max_samples=10000,
        chunk_size=500,       # reverted
        chunk_overlap=100,     # reverted
        semantic_top_k=3,
        min_similarity=0.60,
        add_same_title_edges=True,
        add_adjacent_chunk_edges=True,
        add_hyperlink_edges_flag=True,
        add_keyword_overlap_edges=True,
        min_keyword_overlap=3,
        max_nodes_for_semantic_knn=10000,
    )

    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    edge_type_counts = defaultdict(int)
    for _, _, data in G.edges(data=True):
        for edge_type in data["edge_types"]:
            edge_type_counts[edge_type] += 1

    print("\nEdge type counts:")
    for k, v in sorted(edge_type_counts.items()):
        print(f"  {k}: {v}")