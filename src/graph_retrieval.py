import re
import numpy as np
from sentence_transformers import SentenceTransformer

from hybrid_graph_builder import build_hybrid_chunk_graph, get_bridge_edges_only


def cosine_scores(query_vec, candidate_matrix):
    return candidate_matrix @ query_vec


# ── Bridge scoring ────────────────────────────────────────────────────────────

def extract_titles_from_chunks(chunk_list):
    return {c.metadata.get("title", "") for c in chunk_list if c.metadata.get("title")}


def title_bridge_score(node_text, hop1_titles):
    """
    Check if any hop-1 article title is mentioned in this node's text.
    Most reliable bridge signal for HotpotQA.
    """
    if not hop1_titles:
        return 0.0
    hits = sum(1 for t in hop1_titles if t and t.lower() in node_text.lower())
    return min(1.0, hits / max(len(hop1_titles), 1))


def keyword_bridge_score(node_text, hop1_texts, min_len=5):
    """
    Fallback: shared content words between node and hop-1 docs.
    Used only when title_bridge_score = 0.
    """
    stopwords = {
        "which", "where", "when", "what", "that", "this", "with", "from",
        "have", "were", "about", "after", "before", "into", "their", "there",
        "been", "being", "also", "some", "such", "most", "more", "only",
        "other", "many", "would", "could", "should", "these", "those",
    }

    def keywords(text):
        tokens = re.findall(r"[a-zA-Z]{%d,}" % min_len, text.lower())
        return {t for t in tokens if t not in stopwords}

    hop1_kws = set()
    for t in hop1_texts:
        hop1_kws |= keywords(t)

    node_kws = keywords(node_text)
    if not hop1_kws:
        return 0.0
    return min(1.0, len(node_kws & hop1_kws) / max(len(hop1_kws), 1))


# ── Hop-2 candidate expansion ─────────────────────────────────────────────────

def get_hop2_candidates(seed_ids, graph, chunks, use_bridge_only=True):
    """
    Expand from hop-1 seeds via graph edges.
    Skips same-title neighbors — they can't help recall on the second gold doc.

    Edge priority:
      hyperlink        → 1.0
      keyword_overlap  → 0.8
      adjacent_chunk   → 0.3  (same-doc, less useful for multi-hop)
      same_title       → 0.2
      semantic_knn     → 0.05 (demoted — redundant with retrieval)
    """
    edge_priority = {
        "hyperlink":       1.0,
        "keyword_overlap": 0.8,
        "adjacent_chunk":  0.3,
        "same_title":      0.2,
        "semantic_knn":    0.05,
    }

    seed_set    = set(seed_ids)
    seed_titles = {chunks[s].metadata.get("title", "") for s in seed_ids}
    candidates  = {}

    for seed in seed_ids:
        if seed not in graph:
            continue
        for neighbor in graph.neighbors(seed):
            if neighbor in seed_set:
                continue

            # Skip same-title neighbors
            neighbor_title = chunks[neighbor].metadata.get("title", "")
            if use_bridge_only and neighbor_title in seed_titles:
                continue

            edge_types = graph[seed][neighbor].get("edge_types", set())
            priority   = max(
                (edge_priority.get(et, 0.0) for et in edge_types),
                default=0.0,
            )
            if priority == 0.0:
                continue

            if neighbor not in candidates or candidates[neighbor] < priority:
                candidates[neighbor] = priority

    return candidates


# ── Main two-hop retrieval ────────────────────────────────────────────────────

def graph_retrieve_two_hop(
    query,
    chunks,
    embeddings,
    graph,
    model,
    seed_k=10,
    final_k=5,
    hop2_weight=0.4,
    hop2_min_score=0.10,      # filter out noisy low-score hop-2 candidates
    use_bridge_graph=True,
):
    """
    Two-hop graph retrieval for multi-hop questions (HotpotQA).

    Hop 1: dense retrieval → top seed_k candidates
    Hop 2: graph expand from hop-1 → score by query_sim + bridge_score
           filtered by hop2_min_score to remove noise

    Hop-1 and hop-2 each get their own final_k quota —
    they do NOT compete, guaranteeing hop-2 docs appear in output.
    """
    query_vec  = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    all_scores = embeddings @ query_vec

    # ── HOP 1 ─────────────────────────────────────────────────────────────────
    hop1_ids    = np.argsort(all_scores)[-seed_k:][::-1].tolist()
    hop1_chunks = [chunks[i] for i in hop1_ids]
    hop1_titles = extract_titles_from_chunks(hop1_chunks)
    hop1_texts  = [c.page_content for c in hop1_chunks]

    # ── HOP 2 ─────────────────────────────────────────────────────────────────
    hop2_candidates = get_hop2_candidates(
        seed_ids=hop1_ids,
        graph=graph,
        chunks=chunks,
        use_bridge_only=True,
    )

    hop2_scored = []
    for cand_id, edge_priority in hop2_candidates.items():
        query_sim = float(all_scores[cand_id])

        t_score  = title_bridge_score(chunks[cand_id].page_content, hop1_titles)
        k_score  = keyword_bridge_score(chunks[cand_id].page_content, hop1_texts) \
                   if t_score == 0.0 else 0.0
        b_score  = max(t_score, k_score)

        combined = (1 - hop2_weight) * query_sim + hop2_weight * (b_score * edge_priority)

        # Filter low-quality candidates before they pollute results
        if combined < hop2_min_score:
            continue

        hop2_scored.append((cand_id, combined, query_sim, b_score))

    hop2_scored.sort(key=lambda x: x[1], reverse=True)

    # ── MERGE: guaranteed separate quota for hop-1 and hop-2 ─────────────────
    seen    = set()
    results = []

    # Hop-1: top final_k
    hop1_count = 0
    for idx in hop1_ids:
        if hop1_count >= final_k:
            break
        if idx not in seen:
            seen.add(idx)
            results.append({
                "chunk_id":     int(idx),
                "score":        float(all_scores[idx]),
                "hop":          1,
                "bridge_score": 0.0,
                "metadata":     chunks[idx].metadata,
                "text":         chunks[idx].page_content,
            })
            hop1_count += 1

    # Hop-2: top final_k (separate quota — not competing with hop-1)
    hop2_count = 0
    for cand_id, combined, query_sim, b_score in hop2_scored:
        if hop2_count >= final_k:
            break
        if cand_id not in seen:
            seen.add(cand_id)
            results.append({
                "chunk_id":     int(cand_id),
                "score":        combined,
                "hop":          2,
                "bridge_score": round(b_score, 4),
                "metadata":     chunks[cand_id].metadata,
                "text":         chunks[cand_id].page_content,
            })
            hop2_count += 1

    return results   # up to 2 * final_k results total


# ── Backward-compatible single-hop ────────────────────────────────────────────

def graph_retrieve_top_k(
    query,
    chunks,
    embeddings,
    graph,
    model,
    seed_k=5,
    neighbor_hops=1,
    final_k=5,
):
    """Original single-hop retrieval — kept for ablation in eval files."""
    query_vec    = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    seed_scores  = embeddings @ query_vec
    seed_indices = np.argsort(seed_scores)[-seed_k:][::-1]

    candidate_ids = set(seed_indices.tolist())
    frontier      = set(seed_indices.tolist())

    for _ in range(neighbor_hops):
        next_frontier = set()
        for node_id in frontier:
            next_frontier.update(graph.neighbors(node_id))
        candidate_ids.update(next_frontier)
        frontier = next_frontier

    candidate_ids    = sorted(candidate_ids)
    candidate_matrix = embeddings[candidate_ids]
    candidate_scores = cosine_scores(query_vec, candidate_matrix)
    ranked_local     = np.argsort(candidate_scores)[-final_k:][::-1]

    return [
        {
            "chunk_id": int(candidate_ids[i]),
            "score":    float(candidate_scores[i]),
            "hop":      1,
            "metadata": chunks[candidate_ids[i]].metadata,
            "text":     chunks[candidate_ids[i]].page_content,
        }
        for i in ranked_local
    ]


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    G, chunks, embeddings = build_hybrid_chunk_graph(
        split="train",
        max_samples=10000,
        chunk_size=500,
        chunk_overlap=100,
        semantic_top_k=3,
        min_similarity=0.60,
        add_same_title_edges=True,
        add_adjacent_chunk_edges=True,
        add_hyperlink_edges_flag=True,
        add_keyword_overlap_edges=True,
        min_keyword_overlap=3,
    )

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Use a bridge question — comparison questions don't need hop-2
    query = "Were Pavel Urysohn and Nikolai Alexandrovich Vasiliev both known for their contributions to mathematics?"

    print("=" * 60)
    print("TWO-HOP GRAPH RETRIEVAL")
    print("=" * 60)

    results = graph_retrieve_two_hop(
        query=query,
        chunks=chunks,
        embeddings=embeddings,
        graph=G,
        model=model,
        seed_k=10,
        final_k=5,
        hop2_weight=0.4,
        hop2_min_score=0.10,
    )

    hop1_count = sum(1 for r in results if r["hop"] == 1)
    hop2_count = sum(1 for r in results if r["hop"] == 2)
    print(f"\nTotal: {len(results)} | Hop-1: {hop1_count} | Hop-2: {hop2_count}")

    for rank, item in enumerate(results, 1):
        print(f"\nRank {rank} | Hop {item['hop']} | Score: {item['score']:.4f} | Bridge: {item['bridge_score']:.4f}")
        print(f"  Title: {item['metadata'].get('title', '')}")
        print(f"  Text:  {item['text'][:200]}")