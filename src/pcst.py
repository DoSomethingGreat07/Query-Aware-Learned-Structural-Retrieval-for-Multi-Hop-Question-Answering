import heapq
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from hybrid_graph_builder import build_hybrid_chunk_graph, get_bridge_edges_only


# ── Helpers ───────────────────────────────────────────────────────────────────

def baseline_retrieve(query, chunks, embeddings, model, top_k=10):
    query_vec = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True,
    )[0]
    scores  = embeddings @ query_vec
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return query_vec, scores, top_idx


def extract_titles_from_chunks(chunk_list):
    return {c.metadata.get("title", "") for c in chunk_list if c.metadata.get("title")}


def title_bridge_score(node_text, hop1_titles):
    if not hop1_titles:
        return 0.0
    hits = sum(1 for t in hop1_titles if t and t.lower() in node_text.lower())
    return min(1.0, hits / max(len(hop1_titles), 1))


def keyword_bridge_score(node_text, hop1_texts, min_len=5):
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


# ── Edge cost: type-aware ─────────────────────────────────────────────────────

BRIDGE_EDGE_TYPES     = {"hyperlink", "keyword_overlap"}
STRUCTURAL_EDGE_TYPES = {"adjacent_chunk", "same_title"}


def compute_edge_cost(edge_data, edge_cost_scale):
    """
    Hyperlink / keyword edges → LOW cost  (PCST prefers to traverse them)
    Semantic edges            → HIGH cost (penalised — redundant with retrieval)
    """
    edge_types = edge_data.get("edge_types", set())
    weight     = edge_data.get("weight", 0.5)

    if edge_types & BRIDGE_EDGE_TYPES:
        base_cost = max(0.0, 0.2 - 0.05 * len(edge_types & BRIDGE_EDGE_TYPES))
    elif edge_types & STRUCTURAL_EDGE_TYPES:
        base_cost = 1.0 - weight
    else:
        base_cost = (1.0 - weight) * 2.0   # semantic_knn penalised heavily

    return edge_cost_scale * base_cost


# ── PCST core — seeds are LOCKED anchors ─────────────────────────────────────

def pcst_select_subgraph(
    graph,
    seed_nodes,
    node_prizes,
    max_extra=5,          # max ADDITIONAL nodes beyond seeds
    edge_cost_scale=1.0,
):
    """
    Greedy PCST-style expansion.

    KEY FIX vs previous version:
      Seeds are LOCKED — they are always in the output.
      max_extra controls how many ADDITIONAL nodes PCST may add.
      This prevents PCST from evicting good seed nodes (which caused
      hop1_contributed to drop from 0.94 → 0.84).

    gain(neighbor) = node_prize[neighbor] - edge_cost(seed→neighbor)
    """
    selected   = set(seed_nodes)   # seeds locked in permanently
    frontier   = []
    extra_added = 0

    def push_frontier_from(node):
        for nbr in graph.neighbors(node):
            if nbr in selected:
                continue
            cost = compute_edge_cost(graph[node][nbr], edge_cost_scale)
            gain = float(node_prizes[nbr]) - cost
            heapq.heappush(frontier, (-gain, node, nbr))

    for s in seed_nodes:
        if s in graph:
            push_frontier_from(s)

    while frontier and extra_added < max_extra:
        neg_gain, parent, nbr = heapq.heappop(frontier)
        if nbr in selected:
            continue
        if -neg_gain <= 0:
            break                  # no more positive-gain nodes
        selected.add(nbr)
        push_frontier_from(nbr)
        extra_added += 1

    return sorted(selected)


# ── Two-hop PCST retrieval ────────────────────────────────────────────────────

def retrieve_with_pcst(
    query,
    chunks,
    embeddings,
    graph,
    model,
    seed_k=10,
    final_k=5,
    edge_cost_scale=1.0,
    hop2_bridge_weight=0.4,
    hop2_min_score=0.10,
    use_bridge_graph=True,
):
    """
    Two-hop PCST retrieval.

    Hop 1:
      - Dense retrieval → top seed_k seeds  (locked anchors)
      - PCST adds up to seed_k extra nodes using query_sim prizes
      - Seeds are always kept → hop1_contributed stays high

    Hop 2:
      - Extract bridge signal from hop-1 results
      - Prizes = query_sim + bridge_score (title-based + keyword fallback)
      - PCST expands from hop-1 nodes using bridge-aware graph
      - hop-1 and hop-2 have SEPARATE quotas → hop-2 always appears

    Final: deduplicated, up to 2*final_k results
    """
    query_vec, scores, seed_idx = baseline_retrieve(
        query=query, chunks=chunks, embeddings=embeddings,
        model=model, top_k=seed_k,
    )

    # ── HOP 1 PCST ───────────────────────────────────────────────────────────
    # Seeds locked; PCST may add up to seed_k extra nodes
    hop1_extra = pcst_select_subgraph(
        graph=graph,
        seed_nodes=seed_idx.tolist(),
        node_prizes=scores,
        max_extra=seed_k,
        edge_cost_scale=edge_cost_scale,
    )

    # Seeds always first, then PCST extras, capped at seed_k
    hop1_selected = list(
        dict.fromkeys(
            seed_idx.tolist() + [n for n in hop1_extra if n not in set(seed_idx.tolist())]
        )
    )[:seed_k]

    # ── Extract bridge signal from hop-1 ─────────────────────────────────────
    hop1_chunks  = [chunks[i] for i in hop1_selected]
    hop1_titles  = extract_titles_from_chunks(hop1_chunks)
    hop1_texts   = [c.page_content for c in hop1_chunks]
    hop1_title_set = {chunks[i].metadata.get("title", "") for i in hop1_selected}

    # ── Compute hop-2 prizes ──────────────────────────────────────────────────
    hop2_prizes = np.zeros(len(chunks), dtype=np.float32)
    for i in range(len(chunks)):
        # Skip nodes already in hop-1
        if i in set(hop1_selected):
            hop2_prizes[i] = 0.0
            continue

        # Skip same-title nodes — won't help recall on second doc
        if chunks[i].metadata.get("title", "") in hop1_title_set:
            hop2_prizes[i] = 0.0
            continue

        query_sim = float(scores[i])
        t_score   = title_bridge_score(chunks[i].page_content, hop1_titles)
        k_score   = keyword_bridge_score(chunks[i].page_content, hop1_texts) \
                    if t_score == 0.0 else 0.0
        b_score   = max(t_score, k_score)

        hop2_prizes[i] = (
            (1 - hop2_bridge_weight) * query_sim
            + hop2_bridge_weight * b_score
        )

    # ── HOP 2 PCST ───────────────────────────────────────────────────────────
    traverse_graph = get_bridge_edges_only(graph) if use_bridge_graph else graph

    hop2_extra = pcst_select_subgraph(
        graph=traverse_graph,
        seed_nodes=hop1_selected,
        node_prizes=hop2_prizes,
        max_extra=seed_k,
        edge_cost_scale=edge_cost_scale * 0.7,
    )

    hop2_candidates = [
        n for n in hop2_extra
        if n not in set(hop1_selected)
        and float(hop2_prizes[n]) >= hop2_min_score
    ]

    # Sort hop-2 candidates by prize
    hop2_candidates.sort(key=lambda i: float(hop2_prizes[i]), reverse=True)

    # ── MERGE: separate quotas ────────────────────────────────────────────────
    seen    = set()
    results = []

    # Hop-1: top final_k (seeds first)
    hop1_count = 0
    for idx in hop1_selected:
        if hop1_count >= final_k:
            break
        if idx not in seen:
            seen.add(idx)
            b = title_bridge_score(chunks[idx].page_content, hop1_titles)
            results.append({
                "chunk_id":     int(idx),
                "score":        float(scores[idx]),
                "query_score":  float(scores[idx]),
                "bridge_score": round(b, 4),
                "hop":          1,
                "metadata":     chunks[idx].metadata,
                "text":         chunks[idx].page_content,
            })
            hop1_count += 1

    # Hop-2: top final_k (separate quota)
    hop2_count = 0
    for idx in hop2_candidates:
        if hop2_count >= final_k:
            break
        if idx not in seen:
            seen.add(idx)
            b = max(
                title_bridge_score(chunks[idx].page_content, hop1_titles),
                keyword_bridge_score(chunks[idx].page_content, hop1_texts),
            )
            results.append({
                "chunk_id":     int(idx),
                "score":        float(hop2_prizes[idx]),
                "query_score":  float(scores[idx]),
                "bridge_score": round(b, 4),
                "hop":          2,
                "metadata":     chunks[idx].metadata,
                "text":         chunks[idx].page_content,
            })
            hop2_count += 1

    return results   # up to 2 * final_k


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
    query = "Were Pavel Urysohn and Nikolai Alexandrovich Vasiliev both known for their contributions to mathematics?"

    results = retrieve_with_pcst(
        query=query,
        chunks=chunks,
        embeddings=embeddings,
        graph=G,
        model=model,
        seed_k=10,
        final_k=5,
        edge_cost_scale=1.0,
        hop2_bridge_weight=0.4,
        hop2_min_score=0.10,
        use_bridge_graph=True,
    )

    print("=" * 60)
    print("TWO-HOP PCST RETRIEVAL")
    print("=" * 60)

    hop1_count = sum(1 for r in results if r["hop"] == 1)
    hop2_count = sum(1 for r in results if r["hop"] == 2)
    print(f"\nTotal: {len(results)} | Hop-1: {hop1_count} | Hop-2: {hop2_count}")

    for rank, item in enumerate(results, 1):
        print(f"\nRank {rank} | Hop {item['hop']} | Score: {item['score']:.4f} "
              f"| Query: {item['query_score']:.4f} | Bridge: {item['bridge_score']:.4f}")
        print(f"  Title: {item['metadata'].get('title', '')}")
        print(f"  Text:  {item['text'][:200]}")