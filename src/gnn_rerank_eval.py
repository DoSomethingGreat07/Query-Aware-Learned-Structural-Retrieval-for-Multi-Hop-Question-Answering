import re
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from gnn_train import train_graphsage


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


def build_projection_matrix(base_embeddings, gnn_embeddings):
    W_T = np.linalg.lstsq(base_embeddings, gnn_embeddings, rcond=None)[0]
    return W_T.T


def project_query_to_gnn_space(query_vec, projection_matrix):
    projected = projection_matrix @ query_vec
    norm = np.linalg.norm(projected)
    return projected / (norm + 1e-12)


def retrieve_seed_candidates(query, base_embeddings, model, seed_k=20):
    query_vec = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True,
    )[0]
    scores  = base_embeddings @ query_vec
    top_idx = np.argsort(scores)[-seed_k:][::-1]
    return top_idx, query_vec, scores


def retrieve_two_hop_gnn(
    query, base_embeddings, gnn_embeddings, chunks, graph, model,
    projection_matrix, seed_k=20, final_k=5,
    hop2_bridge_weight=0.4, hop2_min_score=0.10,
):
    seed_indices, query_vec, base_scores = retrieve_seed_candidates(
        query, base_embeddings, model, seed_k=seed_k
    )
    query_proj  = project_query_to_gnn_space(query_vec, projection_matrix)
    hop1_ids    = seed_indices[:final_k].tolist()
    hop1_titles = {chunks[i].metadata.get("title", "") for i in hop1_ids}
    hop1_texts  = [chunks[i].page_content for i in hop1_ids]

    BRIDGE_TYPES = {"hyperlink", "keyword_overlap"}
    hop2_candidates = {}
    for seed in hop1_ids:
        if seed not in graph:
            continue
        for neighbor in graph.neighbors(seed):
            if neighbor in set(hop1_ids):
                continue
            n_title    = chunks[neighbor].metadata.get("title", "")
            edge_types = graph[seed][neighbor].get("edge_types", set())
            if n_title in hop1_titles:
                continue
            if edge_types & BRIDGE_TYPES:
                w = graph[seed][neighbor].get("weight", 0.5)
                if neighbor not in hop2_candidates or hop2_candidates[neighbor] < w:
                    hop2_candidates[neighbor] = w

    hop2_scored = []
    for cand_id, edge_weight in hop2_candidates.items():
        cand_gnn = gnn_embeddings[cand_id]
        cand_gnn = cand_gnn / (np.linalg.norm(cand_gnn) + 1e-12)
        gnn_sim  = float(cand_gnn @ query_proj)
        t_score  = title_bridge_score(chunks[cand_id].page_content, hop1_titles)
        k_score  = keyword_bridge_score(chunks[cand_id].page_content, hop1_texts) \
                   if t_score == 0.0 else 0.0
        b_score  = max(t_score, k_score)
        combined = (1 - hop2_bridge_weight) * gnn_sim + hop2_bridge_weight * b_score
        if combined < hop2_min_score:
            continue
        hop2_scored.append((cand_id, combined, b_score))

    hop2_scored.sort(key=lambda x: x[1], reverse=True)

    seen = set(); results = []
    for idx in hop1_ids:
        if idx not in seen:
            seen.add(idx)
            results.append({"chunk_id": int(idx), "score": float(base_scores[idx]),
                            "hop": 1, "metadata": chunks[idx].metadata,
                            "text": chunks[idx].page_content})
    hop2_count = 0
    for cand_id, combined, b_score in hop2_scored:
        if hop2_count >= final_k: break
        if cand_id not in seen:
            seen.add(cand_id)
            results.append({"chunk_id": int(cand_id), "score": combined,
                            "bridge_score": round(b_score, 4), "hop": 2,
                            "metadata": chunks[cand_id].metadata,
                            "text": chunks[cand_id].page_content})
            hop2_count += 1
    return results


def evaluate_gnn_rerank(
    split="train", max_samples=10000, seed_k=20, final_k=5,
    hop2_bridge_weight=0.4, hop2_min_score=0.10, epochs=30,
):
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    _, graph, chunks, base_embeddings, gnn_embeddings = train_graphsage(
        split=split, max_samples=max_samples, epochs=epochs,
    )

    print("Building query projection matrix...")
    projection_matrix = build_projection_matrix(base_embeddings, gnn_embeddings)
    print(f"  base_dim={base_embeddings.shape[1]}, gnn_dim={gnn_embeddings.shape[1]}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    exact_hits = partial_hits = hop1_contributed = hop2_contributed = 0
    type_counters = {
        "bridge":     {"exact": 0, "partial": 0, "hop2": 0, "total": 0},
        "comparison": {"exact": 0, "partial": 0, "hop2": 0, "total": 0},
        "unknown":    {"exact": 0, "partial": 0, "hop2": 0, "total": 0},
    }

    for item in dataset:
        question    = item["question"]
        gold_titles = set(item["supporting_facts"]["title"])
        q_type      = item.get("type", "unknown")

        results = retrieve_two_hop_gnn(
            query=question, base_embeddings=base_embeddings,
            gnn_embeddings=gnn_embeddings, chunks=chunks,
            graph=graph, model=model, projection_matrix=projection_matrix,
            seed_k=seed_k, final_k=final_k,
            hop2_bridge_weight=hop2_bridge_weight, hop2_min_score=hop2_min_score,
        )

        retrieved_titles = {r["metadata"]["title"] for r in results}
        overlap = gold_titles & retrieved_titles
        is_exact   = int(len(overlap) == len(gold_titles))
        is_partial = int(len(overlap) > 0)
        exact_hits += is_exact; partial_hits += is_partial

        hop1_titles = {r["metadata"]["title"] for r in results if r.get("hop") == 1}
        hop2_titles = {r["metadata"]["title"] for r in results if r.get("hop") == 2}
        h2 = int(bool(gold_titles & hop2_titles))
        if gold_titles & hop1_titles: hop1_contributed += 1
        if h2:                        hop2_contributed += 1

        bucket = q_type if q_type in type_counters else "unknown"
        type_counters[bucket]["exact"]   += is_exact
        type_counters[bucket]["partial"] += is_partial
        type_counters[bucket]["hop2"]    += h2
        type_counters[bucket]["total"]   += 1

    total   = len(dataset)
    metrics = {
        "total_questions":                       total,
        f"gnn_support_recall@{final_k}":         round(exact_hits / total, 4),
        f"gnn_partial_support_recall@{final_k}": round(partial_hits / total, 4),
        f"gnn_hop1_contributed@{final_k}":       round(hop1_contributed / total, 4),
        f"gnn_hop2_contributed@{final_k}":       round(hop2_contributed / total, 4),
    }
    for q_type, c in type_counters.items():
        if c["total"] == 0: continue
        metrics[f"gnn_{q_type}_support_recall@{final_k}"] = round(c["exact"]   / c["total"], 4)
        metrics[f"gnn_{q_type}_partial_recall@{final_k}"] = round(c["partial"] / c["total"], 4)
        metrics[f"gnn_{q_type}_hop2_hit@{final_k}"]       = round(c["hop2"]    / c["total"], 4)
        metrics[f"gnn_{q_type}_count"]                    = c["total"]
    return metrics


if __name__ == "__main__":
    print("=" * 55)
    print("TWO-HOP GNN RERANK EVALUATION (10k samples)")
    print("=" * 55)
    metrics = evaluate_gnn_rerank(
        split="train", max_samples=10000, seed_k=20, final_k=5,
        hop2_bridge_weight=0.4, hop2_min_score=0.10, epochs=30,
    )
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    h1 = metrics.get("gnn_hop1_contributed@5", 0)
    h2 = metrics.get("gnn_hop2_contributed@5", 0)
    print(f"\n{'✓' if h1 >= 0.85 else '⚠'} hop1_contributed = {h1}")
    print(f"{'✓' if h2 >= 0.10 else '⚠'} hop2_contributed = {h2}")