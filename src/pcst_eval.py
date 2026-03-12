from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from hybrid_graph_builder import build_hybrid_chunk_graph
from pcst import retrieve_with_pcst


def evaluate_pcst_retrieval(
    split="train", max_samples=10000, k=5,
    seed_k=20,               # increased from 10
    edge_cost_scale=1.0,
    hop2_bridge_weight=0.4, use_bridge_graph=True,
):
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    G, chunks, embeddings = build_hybrid_chunk_graph(
        split=split, max_samples=max_samples,
        chunk_size=500, chunk_overlap=100,    # reverted
        semantic_top_k=3, min_similarity=0.60,
        add_same_title_edges=True, add_adjacent_chunk_edges=True,
        add_hyperlink_edges_flag=True, add_keyword_overlap_edges=True,
        min_keyword_overlap=3, max_nodes_for_semantic_knn=10000,
    )

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    exact_hits = partial_hits = hop1_contributed = hop2_contributed = 0
    type_counters = {
        "bridge":     {"exact": 0, "partial": 0, "hop2": 0, "total": 0},
        "comparison": {"exact": 0, "partial": 0, "hop2": 0, "total": 0},
        "unknown":    {"exact": 0, "partial": 0, "hop2": 0, "total": 0},
    }

    for item in tqdm(dataset, desc="Evaluating PCST", unit="q"):
        question    = item["question"]
        gold_titles = set(item["supporting_facts"]["title"])
        q_type      = item.get("type", "unknown")

        results = retrieve_with_pcst(
            query=question, chunks=chunks, embeddings=embeddings,
            graph=G, model=model, seed_k=seed_k, final_k=k,
            edge_cost_scale=edge_cost_scale,
            hop2_bridge_weight=hop2_bridge_weight,
            use_bridge_graph=use_bridge_graph,
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
        "total_questions":                  total,
        f"pcst_support_recall@{k}":         round(exact_hits / total, 4),
        f"pcst_partial_support_recall@{k}": round(partial_hits / total, 4),
        f"pcst_hop1_contributed@{k}":       round(hop1_contributed / total, 4),
        f"pcst_hop2_contributed@{k}":       round(hop2_contributed / total, 4),
    }
    for q_type, c in type_counters.items():
        if c["total"] == 0: continue
        metrics[f"pcst_{q_type}_support_recall@{k}"] = round(c["exact"]   / c["total"], 4)
        metrics[f"pcst_{q_type}_partial_recall@{k}"] = round(c["partial"] / c["total"], 4)
        metrics[f"pcst_{q_type}_hop2_hit@{k}"]       = round(c["hop2"]    / c["total"], 4)
        metrics[f"pcst_{q_type}_count"]              = c["total"]
    return metrics


if __name__ == "__main__":
    print("=" * 55)
    print("TWO-HOP PCST RETRIEVAL EVALUATION (10k samples)")
    print("=" * 55)
    metrics = evaluate_pcst_retrieval(split="train", max_samples=10000, k=5)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    hop2 = metrics.get("pcst_hop2_contributed@5", 0)
    if hop2 < 0.10:
        print("\n⚠ hop2_contributed low — check hyperlink edges in graph builder")
    else:
        print(f"\n✓ Hop-2 firing on {hop2*100:.1f}% of questions")