from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from hybrid_graph_builder import build_hybrid_chunk_graph
from graph_retrieval import graph_retrieve_two_hop, graph_retrieve_top_k


def evaluate_graph_retrieval(
    split="train",
    max_samples=10000,
    k=5,
    seed_k=10,
    hop2_weight=0.4,
    use_two_hop=True,
):
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    G, chunks, embeddings = build_hybrid_chunk_graph(
        split=split,
        max_samples=max_samples,
        chunk_size=500,
        chunk_overlap=100,
        semantic_top_k=3,
        min_similarity=0.60,
        add_same_title_edges=True,
        add_adjacent_chunk_edges=True,
        add_hyperlink_edges_flag=True,
        add_keyword_overlap_edges=True,
        min_keyword_overlap=3,
        max_nodes_for_semantic_knn=10000,
    )

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    exact_hits       = 0
    partial_hits     = 0
    hop1_contributed = 0
    hop2_contributed = 0

    type_counters = {
        "bridge":     {"exact": 0, "partial": 0, "hop2": 0, "total": 0},
        "comparison": {"exact": 0, "partial": 0, "hop2": 0, "total": 0},
        "unknown":    {"exact": 0, "partial": 0, "hop2": 0, "total": 0},
    }

    for item in dataset:
        question    = item["question"]
        gold_titles = set(item["supporting_facts"]["title"])
        q_type      = item.get("type", "unknown")

        if use_two_hop:
            results = graph_retrieve_two_hop(
                query=question,
                chunks=chunks,
                embeddings=embeddings,
                graph=G,
                model=model,
                seed_k=seed_k,
                final_k=k,
                hop2_weight=hop2_weight,
                use_bridge_graph=True,
            )
        else:
            results = graph_retrieve_top_k(
                query=question,
                chunks=chunks,
                embeddings=embeddings,
                graph=G,
                model=model,
                seed_k=seed_k,
                neighbor_hops=1,
                final_k=k,
            )

        retrieved_titles = {r["metadata"]["title"] for r in results}
        overlap = gold_titles & retrieved_titles

        is_exact   = int(len(overlap) == len(gold_titles))
        is_partial = int(len(overlap) > 0)

        exact_hits   += is_exact
        partial_hits += is_partial

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

    total = len(dataset)
    tag   = "two_hop" if use_two_hop else "single_hop"

    metrics = {
        "total_questions":              total,
        "mode":                         tag,
        f"support_recall@{k}":         round(exact_hits / total, 4),
        f"partial_support_recall@{k}": round(partial_hits / total, 4),
        f"hop1_contributed@{k}":       round(hop1_contributed / total, 4),
        f"hop2_contributed@{k}":       round(hop2_contributed / total, 4),
    }

    for q_type, c in type_counters.items():
        if c["total"] == 0:
            continue
        metrics[f"{q_type}_support_recall@{k}"] = round(c["exact"]   / c["total"], 4)
        metrics[f"{q_type}_partial_recall@{k}"] = round(c["partial"] / c["total"], 4)
        metrics[f"{q_type}_hop2_hit@{k}"]       = round(c["hop2"]    / c["total"], 4)
        metrics[f"{q_type}_count"]              = c["total"]

    return metrics


if __name__ == "__main__":
    print("=" * 55)
    print("TWO-HOP GRAPH RETRIEVAL EVALUATION")
    print("=" * 55)

    two_hop = evaluate_graph_retrieval(
        split="train", max_samples=10000, k=5,
        seed_k=10, hop2_weight=0.4, use_two_hop=True,
    )

    print(f"\nOverall ({two_hop['total_questions']} questions)")
    print(f"  support_recall@5:         {two_hop['support_recall@5']}")
    print(f"  partial_support_recall@5: {two_hop['partial_support_recall@5']}")
    print(f"  hop1_contributed@5:       {two_hop['hop1_contributed@5']}")
    print(f"  hop2_contributed@5:       {two_hop['hop2_contributed@5']}")

    print(f"\nBridge questions ({two_hop.get('bridge_count', 0)})")
    print(f"  support_recall@5: {two_hop.get('bridge_support_recall@5', 'N/A')}")
    print(f"  hop2_hit@5:       {two_hop.get('bridge_hop2_hit@5', 'N/A')}")

    print(f"\nComparison questions ({two_hop.get('comparison_count', 0)})")
    print(f"  support_recall@5: {two_hop.get('comparison_support_recall@5', 'N/A')}")
    print(f"  hop2_hit@5:       {two_hop.get('comparison_hop2_hit@5', 'N/A')}")

    print("\n" + "=" * 55)
    print("SINGLE-HOP ABLATION")
    print("=" * 55)

    single = evaluate_graph_retrieval(
        split="train", max_samples=10000, k=5,
        seed_k=10, use_two_hop=False,
    )

    print(f"\n  support_recall@5:         {single['support_recall@5']}")
    print(f"  partial_support_recall@5: {single['partial_support_recall@5']}")

    print("\nDelta (two-hop - single-hop):")
    for m in ["support_recall@5", "partial_support_recall@5"]:
        delta = two_hop[m] - single[m]
        print(f"  {m}: {delta:+.4f}")