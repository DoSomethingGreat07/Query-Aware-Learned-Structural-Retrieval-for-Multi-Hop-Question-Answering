import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from embeddings import generate_chunk_embeddings


def evaluate_baseline_retrieval(
    split="train",
    max_samples=10000,
    k=5,
):
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    chunks, embeddings, _ = generate_chunk_embeddings(
        split=split,
        max_samples=max_samples,
        chunk_size=500,
        chunk_overlap=100,
    )

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    exact_hits   = 0
    partial_hits = 0

    type_counters = {
        "bridge":     {"exact": 0, "partial": 0, "total": 0},
        "comparison": {"exact": 0, "partial": 0, "total": 0},
        "unknown":    {"exact": 0, "partial": 0, "total": 0},
    }

    for item in dataset:
        question    = item["question"]
        gold_titles = set(item["supporting_facts"]["title"])
        q_type      = item.get("type", "unknown")

        query_vec = model.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        )[0]

        scores  = embeddings @ query_vec
        top_idx = np.argsort(scores)[-k:][::-1]

        retrieved_titles = {chunks[i].metadata["title"] for i in top_idx}
        overlap = gold_titles & retrieved_titles

        is_exact   = int(len(overlap) == len(gold_titles))
        is_partial = int(len(overlap) > 0)

        exact_hits   += is_exact
        partial_hits += is_partial

        bucket = q_type if q_type in type_counters else "unknown"
        type_counters[bucket]["exact"]   += is_exact
        type_counters[bucket]["partial"] += is_partial
        type_counters[bucket]["total"]   += 1

    total   = len(dataset)
    metrics = {
        "total_questions":              total,
        f"support_recall@{k}":         round(exact_hits / total, 4),
        f"partial_support_recall@{k}": round(partial_hits / total, 4),
    }

    for q_type, c in type_counters.items():
        if c["total"] == 0:
            continue
        metrics[f"{q_type}_support_recall@{k}"] = round(c["exact"]   / c["total"], 4)
        metrics[f"{q_type}_partial_recall@{k}"] = round(c["partial"] / c["total"], 4)
        metrics[f"{q_type}_count"]              = c["total"]

    return metrics


if __name__ == "__main__":
    print("=" * 55)
    print("BASELINE RETRIEVAL EVALUATION")
    print("=" * 55)

    metrics = evaluate_baseline_retrieval(split="train", max_samples=10000, k=5)

    print(f"\nOverall ({metrics['total_questions']} questions)")
    print(f"  support_recall@5:         {metrics['support_recall@5']}")
    print(f"  partial_support_recall@5: {metrics['partial_support_recall@5']}")

    print(f"\nBridge questions ({metrics.get('bridge_count', 0)})")
    print(f"  support_recall@5: {metrics.get('bridge_support_recall@5', 'N/A')}")
    print(f"  partial_recall@5: {metrics.get('bridge_partial_recall@5', 'N/A')}")

    print(f"\nComparison questions ({metrics.get('comparison_count', 0)})")
    print(f"  support_recall@5: {metrics.get('comparison_support_recall@5', 'N/A')}")
    print(f"  partial_recall@5: {metrics.get('comparison_partial_recall@5', 'N/A')}")