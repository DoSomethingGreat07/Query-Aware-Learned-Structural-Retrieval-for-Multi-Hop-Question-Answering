"""
llm_eval.py — End-to-end RAG evaluation for HotpotQA using GPT-4o-mini.
Optimized for: multi-hop reasoning, concise answers, graceful fallback.
"""

import re
import os
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

from hybrid_graph_builder import build_hybrid_chunk_graph
from graph_retrieval import graph_retrieve_two_hop

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"
client    = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── LLM Call ──────────────────────────────────────────────────────────────────

def call_gpt4(system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=150,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise question-answering assistant specializing in multi-hop factual questions.

Your job:
- Read the provided sources carefully
- Connect facts across multiple sources when needed
- Give a short, direct factual answer

Rules:
- Answer using ONLY the information in the provided sources
- Never use outside knowledge or make assumptions
- Even if context is incomplete, give your best answer based on available evidence
- Only say "Insufficient context" if the sources contain absolutely NO relevant information
- Keep your answer as short as possible: a name, date, number, place, or yes/no
- Do not explain unless it is a yes/no question"""


def build_user_prompt(question: str, results: list) -> str:
    source_blocks = []
    for i, r in enumerate(results, 1):
        title = r["metadata"].get("title", "Unknown")
        text  = r["text"].strip()
        hop   = r.get("hop", 1)
        label = "direct evidence" if hop == 1 else "bridging evidence"
        source_blocks.append(f"Source {i} [{title}] ({label}):\n{text}")

    sources = "\n\n".join(source_blocks)

    return f"""Sources:
{sources}

Question: {question}

Instructions:
- Comparison question (e.g. "which came first", "who is older"): compare facts from sources directly.
- Bridge question (e.g. requires two steps): find the connecting fact across sources.
- Answer as concisely as possible. A name, date, number, place, or yes/no.
- Use best available evidence even if context is partial.
- Only say "Insufficient context" if sources have NO relevant information at all.

Answer:"""


# ── Answer Extraction + Matching ─────────────────────────────────────────────

def extract_answer(llm_response: str) -> str:
    lines = [l.strip() for l in llm_response.strip().splitlines() if l.strip()]
    if not lines:
        return llm_response.strip()
    return re.sub(r"^answer[:\s]+", "", lines[0], flags=re.IGNORECASE).strip()


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answers_match(predicted: str, gold: str) -> bool:
    pred   = normalize(predicted)
    gold_n = normalize(gold)

    if "insufficient context" in pred:
        return False
    if pred == gold_n:
        return True
    if gold_n in pred or pred in gold_n:
        return True

    # Token overlap — handles "Walter Coy" vs "Walter Darwin Coy"
    # and "U.S. Highway 60" vs "US 60"
    pred_tokens = set(pred.split())
    gold_tokens = set(gold_n.split())
    if gold_tokens and len(pred_tokens & gold_tokens) / len(gold_tokens) >= 0.8:
        return True

    # Yes/No aliases
    yes_aliases = {"yes", "true", "correct", "indeed"}
    no_aliases  = {"no", "false", "incorrect", "neither"}
    if gold_n in yes_aliases and pred in yes_aliases: return True
    if gold_n in no_aliases  and pred in no_aliases:  return True

    return False


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_end_to_end(
    split="train",
    graph_max_samples=10000,
    eval_max_samples=200,      # increased from 50
    seed_k=10,
    final_k=5,
    verbose=True,
    save_results=True,
):
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    dataset = dataset.select(range(min(eval_max_samples, len(dataset))))

    G, chunks, embeddings = build_hybrid_chunk_graph(
        split=split,
        max_samples=graph_max_samples,
        chunk_size=500,          # increased from 300
        chunk_overlap=100,       # increased from 50
        semantic_top_k=3,
        min_similarity=0.60,
        add_same_title_edges=True,
        add_adjacent_chunk_edges=True,
        add_hyperlink_edges_flag=True,
        add_keyword_overlap_edges=True,
        min_keyword_overlap=3,
        max_nodes_for_semantic_knn=10000,
    )

    retrieval_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    exact_hits = insufficient = 0
    type_counters = {
        "bridge":     {"correct": 0, "total": 0},
        "comparison": {"correct": 0, "total": 0},
        "unknown":    {"correct": 0, "total": 0},
    }
    all_results = []

    print(f"\nEvaluating {eval_max_samples} questions with {LLM_MODEL}...\n")
    print("=" * 60)

    for idx, item in enumerate(dataset):
        question    = item["question"]
        gold_answer = item["answer"].strip()
        q_type      = item.get("type", "unknown")
        gold_titles = set(item["supporting_facts"]["title"])

        retrieved = graph_retrieve_two_hop(
            query=question, chunks=chunks, embeddings=embeddings,
            graph=G, model=retrieval_model, seed_k=seed_k, final_k=final_k,
        )

        retrieved_titles = {r["metadata"]["title"] for r in retrieved}
        gold_retrieved   = gold_titles.issubset(retrieved_titles)

        user_prompt = build_user_prompt(question, retrieved)
        response    = call_gpt4(SYSTEM_PROMPT, user_prompt)
        predicted   = extract_answer(response)

        is_correct = answers_match(predicted, gold_answer)
        is_insuff  = "insufficient context" in predicted.lower()

        if is_correct: exact_hits  += 1
        if is_insuff:  insufficient += 1

        bucket = q_type if q_type in type_counters else "unknown"
        type_counters[bucket]["correct"] += int(is_correct)
        type_counters[bucket]["total"]   += 1

        all_results.append({
            "idx": idx, "question": question,
            "gold_answer": gold_answer, "predicted": predicted,
            "correct": is_correct, "q_type": q_type,
            "gold_retrieved": gold_retrieved,
            "retrieved_titles": list(retrieved_titles),
            "raw_response": response,
        })

        if verbose:
            icon = "✅" if is_correct else "❌"
            print(f"[{idx+1:03d}] {icon} {q_type.upper():<12} "
                  f"Gold: {gold_answer:<25} Pred: {predicted}")
            if not is_correct:
                print(f"       Gold retrieved: {gold_retrieved} | "
                      f"Titles: {list(retrieved_titles)[:3]}")

    total = len(dataset)
    correct_with_gold    = sum(1 for r in all_results if r["correct"] and r["gold_retrieved"])
    correct_without_gold = sum(1 for r in all_results if r["correct"] and not r["gold_retrieved"])

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOverall accuracy:     {exact_hits}/{total} = {exact_hits/total:.3f}")
    print(f"Insufficient context: {insufficient}/{total} = {insufficient/total:.3f}")

    for q_type, c in type_counters.items():
        if c["total"] == 0: continue
        acc = c["correct"] / c["total"]
        print(f"\n{q_type.capitalize()} ({c['total']}): {c['correct']}/{c['total']} = {acc:.3f}")

    print(f"\nCorrect WITH gold docs:    {correct_with_gold}")
    print(f"Correct WITHOUT gold docs: {correct_without_gold}")

    if save_results:
        with open("llm_eval_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\nSaved → llm_eval_results.json")

    metrics = {
        "total":                total,
        "answer_accuracy":      round(exact_hits / total, 4),
        "insufficient_context": round(insufficient / total, 4),
        "bridge_accuracy":      round(type_counters["bridge"]["correct"] /
                                      max(type_counters["bridge"]["total"], 1), 4),
        "comparison_accuracy":  round(type_counters["comparison"]["correct"] /
                                      max(type_counters["comparison"]["total"], 1), 4),
    }

    print("\n" + "=" * 60)
    acc = metrics["answer_accuracy"]
    if acc >= 0.65:
        print(f"✅ Accuracy {acc:.3f} — excellent, 10k scale confirmed")
    elif acc >= 0.55:
        print(f"✅ Accuracy {acc:.3f} — good, proceed with 10k")
    elif acc >= 0.40:
        print(f"⚠️  Accuracy {acc:.3f} — inspect llm_eval_results.json")
    else:
        print(f"❌ Accuracy {acc:.3f} — fix retrieval or prompt first")

    return metrics


if __name__ == "__main__":
    metrics = evaluate_end_to_end(
        split="train",
        graph_max_samples=10000,
        eval_max_samples=200,
        seed_k=10,
        final_k=5,
        verbose=True,
        save_results=True,
    )