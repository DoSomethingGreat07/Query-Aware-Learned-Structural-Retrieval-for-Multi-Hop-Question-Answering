import re
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from hybrid_graph_builder import build_hybrid_chunk_graph


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.normalize(x, p=2, dim=1)


def build_pyg_data(graph, embeddings):
    x = torch.tensor(embeddings, dtype=torch.float)
    BRIDGE_TYPES = {"hyperlink", "keyword_overlap", "adjacent_chunk", "same_title"}
    edge_list = []
    for u, v, data in graph.edges(data=True):
        if data.get("edge_types", set()) & BRIDGE_TYPES:
            edge_list.append([u, v])
            edge_list.append([v, u])
    if not edge_list:
        for u, v in graph.edges():
            edge_list.append([u, v])
            edge_list.append([v, u])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def extract_keywords(text, min_len=4):
    stopwords = {
        "This", "That", "With", "From", "Have", "Were", "Which", "Their",
        "About", "After", "Before", "Into", "During", "Than", "Then", "Them",
        "They", "Been", "Being", "Also", "Some", "Such", "Most", "More",
        "Only", "Other", "Many", "First", "Second", "Third", "Would", "Could",
        "Should", "Where", "When", "What", "Who", "Whose", "While", "Because",
    }
    tokens = re.findall(r"[A-Z][A-Za-z0-9\-]{3,}", text)
    return {t for t in tokens if t not in stopwords}


def mine_training_pairs(chunks, graph, embeddings, max_pairs=10000):
    BRIDGE_TYPES = {"hyperlink", "keyword_overlap"}
    bridge_pairs = [
        (u, v) for u, v, data in graph.edges(data=True)
        if data.get("edge_types", set()) & BRIDGE_TYPES
    ]
    if not bridge_pairs:
        bridge_pairs = [
            (u, v) for u, v, data in graph.edges(data=True)
            if chunks[u].metadata.get("title") != chunks[v].metadata.get("title")
        ]

    num_nodes     = len(chunks)
    neighbor_sets = {n: set(graph.neighbors(n)) for n in graph.nodes()}
    triplets      = []
    random.shuffle(bridge_pairs)

    for anchor, pos in bridge_pairs[:max_pairs]:
        anchor_title = chunks[anchor].metadata.get("title", "")
        neg = None
        for _ in range(20):
            neg_cand  = random.randint(0, num_nodes - 1)
            neg_title = chunks[neg_cand].metadata.get("title", "")
            if (neg_cand != anchor and neg_cand != pos
                    and neg_cand not in neighbor_sets.get(anchor, set())
                    and neg_title != anchor_title):
                neg = neg_cand
                break
        if neg is not None:
            triplets.append((anchor, pos, neg))
    return triplets


def mine_hop2_pairs(chunks, graph, embeddings, max_pairs=3000):
    BRIDGE_TYPES  = {"hyperlink", "keyword_overlap"}
    num_nodes     = len(chunks)
    neighbor_sets = {n: set(graph.neighbors(n)) for n in graph.nodes()}
    hop2_triplets = []

    for u, v, data in list(graph.edges(data=True))[:max_pairs * 2]:
        if not (data.get("edge_types", set()) & BRIDGE_TYPES):
            continue
        v_keywords = extract_keywords(chunks[v].page_content)
        if not v_keywords:
            continue
        best_c, best_overlap = None, 0
        for c in random.sample(range(num_nodes), min(50, num_nodes)):
            if c in {u, v} or c in neighbor_sets.get(u, set()):
                continue
            c_keywords = extract_keywords(chunks[c].page_content)
            overlap = len(c_keywords & v_keywords)
            if overlap > best_overlap:
                best_overlap = overlap
                best_c = c
        if best_c is None or best_overlap < 2:
            continue
        neg = random.randint(0, num_nodes - 1)
        while neg in {u, v, best_c}:
            neg = random.randint(0, num_nodes - 1)
        hop2_triplets.append(([u, v], best_c, neg))
    return hop2_triplets[:max_pairs]


def contrastive_retrieval_loss(z, triplets, temperature=0.07):
    if not triplets:
        return torch.tensor(0.0, requires_grad=True)
    anchors   = torch.stack([z[a] for a, p, n in triplets])
    positives = torch.stack([z[p] for a, p, n in triplets])
    negatives = torch.stack([z[n] for a, p, n in triplets])
    pos_sim   = F.cosine_similarity(anchors, positives) / temperature
    neg_sim   = F.cosine_similarity(anchors, negatives) / temperature
    logits    = torch.stack([pos_sim, neg_sim], dim=1)
    labels    = torch.zeros(len(triplets), dtype=torch.long)
    return F.cross_entropy(logits, labels)


def hop2_contrastive_loss(z, hop2_triplets, temperature=0.07):
    if not hop2_triplets:
        return torch.tensor(0.0, requires_grad=True)
    contexts, positives_z, negatives_z = [], [], []
    for context_ids, pos_id, neg_id in hop2_triplets:
        contexts.append(z[context_ids].mean(dim=0))
        positives_z.append(z[pos_id])
        negatives_z.append(z[neg_id])
    ctx_t = torch.stack(contexts)
    pos_t = torch.stack(positives_z)
    neg_t = torch.stack(negatives_z)
    pos_sim = F.cosine_similarity(ctx_t, pos_t) / temperature
    neg_sim = F.cosine_similarity(ctx_t, neg_t) / temperature
    logits  = torch.stack([pos_sim, neg_sim], dim=1)
    labels  = torch.zeros(len(hop2_triplets), dtype=torch.long)
    return F.cross_entropy(logits, labels)


def train_graphsage(
    split="train",
    max_samples=10000,
    chunk_size=500,
    chunk_overlap=100,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    hidden_channels=128,
    out_channels=128,
    lr=1e-3,
    epochs=30,               # fewer epochs — more data converges faster
    temperature=0.07,
    hop2_loss_weight=0.4,
    max_triplets=10000,      # scaled up from 2000
    max_hop2_pairs=3000,     # scaled up from 500
    seed=42,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    graph, chunks, embeddings = build_hybrid_chunk_graph(
        split=split, max_samples=max_samples,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        model_name=model_name,
        semantic_top_k=3, min_similarity=0.60,
        add_same_title_edges=True, add_adjacent_chunk_edges=True,
        add_hyperlink_edges_flag=True, add_keyword_overlap_edges=True,
        min_keyword_overlap=3, max_nodes_for_semantic_knn=10000,
    )

    data = build_pyg_data(graph, embeddings)
    print(f"Graph: {data.num_nodes} nodes, {data.edge_index.shape[1]//2} edges (bridge only)")

    print("Mining hop-1 contrastive pairs...")
    triplets = mine_training_pairs(chunks, graph, embeddings, max_pairs=max_triplets)
    print(f"  → {len(triplets)} triplets")

    print("Mining hop-2 supervision pairs...")
    hop2_triplets = mine_hop2_pairs(chunks, graph, embeddings, max_pairs=max_hop2_pairs)
    print(f"  → {len(hop2_triplets)} hop-2 triplets")

    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        z          = model(data.x, data.edge_index)
        loss_hop1  = contrastive_retrieval_loss(z, triplets, temperature=temperature)
        loss_hop2  = hop2_contrastive_loss(z, hop2_triplets, temperature=temperature)
        loss       = loss_hop1 + hop2_loss_weight * loss_hop2
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} "
                  f"| Hop1: {loss_hop1.item():.4f} | Hop2: {loss_hop2.item():.4f}")

    model.eval()
    with torch.no_grad():
        gnn_embeddings = model(data.x, data.edge_index).cpu().numpy()

    print(f"\nTraining complete.")
    print(f"Original embedding shape: {embeddings.shape}")
    print(f"GNN embedding shape:      {gnn_embeddings.shape}")
    return model, graph, chunks, embeddings, gnn_embeddings


if __name__ == "__main__":
    train_graphsage(split="train", max_samples=10000, epochs=30)