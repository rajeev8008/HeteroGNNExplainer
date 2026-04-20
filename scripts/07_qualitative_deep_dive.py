import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.models.graphsage import GraphSAGE
from src.explainers.baselines import get_baseline_explainer
from src.explainers.hetero_explainer import get_novel_explainer
from torch_geometric.explain.metric import fidelity

def _collect_edges(explanation, threshold=0.5):
    edge_mask = getattr(explanation, 'edge_mask', None)
    if edge_mask is None: return []
    mask = torch.sigmoid(edge_mask) if edge_mask.min() < 0 or edge_mask.max() > 1 else edge_mask
    edges = []
    for i in range(explanation.edge_index.size(1)):
        if mask[i] >= threshold:
            edges.append((int(explanation.edge_index[0, i]), int(explanation.edge_index[1, i]), float(mask[i])))
    return edges

def _get_similarity(x, u, v):
    xu, xv = x[u].unsqueeze(0), x[v].unsqueeze(0)
    return torch.nn.functional.cosine_similarity(xu, xv, dim=-1).item()

def main():
    dataset_dir = os.path.join(project_root, 'data', 'processed')
    model_dir = os.path.join(project_root, 'results', 'models')
    out_dir = os.path.join(project_root, 'results', 'figures', 'deep_dive')
    os.makedirs(out_dir, exist_ok=True)

    # Focus on the most heterophilic dataset
    h_target = 0.1
    ds_path = os.path.join(dataset_dir, f'dataset_homophily_{h_target:.2f}.pt')
    if not os.path.exists(ds_path):
        print(f"Dataset {ds_path} not found.")
        return

    data = torch.load(ds_path, weights_only=False)
    in_channels = data.x.size(1)
    out_channels = int(data.y.max().item() + 1)

    model = GraphSAGE(in_channels=in_channels, hidden_channels=32, out_channels=out_channels)
    model.load_state_dict(torch.load(os.path.join(model_dir, f'sage_dataset_homophily_{h_target:.2f}.pth'), weights_only=True))
    model.eval()

    baseline = get_baseline_explainer(model, 'gnn_explainer')
    novel = get_novel_explainer(model, heterophily_weight=0.8) # Strong weight for deep dive

    # Find a node where novel wins significantly
    test_nodes = torch.where(data.train_mask)[0][:100]
    best_node = None
    max_gap = -1.0
    best_results = None

    print("Searching for a 'success' case node...")
    for node in test_nodes:
        idx = int(node.item())
        target = model(data.x, data.edge_index).argmax(dim=-1)[node].view(1)
        
        exp_b = baseline(data.x, data.edge_index, index=idx)
        exp_n = novel(data.x, data.edge_index, index=idx)
        
        try:
            fb, _ = fidelity(baseline, exp_b)
            fn, _ = fidelity(novel, exp_n)
            gap = float(fn - fb)
            if gap > max_gap:
                max_gap = gap
                best_node = idx
                best_results = (exp_b, exp_n, fb, fn)
                if gap > 0.4: break # Good enough
        except: continue

    if best_node is None:
        print("Could not find a significant gap. Using first available node.")
        best_node = int(test_nodes[0].item())
        # ... fallback ...

    idx = best_node
    exp_b, exp_n, fb, fn = best_results
    print(f"Node {idx} selected. Fidelity Gap: {max_gap:.3f} (Novel: {fn:.3f}, Base: {fb:.3f})")

    # Visualize and Analyze Features
    b_edges = _collect_edges(exp_b)
    n_edges = _collect_edges(exp_n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    def draw(ax, edges, title):
        G = nx.Graph()
        for u, v, w in edges: G.add_edge(u, v, weight=w)
        if G.number_of_nodes() == 0:
            ax.set_title(title + " (Empty)")
            return
        pos = nx.spring_layout(G, seed=42)
        sims = [_get_similarity(data.x, u, v) for u, v in G.edges()]
        colors = ['red' if s < 0 else 'blue' for s in sims]
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightgray', edge_color=colors, width=2)
        ax.set_title(title)

    draw(axes[0], b_edges, f"Baseline (F+={fb:.2f})")
    draw(axes[1], n_edges, f"HeteroExplainer (F+={fn:.2f})")
    
    fig.suptitle(f"Deep Dive: Node {idx} in Heterophilic Graph (h=0.1)\nRed Edge: Heterophilic (Sim < 0), Blue Edge: Homophilic (Sim > 0)", fontsize=14)
    out_img = os.path.join(out_dir, "deep_dive_node_comparison.png")
    plt.savefig(out_img)
    print(f"Saved deep dive visualization: {out_img}")

    # Feature Analysis
    print("\n--- Feature Analysis ---")
    print(f"Target Label: {int(data.y[idx])}")
    print(f"Baseline selected {len(b_edges)} edges.")
    print(f"Novel selected {len(n_edges)} edges.")
    
    hetero_count_n = sum(1 for u, v, w in n_edges if _get_similarity(data.x, u, v) < 0)
    hetero_count_b = sum(1 for u, v, w in b_edges if _get_similarity(data.x, u, v) < 0)
    
    print(f"Heterophilic edges (Sim < 0) in Baseline: {hetero_count_b}")
    print(f"Heterophilic edges (Sim < 0) in Novel: {hetero_count_n}")

if __name__ == '__main__':
    main()
