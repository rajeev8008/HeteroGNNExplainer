import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from torch_geometric.explain import fidelity

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.models.graphsage import GraphSAGE
from src.explainers.baselines import get_baseline_explainer
from src.explainers.hetero_explainer import get_novel_explainer

def main():
    out_dir = os.path.join(project_root, 'results', 'figures', 'failure_analysis')
    os.makedirs(out_dir, exist_ok=True)
    
    # Use h=0.5 where it's most ambiguous
    ds_path = os.path.join(project_root, 'data', 'processed', 'dataset_homophily_0.50.pt')
    if not os.path.exists(ds_path): ds_path = os.path.join(project_root, 'data', 'processed', 'dataset_homophily_0.10.pt')
    
    data = torch.load(ds_path, weights_only=False)
    in_channels = data.x.size(1)
    out_channels = int(data.y.max().item() + 1)
    
    model = GraphSAGE(in_channels=in_channels, hidden_channels=32, out_channels=out_channels)
    model_path = os.path.join(project_root, 'results', 'models', f'sage_{os.path.basename(ds_path).replace(".pt", "")}.pth')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    baseline = get_baseline_explainer(model, 'gnn_explainer')
    novel = get_novel_explainer(model, heterophily_weight=0.1)
    
    test_nodes = torch.where(data.train_mask)[0][:100]
    
    worst_node = None
    min_gap = 1.0 # Looking for Novel < Baseline
    worst_exps = None
    
    all_mask_vectors = []
    
    print("Searching for 'Confused Nodes' and collecting embeddings...")
    for node in test_nodes:
        idx = int(node.item())
        exp_n = novel(data.x, data.edge_index, index=idx)
        exp_b = baseline(data.x, data.edge_index, index=idx)
        
        try:
            fn, _ = fidelity(novel, exp_n)
            fb, _ = fidelity(baseline, exp_b)
            gap = float(fn - fb)
            if gap < min_gap:
                min_gap = gap
                worst_node = idx
                worst_exps = (exp_b, exp_n, fb, fn)
        except: continue
        
        # Collect mask for t-SNE (padding to 20 neighbors)
        mask = torch.sigmoid(exp_n.edge_mask)
        padded = torch.zeros(20)
        n = min(20, mask.size(0))
        padded[:n] = mask[:n]
        all_mask_vectors.append(padded.detach().cpu().numpy())

    # --- 1. Confused Node Visualization ---
    if worst_node is not None:
        idx = worst_node
        exp_b, exp_n, fb, fn = worst_exps
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        def draw(ax, exp, fid, title):
            G = nx.Graph()
            mask = torch.sigmoid(exp.edge_mask)
            for i in range(exp.edge_index.size(1)):
                if mask[i] > 0.5:
                    G.add_edge(int(exp.edge_index[0, i]), int(exp.edge_index[1, i]))
            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightcoral', node_size=300)
            ax.set_title(f"{title}\nFidelity+: {fid:.3f}")
        
        draw(axes[0], exp_b, fb, "Baseline (Wins)")
        draw(axes[1], exp_n, fn, "Novel (Confused)")
        fig.suptitle(f"Failure Case: Node {idx}\n(Novel performs worse than baseline)", fontsize=14)
        plt.savefig(os.path.join(out_dir, 'confused_node_deep_dive.png'))
        plt.close()

    # --- 2. t-SNE of Explanations ---
    if len(all_mask_vectors) > 10:
        embeddings = TSNE(n_components=2, perplexity=min(30, len(all_mask_vectors)-1), random_state=42).fit_transform(np.array(all_mask_vectors))
        plt.figure(figsize=(8, 8))
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c='teal', alpha=0.6)
        plt.title('t-SNE Projection of Explanation Edge-Masks')
        plt.xlabel('TSNE-1')
        plt.ylabel('TSNE-2')
        plt.savefig(os.path.join(out_dir, 'tsne_explanations.png'))
        plt.close()

    print(f"[SUCCESS] Failure analysis saved to {out_dir}")

if __name__ == '__main__':
    main()
