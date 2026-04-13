import os
import sys
import glob
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.models.graphsage import GraphSAGE
from src.explainers.baselines import get_baseline_explainer
from src.explainers.hetero_explainer import get_novel_explainer


def _safe_sigmoid(mask):
    if mask.min().item() < 0.0 or mask.max().item() > 1.0:
        return torch.sigmoid(mask)
    return mask


def _collect_explainer_subgraph(data, explanation, threshold=0.5):
    edge_mask = getattr(explanation, 'edge_mask', None)
    if edge_mask is None:
        return [], set()

    mask = _safe_sigmoid(edge_mask)
    edge_index = data.edge_index
    selected_edges = []
    selected_nodes = set()

    for i in range(edge_index.size(1)):
        if float(mask[i]) >= threshold:
            u = int(edge_index[0, i])
            v = int(edge_index[1, i])
            selected_edges.append((u, v, float(mask[i])))
            selected_nodes.add(u)
            selected_nodes.add(v)

    return selected_edges, selected_nodes


def _feature_dissimilarity(x, u, v):
    xu = x[u]
    xv = x[v]
    sim = torch.nn.functional.cosine_similarity(xu.unsqueeze(0), xv.unsqueeze(0), dim=-1).item()
    return 1.0 - sim


def _draw_subgraph(ax, data, selected_edges, title, highlight_dissimilar=False, dissim_threshold=1.2):
    g = nx.Graph()
    for u, v, w in selected_edges:
        g.add_edge(u, v, weight=w)

    if g.number_of_nodes() == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No edges selected', ha='center', va='center')
        ax.axis('off')
        return

    pos = nx.spring_layout(g, seed=42)
    widths = [1.0 + 2.0 * g[u][v]['weight'] for u, v in g.edges()]

    edge_colors = []
    for u, v in g.edges():
        if highlight_dissimilar:
            dissim = _feature_dissimilarity(data.x, u, v)
            edge_colors.append('crimson' if dissim >= dissim_threshold else 'steelblue')
        else:
            edge_colors.append('steelblue')

    nx.draw_networkx_nodes(g, pos=pos, ax=ax, node_size=220, node_color='lightgray', edgecolors='black', linewidths=0.4)
    nx.draw_networkx_edges(g, pos=pos, ax=ax, width=widths, edge_color=edge_colors, alpha=0.9)
    nx.draw_networkx_labels(g, pos=pos, ax=ax, font_size=7)

    ax.set_title(title)
    ax.axis('off')


def main():
    plt.rcParams.update(
        {
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'cm',
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'legend.fontsize': 10,
        }
    )

    dataset_dir = os.path.join(project_root, 'data', 'processed')
    model_dir = os.path.join(project_root, 'results', 'models')
    out_dir = os.path.join(project_root, 'results', 'figures', 'qualitative')
    os.makedirs(out_dir, exist_ok=True)

    dataset_files = sorted(glob.glob(os.path.join(dataset_dir, 'dataset_homophily_*.pt')))
    if not dataset_files:
        print('No datasets found! Run scripts/01_generate_datasets.py first.')
        return

    print('=' * 60)
    print('PHASE 6: Side-by-Side Explainer Visualization')
    print('=' * 60)

    # Keep one representative node per dataset for publication-style qualitative comparison.
    for ds_path in dataset_files:
        filename = os.path.basename(ds_path)
        base_name = filename.replace('.pt', '')
        h_val = float(base_name.split('_')[-1])

        data = torch.load(ds_path, weights_only=False)
        in_channels = data.x.size(1) if hasattr(data, 'x') and data.x is not None else 10
        out_channels = int(data.y.max().item() + 1) if hasattr(data, 'y') and data.y is not None else 2

        model = GraphSAGE(in_channels=in_channels, hidden_channels=32, out_channels=out_channels)
        model_path = os.path.join(model_dir, f'sage_{base_name}.pth')

        if not os.path.exists(model_path):
            print(f'[WARN] Missing GraphSAGE weights for {base_name}, skipping visualization.')
            continue

        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        baseline = get_baseline_explainer(model, 'gnn_explainer')
        hetero = get_novel_explainer(model, heterophily_weight=0.5)

        if hasattr(data, 'train_mask') and data.train_mask is not None and data.train_mask.sum() > 0:
            node_idx = int(torch.where(data.train_mask)[0][0].item())
        else:
            node_idx = 0

        exp_base = baseline(data.x, data.edge_index, index=node_idx)
        exp_hetero = hetero(data.x, data.edge_index, index=node_idx)

        base_edges, _ = _collect_explainer_subgraph(data, exp_base, threshold=0.5)
        hetero_edges, _ = _collect_explainer_subgraph(data, exp_hetero, threshold=0.5)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        _draw_subgraph(
            axes[0],
            data,
            base_edges,
            title=f'Standard GNNExplainer (h={h_val:.2f})',
            highlight_dissimilar=False,
        )
        _draw_subgraph(
            axes[1],
            data,
            hetero_edges,
            title=f'HeteroGNNExplainer (h={h_val:.2f})',
            highlight_dissimilar=True,
            dissim_threshold=1.2,
        )

        fig.suptitle('Structural vs Feature-Distance Emphasis in Explanations', fontsize=13)
        fig.tight_layout()

        out_path = os.path.join(out_dir, f'explainer_side_by_side_h{h_val:.2f}.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'[SUCCESS] Saved qualitative comparison: {out_path}')

    print('Visualization complete.')


if __name__ == '__main__':
    main()
