import os
import sys
import glob
import torch
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.models.graphsage import GraphSAGE
from src.explainers.baselines import get_baseline_explainer
from src.explainers.hetero_explainer import get_novel_explainer, set_explainer_heterophily_weight
from torch_geometric.explain.metric import fidelity


def _to_scalar(v):
    return float(v.item()) if hasattr(v, 'item') else float(v)


def evaluate_dataset(data, model, baseline_explainer, novel_explainer, sample_size=50):
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        candidate_nodes = torch.where(data.train_mask)[0]
    else:
        candidate_nodes = torch.arange(data.num_nodes)

    if candidate_nodes.numel() == 0:
        return 0.0, 0.0

    n = min(sample_size, int(candidate_nodes.numel()))
    chosen = candidate_nodes[torch.randperm(candidate_nodes.numel())[:n]]

    base_scores = []
    novel_scores = []

    for node in chosen:
        idx = int(node.item())
        exp_base = baseline_explainer(data.x, data.edge_index, index=idx)
        exp_novel = novel_explainer(data.x, data.edge_index, index=idx)

        try:
            fid_base_plus, _ = fidelity(baseline_explainer, exp_base)
            base_scores.append(_to_scalar(fid_base_plus))
        except Exception:
            base_scores.append(0.0)

        try:
            fid_novel_plus, _ = fidelity(novel_explainer, exp_novel)
            novel_scores.append(_to_scalar(fid_novel_plus))
        except Exception:
            novel_scores.append(0.0)

    mean_base = float(np.mean(base_scores)) if base_scores else 0.0
    mean_novel = float(np.mean(novel_scores)) if novel_scores else 0.0
    return mean_base, mean_novel


def main():
    dataset_dir = os.path.join(project_root, 'data', 'processed')
    model_dir = os.path.join(project_root, 'results', 'models')
    out_dir = os.path.join(project_root, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    dataset_files = sorted(glob.glob(os.path.join(dataset_dir, 'dataset_homophily_*.pt')))
    if not dataset_files:
        print('No datasets found! Run scripts/01_generate_datasets.py first.')
        return

    sweep_weights = np.round(np.linspace(0.0, 1.0, 11), 2)

    all_results = []
    print('=' * 60)
    print('PHASE 5: Heterophily Weight Sweep (0.0 -> 1.0)')
    print('=' * 60)

    for weight in sweep_weights:
        print(f'\n--- Evaluating heterophily_weight={weight:.2f} ---')

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
                continue

            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()

            baseline_explainer = get_baseline_explainer(model, 'gnn_explainer')
            novel_explainer = get_novel_explainer(model, heterophily_weight=weight)
            set_explainer_heterophily_weight(novel_explainer, float(weight))

            base_mean, novel_mean = evaluate_dataset(
                data=data,
                model=model,
                baseline_explainer=baseline_explainer,
                novel_explainer=novel_explainer,
                sample_size=5, # Reduced size for faster sweep
            )
            gap = novel_mean - base_mean
            all_results.append({
                'homophily': h_val,
                'weight': weight,
                'gap': gap,
                'dataset': filename
            })
            print(f'  h={h_val:.2f} | Gap={gap:.4f}')

    if not all_results:
        print('No valid evaluations were completed.')
        return

    df_all = pd.DataFrame(all_results)
    
    # Find best weight per homophily level
    best_per_h = []
    for h, group in df_all.groupby('homophily'):
        best_row = group.loc[group['gap'].idxmax()]
        best_per_h.append({
            'homophily': h,
            'best_weight': best_row['weight'],
            'max_gap': best_row['gap']
        })
    
    df_best = pd.DataFrame(best_per_h)
    best_weights_csv = os.path.join(out_dir, 'optimal_heterophily_weights.csv')
    df_best.to_csv(best_weights_csv, index=False)
    print(f"\n[SUCCESS] Optimal weights per homophily saved to {best_weights_csv}")

    # Also save aggregate sweep for visualization
    df_agg = df_all.groupby('weight')['gap'].agg(['mean', 'std']).reset_index()
    df_agg.columns = ['heterophily_weight', 'mean_fidelity_plus_gap', 'std_fidelity_plus_gap']
    out_csv = os.path.join(out_dir, 'heterophily_weight_sweep.csv')
    df_agg.to_csv(out_csv, index=False)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(df_agg['heterophily_weight'], df_agg['mean_fidelity_plus_gap'], marker='o', linestyle='-', color='dodgerblue')
    plt.xlabel('Heterophily Weight')
    plt.ylabel('Mean Fidelity+ Gap (Novel - Baseline)')
    plt.title('Effect of Heterophily Weight on Explainer Performance')
    plt.grid(True)
    out_plot = os.path.join(out_dir, 'heterophily_weight_sweep.png')
    plt.savefig(out_plot, dpi=300)
    plt.close()

    print(f"Saved sweep plot: {out_plot}")


if __name__ == '__main__':
    main()
