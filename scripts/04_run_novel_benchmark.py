import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.models.graphsage import GraphSAGE
from src.explainers.baselines import get_baseline_explainer
from src.explainers.hetero_explainer import get_novel_explainer
from torch_geometric.explain.metric import fidelity


def _to_scalar(v):
    return float(v.item()) if hasattr(v, 'item') else float(v)


def _compute_sparsity(explanation, threshold=0.5):
    """
    Sparsity = fraction of edges removed by the explainer mask.
    Higher sparsity means a more compact explanation.
    """
    edge_mask = getattr(explanation, 'edge_mask', None)
    if edge_mask is None or edge_mask.numel() == 0:
        return 1.0

    mask = edge_mask
    if mask.min().item() < 0.0 or mask.max().item() > 1.0:
        mask = torch.sigmoid(mask)

    kept = (mask >= threshold).float().mean().item()
    return 1.0 - kept

def main():
    dataset_dir = os.path.join(project_root, 'data', 'processed')
    model_dir = os.path.join(project_root, 'results', 'models')
    figures_dir = os.path.join(project_root, 'results', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    dataset_files = sorted(glob.glob(os.path.join(dataset_dir, 'dataset_homophily_*.pt')))
    
    if not dataset_files:
        print("No datasets found! Run Phase 1 first.")
        return
        
    print("="*50)
    print("PHASE 4: Evaluating Novel HeteroGNNExplainer")
    print("="*50)

    results_baseline = []
    results_novel = []
    homophily_vals = []
    rows = []

    for ds_path in dataset_files:
        filename = os.path.basename(ds_path)
        base_name = filename.replace('.pt', '')
        h_str = base_name.split('_')[-1]
        try:
            h_val = float(h_str)
        except ValueError:
            h_val = 0.5
            
        homophily_vals.append(h_val)
        print(f"\n--- Benchmarking Novel Explainer on {filename} (Homophily: {h_val}) ---")
        
        data = torch.load(ds_path, weights_only=False)
        in_channels = data.x.size(1) if hasattr(data, 'x') and data.x is not None else 10
        out_channels = int(data.y.max().item() + 1) if hasattr(data, 'y') and data.y is not None else 2
        
        # Benchmark primarily on GraphSAGE
        sage = GraphSAGE(in_channels=in_channels, hidden_channels=32, out_channels=out_channels)
        
        try:
            sage.load_state_dict(torch.load(os.path.join(model_dir, f'sage_{base_name}.pth'), weights_only=True))
        except FileNotFoundError:
            print(f"  [ERROR] Missing model weights for {base_name}. Run Phase 2.")
            continue
            
        sage.eval()
        
        baseline_explainer = get_baseline_explainer(sage, "gnn_explainer")
        
        # Setup our proposed explainer
        novel_explainer = get_novel_explainer(sage, heterophily_weight=0.5)
        
        if hasattr(data, 'train_mask') and data.train_mask is not None:
            candidate_nodes = torch.where(data.train_mask)[0]
        else:
            candidate_nodes = torch.arange(data.num_nodes)

        num_eval = min(50, int(candidate_nodes.numel()))
        test_nodes = candidate_nodes[torch.randperm(candidate_nodes.numel())[:num_eval]]
        
        base_fid_plus_list = []
        base_fid_minus_list = []
        base_sparsity_list = []

        novel_fid_plus_list = []
        novel_fid_minus_list = []
        novel_sparsity_list = []
        
        for node in test_nodes:
            idx = int(node.item())
            
            # Baseline Explanation
            exp_base = baseline_explainer(data.x, data.edge_index, index=idx)
            try:
                fid_b_plus, fid_b_minus = fidelity(baseline_explainer, exp_base)
                base_fid_plus_list.append(_to_scalar(fid_b_plus))
                base_fid_minus_list.append(_to_scalar(fid_b_minus))
            except Exception:
                base_fid_plus_list.append(0.0)
                base_fid_minus_list.append(0.0)
            base_sparsity_list.append(_compute_sparsity(exp_base))

            # Novel Explanation
            exp_novel = novel_explainer(data.x, data.edge_index, index=idx)
            try:
                fid_n_plus, fid_n_minus = fidelity(novel_explainer, exp_novel)
                novel_fid_plus_list.append(_to_scalar(fid_n_plus))
                novel_fid_minus_list.append(_to_scalar(fid_n_minus))
            except Exception:
                novel_fid_plus_list.append(0.0)
                novel_fid_minus_list.append(0.0)
            novel_sparsity_list.append(_compute_sparsity(exp_novel))

        avg_base_fid_plus = float(np.mean(base_fid_plus_list)) if base_fid_plus_list else 0.0
        std_base_fid_plus = float(np.std(base_fid_plus_list)) if base_fid_plus_list else 0.0
        avg_base_fid_minus = float(np.mean(base_fid_minus_list)) if base_fid_minus_list else 0.0
        std_base_fid_minus = float(np.std(base_fid_minus_list)) if base_fid_minus_list else 0.0
        avg_base_sparsity = float(np.mean(base_sparsity_list)) if base_sparsity_list else 0.0
        std_base_sparsity = float(np.std(base_sparsity_list)) if base_sparsity_list else 0.0

        avg_novel_fid_plus = float(np.mean(novel_fid_plus_list)) if novel_fid_plus_list else 0.0
        std_novel_fid_plus = float(np.std(novel_fid_plus_list)) if novel_fid_plus_list else 0.0
        avg_novel_fid_minus = float(np.mean(novel_fid_minus_list)) if novel_fid_minus_list else 0.0
        std_novel_fid_minus = float(np.std(novel_fid_minus_list)) if novel_fid_minus_list else 0.0
        avg_novel_sparsity = float(np.mean(novel_sparsity_list)) if novel_sparsity_list else 0.0
        std_novel_sparsity = float(np.std(novel_sparsity_list)) if novel_sparsity_list else 0.0
             
        results_baseline.append(avg_base_fid_plus)
        results_novel.append(avg_novel_fid_plus)

        rows.append({
            'dataset': filename,
            'homophily': h_val,
            'samples': num_eval,
            'baseline_fidelity_plus_mean': avg_base_fid_plus,
            'baseline_fidelity_plus_std': std_base_fid_plus,
            'baseline_fidelity_minus_mean': avg_base_fid_minus,
            'baseline_fidelity_minus_std': std_base_fid_minus,
            'baseline_sparsity_mean': avg_base_sparsity,
            'baseline_sparsity_std': std_base_sparsity,
            'novel_fidelity_plus_mean': avg_novel_fid_plus,
            'novel_fidelity_plus_std': std_novel_fid_plus,
            'novel_fidelity_minus_mean': avg_novel_fid_minus,
            'novel_fidelity_minus_std': std_novel_fid_minus,
            'novel_sparsity_mean': avg_novel_sparsity,
            'novel_sparsity_std': std_novel_sparsity,
            'fidelity_plus_gap': avg_novel_fid_plus - avg_base_fid_plus,
        })

        print(f"  Standard Fidelity+ : {avg_base_fid_plus:.3f} +/- {std_base_fid_plus:.3f}")
        print(f"  Standard Fidelity- : {avg_base_fid_minus:.3f} +/- {std_base_fid_minus:.3f}")
        print(f"  Standard Sparsity  : {avg_base_sparsity:.3f} +/- {std_base_sparsity:.3f}")
        print(f"  Novel Fidelity+    : {avg_novel_fid_plus:.3f} +/- {std_novel_fid_plus:.3f}")
        print(f"  Novel Fidelity-    : {avg_novel_fid_minus:.3f} +/- {std_novel_fid_minus:.3f}")
        print(f"  Novel Sparsity     : {avg_novel_sparsity:.3f} +/- {std_novel_sparsity:.3f}")

    try:
        plt.figure(figsize=(10, 6))
        plt.plot(homophily_vals, results_baseline, marker='o', color='red', label='Standard GNNExplainer')
        plt.plot(homophily_vals, results_novel, marker='*', markersize=10, color='gold', label='Our Novel HeteroGNNExplainer')
        
        plt.xlabel('Dataset Homophily Coefficient (0.1 = Heterophilic -> 0.9 = Homophilic)')
        plt.ylabel('Fidelity+ (Higher implies better explanation)')
        plt.title('Baseline vs Novel Explainer on Heterophilic Graphs')
        plt.grid(True)
        plt.legend()
        
        plot_path = os.path.join(figures_dir, 'novel_contribution_fidelity.png')
        plt.savefig(plot_path)
        print(f"\n[SUCCESS] Phase 4 Complete! Novel contribution diagram saved to {plot_path}")
    except Exception as e:
        print(f"\nPlotting failed: {e}")

    if rows:
        metrics_path = os.path.join(figures_dir, 'novel_benchmark_metrics.csv')
        pd.DataFrame(rows).sort_values('homophily').to_csv(metrics_path, index=False)
        print(f"[SUCCESS] Detailed metrics saved to {metrics_path}")

if __name__ == '__main__':
    main()
