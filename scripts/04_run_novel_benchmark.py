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
    
    # Load optimal weights if they exist (from sweep)
    optimal_weights_path = os.path.join(figures_dir, 'optimal_heterophily_weights.csv')
    opt_weights = {}
    if os.path.exists(optimal_weights_path):
        df_opt = pd.read_csv(optimal_weights_path)
        opt_weights = dict(zip(df_opt['homophily'], df_opt['best_weight']))
        print(f"[INFO] Loaded dynamic weights for {len(opt_weights)} homophily levels.")

    dataset_files = sorted(glob.glob(os.path.join(dataset_dir, 'dataset_homophily_*.pt')))
    
    if not dataset_files:
        print("No datasets found! Run Phase 1 first.")
        return
        
    print("="*50)
    print("PHASE 4: Evaluating Novel HeteroGNNExplainer")
    print("="*50)

    results_baseline = []
    results_novel = []
    results_pg = []
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
        
        # Determine weight for this homophily level
        h_weight = opt_weights.get(h_val, 0.5)
        
        print(f"\n--- Benchmarking on {filename} (Homophily: {h_val}, Alpha: {h_weight}) ---")
        
        data = torch.load(ds_path, weights_only=False)
        in_channels = data.x.size(1) if hasattr(data, 'x') and data.x is not None else 10
        out_channels = int(data.y.max().item() + 1) if hasattr(data, 'y') and data.y is not None else 2
        
        sage = GraphSAGE(in_channels=in_channels, hidden_channels=32, out_channels=out_channels)
        
        try:
            sage.load_state_dict(torch.load(os.path.join(model_dir, f'sage_{base_name}.pth'), weights_only=True))
        except FileNotFoundError:
            print(f"  [ERROR] Missing model weights for {base_name}. Run Phase 2.")
            continue
            
        sage.eval()
        
        baseline_explainer = get_baseline_explainer(sage, "gnn_explainer")
        pg_explainer = get_baseline_explainer(sage, "pg_explainer")
        novel_explainer = get_novel_explainer(sage, heterophily_weight=h_weight)
        
        if hasattr(data, 'train_mask') and data.train_mask is not None:
            candidate_nodes = torch.where(data.train_mask)[0]
        else:
            candidate_nodes = torch.arange(data.num_nodes)

        num_eval = min(50, int(candidate_nodes.numel()))
        test_nodes = candidate_nodes[torch.randperm(candidate_nodes.numel())[:num_eval]]
        
        # Preds for target fixing (remove UserWarnings)
        with torch.no_grad():
            full_logits = sage(data.x, data.edge_index)
            full_preds = full_logits.argmax(dim=-1)

        metrics = {
            'base': {'fid_plus': [], 'fid_minus': [], 'sparsity': []},
            'pg': {'fid_plus': [], 'fid_minus': [], 'sparsity': []},
            'novel': {'fid_plus': [], 'fid_minus': [], 'sparsity': []}
        }

        for node in test_nodes:
            idx = int(node.item())
            target = full_preds[node].view(1) # Ensure correct shape for PyG 2.5
            
            # Baseline
            exp_base = baseline_explainer(data.x, data.edge_index, index=idx, target=target)
            try:
                fp, fm = fidelity(baseline_explainer, exp_base)
                metrics['base']['fid_plus'].append(_to_scalar(fp))
                metrics['base']['fid_minus'].append(_to_scalar(fm))
            except Exception: pass
            metrics['base']['sparsity'].append(_compute_sparsity(exp_base))

            # PGExplainer
            try:
                exp_pg = pg_explainer(data.x, data.edge_index, index=idx, target=target)
                fp, fm = fidelity(pg_explainer, exp_pg)
                metrics['pg']['fid_plus'].append(_to_scalar(fp))
                metrics['pg']['fid_minus'].append(_to_scalar(fm))
                metrics['pg']['sparsity'].append(_compute_sparsity(exp_pg))
            except Exception: pass

            # Novel
            exp_novel = novel_explainer(data.x, data.edge_index, index=idx, target=target)
            try:
                fp, fm = fidelity(novel_explainer, exp_novel)
                metrics['novel']['fid_plus'].append(_to_scalar(fp))
                metrics['novel']['fid_minus'].append(_to_scalar(fm))
            except Exception: pass
            metrics['novel']['sparsity'].append(_compute_sparsity(exp_novel))

        # Calculate averages
        def avg(lst): return float(np.mean(lst)) if lst else 0.0
        
        avg_base_fp = avg(metrics['base']['fid_plus'])
        avg_pg_fp = avg(metrics['pg']['fid_plus'])
        avg_novel_fp = avg(metrics['novel']['fid_plus'])
        
        results_baseline.append(avg_base_fp)
        results_pg.append(avg_pg_fp)
        results_novel.append(avg_novel_fp)

        rows.append({
            'dataset': filename,
            'homophily': h_val,
            'alpha_used': h_weight,
            'baseline_fidelity_plus': avg_base_fp,
            'pg_fidelity_plus': avg_pg_fp,
            'novel_fidelity_plus': avg_novel_fp,
            'novel_sparsity': avg(metrics['novel']['sparsity'])
        })

        print(f"  Standard F+ : {avg_base_fp:.3f}")
        print(f"  PGExp F+    : {avg_pg_fp:.3f}")
        print(f"  Novel F+    : {avg_novel_fp:.3f} (Weight: {h_weight})")

    try:
        plt.figure(figsize=(10, 6))
        plt.plot(homophily_vals, results_baseline, marker='o', color='red', label='Standard GNNExplainer')
        plt.plot(homophily_vals, results_pg, marker='s', color='green', label='PGExplainer (Baseline)')
        plt.plot(homophily_vals, results_novel, marker='*', markersize=12, color='gold', label='HeteroGNNExplainer (Dynamic Alpha)')
        
        plt.xlabel('Homophily Coefficient (0.1 = Heterophilic)')
        plt.ylabel('Fidelity+ (Higher is Better)')
        plt.title('Comparison Across Baselines and Homophily Levels')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plot_path = os.path.join(figures_dir, 'comprehensive_benchmark.png')
        plt.savefig(plot_path)
        print(f"\n[SUCCESS] Comprehensive plot saved to {plot_path}")
    except Exception as e:
        print(f"\nPlotting failed: {e}")

    if rows:
        metrics_path = os.path.join(figures_dir, 'novel_benchmark_metrics.csv')
        pd.DataFrame(rows).sort_values('homophily').to_csv(metrics_path, index=False)
        print(f"[SUCCESS] Metrics saved to {metrics_path}")

if __name__ == '__main__':
    main()
