import os
import sys
import glob
import torch
import torch.nn.functional as F

# Bind the src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE

def train_model(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    
    # Generate mock training nodes if GraphXAI didn't provide train_mask implicitly
    # In standard research benchmarking, we use a fixed 80/20 train/test split.
    num_nodes = data.x.size(0)
    if not hasattr(data, 'train_mask') or data.train_mask is None:
        indices = torch.randperm(num_nodes)
        train_idx = indices[:int(0.8 * num_nodes)]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        data.train_mask = train_mask

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Calculate loss only predicting on the training nodes
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    return model

def main():
    dataset_dir = os.path.join(project_root, 'data', 'processed')
    model_out_dir = os.path.join(project_root, 'results', 'models')
    os.makedirs(model_out_dir, exist_ok=True)
    
    dataset_files = glob.glob(os.path.join(dataset_dir, 'dataset_homophily_*.pt'))
    
    if not dataset_files:
        print("No datasets found! Please run Phase 1 first.")
        return

    print("="*50)
    print("PHASE 2: Training Base 'Black-Box' Models")
    print("="*50)

    for ds_path in dataset_files:
        filename = os.path.basename(ds_path)
        print(f"\n--- Training on {filename} ---")
        
        data = torch.load(ds_path, weights_only=False)
        
        # Robustly determine feature size and label classes
        in_channels = data.x.size(1) if hasattr(data, 'x') and data.x is not None else 10
        # If classes are contiguous 0-N
        out_channels = int(data.y.max().item() + 1) if hasattr(data, 'y') and data.y is not None else 2
        hidden_channels = 32
        
        # Initialize GNNs
        gcn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)
        sage = GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)
        
        print(" >> Training GCN (Homophily-biased)...")
        gcn = train_model(gcn, data)
        print(" >> Training GraphSAGE (Heterophily-aware)...")
        sage = train_model(sage, data)
        
        # Freeze and Save Model Weights
        base_name = filename.replace('.pt', '')
        gcn_path = os.path.join(model_out_dir, f'gcn_{base_name}.pth')
        sage_path = os.path.join(model_out_dir, f'sage_{base_name}.pth')
        
        torch.save(gcn.state_dict(), gcn_path)
        torch.save(sage.state_dict(), sage_path)
        print(f"[SUCCESS] Saved model weights to {model_out_dir}")

    print("\n" + "="*50)
    print("Phase 2 Complete. All baseline models trained and frozen.")
    print("Our target models are now ready for Phase 3: the XAI Benchmarks.")

if __name__ == '__main__':
    main()
