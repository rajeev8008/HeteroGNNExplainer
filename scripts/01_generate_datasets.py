import os
import sys
import torch
import numpy as np

# Bind the src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.data_gen.shape_generator import HeteroDatasetGenerator

def main():
    # Iterate the homophily_coef from 0.1 (highly heterophilic) to 0.9 (highly homophilic) in increments of 0.2.
    homophily_levels = np.arange(0.1, 1.0, 0.2)
    
    # Path setup based on the project structure
    dataset_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(dataset_dir, exist_ok=True)
    
    generator = HeteroDatasetGenerator(num_nodes=1000)
    
    print("="*50)
    print("PHASE 1: Starting Dataset Generation")
    print("="*50)
    
    for h in homophily_levels:
        h_round = round(h, 2)
        print(f"\n--- Processing mixing parameter (homophily level): {h_round} ---")
        
        # Generate the dataset using our PyG wrapper
        data = generator.generate(homophily_coef=h_round)
        
        # Save standard PyG Data object to disk
        out_path = os.path.join(dataset_dir, f'dataset_homophily_{h_round:.2f}.pt')
        torch.save(data, out_path)
        print(f"[SUCCESS] Saved dataset to {out_path}")

    print("\n" + "="*50)
    print("Phase 1 Complete. All datasets generated and saved.")
    print("Proceed to Phase 2: Train the 'Black Box' Models.")

if __name__ == '__main__':
    main()
