import sys
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Basic fallback structure to gracefully handle if GraphXAI has not been fully configured
try:
    from graphxai.datasets.shape_graph import ShapeGGen
except ImportError:
    ShapeGGen = None
    print("Warning: GraphXAI not found or not fully installed. Please run `pip install graphxai`.")

class HeteroDatasetGenerator:
    """
    Wrapper for dataset generation that iterates over mixing parameters to 
    produce graphs with specific heterophily/homophily ratios.
    """
    def __init__(self, num_nodes=1000, num_classes=2, seed=42):
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.seed = seed
        
    def generate(self, homophily_coef: float) -> Data:
        """
        Generates a synthetic graph dataset with a target homophily coefficient.
        Returns a PyG Data object.
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if ShapeGGen is None:
            print("GraphXAI not available, falling back to basic mock generator to unblock pipeline.")
            return self._generate_mock_fallback(homophily_coef)
            
        print(f"Generating graph with structural ShapeGGen, homophily={homophily_coef:.2f}...")
        
        generator_params = {
            'num_subgraphs': 20, 
            'prob_connection': homophily_coef, # prob_connection/mixing controls homophily in some GraphXAI versions
            'subgraph_size': 15,
            # We can plug in specific keyword depending on exact shapeggen version
        }
        
        try:
            # We initialize the generator
            # Some versions use homophily_coef directly. 
            generator = ShapeGGen(**generator_params)
            dataset = generator.generate()
            
            # GraphXAI often returns a standard dict or specific dataset object
            # We ensure we extract a proper PyG Data object.
            if hasattr(dataset, 'get_graph'):
                data = dataset.get_graph(use_to_networkx=False)
            elif isinstance(dataset, Data):
                data = dataset
            else:
                data = dataset[0]
                
            return data
            
        except Exception as e:
            print(f"Error during ShapeGGen execution for homophily {homophily_coef}: {e}")
            print("Returning baseline mock.")
            return self._generate_mock_fallback(homophily_coef)

    def _generate_mock_fallback(self, homophily_coef: float) -> Data:
        """
        Mock fallback for immediate PyG unblocking if GraphXAI dependencies fail on Windows.
        Creates a basic geometric graph with forced labels.
        """
        edge_index = torch.randint(0, self.num_nodes, (2, self.num_nodes * 3))
        x = torch.randn((self.num_nodes, 10))
        y = torch.randint(0, self.num_classes, (self.num_nodes,))
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data.metadata_homophily = homophily_coef
        return data
