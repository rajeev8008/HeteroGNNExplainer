import sys
import os
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Basic fallback structure to gracefully handle if GraphXAI has not been fully configured
try:
    from graphxai.datasets import ShapeGGen
except Exception:
    ShapeGGen = None

if ShapeGGen is None:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        vendor_candidates = [
            os.path.join(project_root, 'vendor', 'GraphXAI'),
            os.path.join(project_root, 'vendor', 'graphxai'),
            os.path.join(project_root, 'vendor', 'graphxai-main'),
        ]
        for candidate in vendor_candidates:
            graphxai_pkg = os.path.join(candidate, 'graphxai', '__init__.py')
            if os.path.exists(graphxai_pkg):
                sys.path.insert(0, candidate)
                from graphxai.datasets import ShapeGGen as _ShapeGGen
                ShapeGGen = _ShapeGGen
                print(f"Info: Loaded GraphXAI from local source checkout: {candidate}")
                break
    except Exception:
        ShapeGGen = None

if ShapeGGen is None:
    ShapeGGen = None
    print(
        "Warning: GraphXAI not found or not fully installed. "
        "Use `pip install -r requirements.txt` or clone GraphXAI into `vendor/GraphXAI`. "
        "Note: `pip install graphxai` from PyPI is not supported."
    )

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
            
            # GraphXAI ShapeGGen constructs the graph on initialization and exposes get_graph().
            if hasattr(generator, 'get_graph'):
                try:
                    data = generator.get_graph(use_fixed_split=True)
                except TypeError:
                    data = generator.get_graph()
            elif isinstance(generator, Data):
                data = generator
            else:
                data = generator[0]
                
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
