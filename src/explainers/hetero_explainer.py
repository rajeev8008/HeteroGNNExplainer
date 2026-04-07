import torch
import torch.nn.functional as F
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain import Explainer

class HeteroGNNExplainerAlgorithm(GNNExplainer):
    """
    Novel Contribution: Heterophily-Aware Explainer Algorithm.
    Extends standard GNNExplainer by modifying the objective function 
    to explicitly account for disassortative mixing.
    """
    def __init__(self, epochs=100, heterophily_weight=0.5, **kwargs):
        super().__init__(epochs=epochs, **kwargs)
        self.heterophily_weight = heterophily_weight
        self.cached_x = None
        self.cached_edge_index = None
        
    def forward(self, model, x, edge_index, **kwargs):
        # Intercept the forward pass to cache features and structure
        # so we can use them inside our custom loss function.
        self.cached_x = x
        self.cached_edge_index = edge_index
        return super().forward(model, x, edge_index, **kwargs)

    def _loss(self, y_hat, y, **kwargs):
        # 1. Base Objective: Fidelity prediction loss + Entropy + Size limits
        base_loss = super()._loss(y_hat, y, **kwargs)
        
        # 2. Novel Objective: Disassortative Mixing Penalty for Heterophily
        hetero_penalty = torch.tensor(0.0, device=base_loss.device)
        
        edge_mask = kwargs.get('edge_mask', getattr(self, 'edge_mask', None))
        
        if edge_mask is not None and self.cached_x is not None and self.cached_edge_index is not None:
            src, dst = self.cached_edge_index
            src_x = self.cached_x[src]
            dst_x = self.cached_x[dst]
            
            # Calculate Cosine Similarity between connected nodes
            # High similarity = Homophilic edge
            # Low similarity = Heterophilic edge
            similarity = F.cosine_similarity(src_x, dst_x, dim=-1)
            
            # In a heterophilic environment, explainers fail because they drop heterophilic edges as "noise".
            # We explicitly PENALIZE the network for relying on highly similar nodes,
            # forcing the edge_mask to highlight edges between dissimilar nodes.
            # Using ReLU to only penalize purely similar edges:
            similarity_cost = F.relu(similarity)
            
            # Weight the penalty by the current edge mask confidence
            hetero_penalty = (edge_mask * similarity_cost).mean()
        
        # Combined Objective
        return base_loss + (self.heterophily_weight * hetero_penalty)


def get_novel_explainer(model, heterophily_weight=0.2):
    """
    Returns the unified PyG Explainer wrapping our Novel Custom Algorithm.
    """
    algorithm = HeteroGNNExplainerAlgorithm(epochs=50, heterophily_weight=heterophily_weight)

    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type='model',
        node_mask_type='object',
        edge_mask_type='object', 
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )
    return explainer
