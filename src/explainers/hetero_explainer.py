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
        self.heterophily_weight = float(heterophily_weight)
        self.cached_x = None
        self.cached_edge_index = None
        self.peak_memory = 0

    def set_heterophily_weight(self, value: float):
        """Dynamically tune the heterophily regularization coefficient."""
        self.heterophily_weight = float(value)
        
    def forward(self, model, x, edge_index, **kwargs):
        # Intercept the forward pass to cache features and structure
        # so we can use them inside our custom loss function.
        self.cached_x = x
        self.cached_edge_index = edge_index
        
        # Track memory overhead
        if torch.cuda.is_available():
            self.peak_memory = max(self.peak_memory, torch.cuda.memory_allocated())
            
        return super().forward(model, x, edge_index, **kwargs)

    def get_memory_report(self):
        """Returns peak memory observed during forward passes."""
        return self.peak_memory

    def _loss(self, y_hat, y, **kwargs):
        # 1. Base Objective: Fidelity prediction loss + Entropy + Size limits
        base_loss = super()._loss(y_hat, y, **kwargs)
        
        # 2. Novel Objective: Margin-based Heterophily Reward
        hetero_loss = torch.tensor(0.0, device=base_loss.device)
        
        edge_mask = getattr(self, 'edge_mask', None)
        
        if edge_mask is not None and self.cached_x is not None and self.cached_edge_index is not None:
            activated_mask = torch.sigmoid(edge_mask)
            
            src, dst = self.cached_edge_index
            src_x = self.cached_x[src]
            dst_x = self.cached_x[dst]
            
            # Calculate Cosine Similarity [-1, 1]
            similarity = F.cosine_similarity(src_x, dst_x, dim=-1)
            
            # MARGIN-BASED REFINEMENT:
            # We calculate the mean similarity of the neighborhood (the cached subgraph).
            # We only reward dissimilarity that is GREATER than average (i.e., similarity < mean).
            avg_sim = similarity.mean()
            
            # If similarity < avg_sim, diff is negative -> Reward.
            # If similarity > avg_sim, diff is positive -> Penalty.
            diff = similarity - avg_sim
            
            weighted_diff = activated_mask * diff
            num_edges = max(int(weighted_diff.numel()), 1)
            hetero_loss = weighted_diff.sum() / num_edges
        
        # Combined Objective
        return base_loss + (self.heterophily_weight * hetero_loss)


def get_novel_explainer(model, heterophily_weight=0.2):
    """
    Returns the unified PyG Explainer wrapping our Novel Custom Algorithm.
    Updated for PyG 2.5+ API compatibility.
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


def set_explainer_heterophily_weight(explainer: Explainer, heterophily_weight: float):
    """Update heterophily weight after explainer creation (useful for sweeps)."""
    algorithm = getattr(explainer, 'algorithm', None)
    if algorithm is None or not hasattr(algorithm, 'set_heterophily_weight'):
        raise ValueError('Provided explainer does not support dynamic heterophily tuning.')
    algorithm.set_heterophily_weight(heterophily_weight)
