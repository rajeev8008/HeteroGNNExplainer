import torch
from torch_geometric.explain import Explainer, GNNExplainer

def get_baseline_explainer(model, explainer_type="gnn_explainer"):
    """
    Returns a unified PyG Explainer wrapping the baseline model.
    """
    if explainer_type == "gnn_explainer":
        # Epochs scale down for faster automated testing, typical is 200
        algorithm = GNNExplainer(epochs=50)
    else:
        raise ValueError(f"Unknown baseline explainer: {explainer_type}")

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
