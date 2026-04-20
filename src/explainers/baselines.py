from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer

def get_baseline_explainer(model, explainer_type="gnn_explainer"):
    """
    Returns a unified PyG Explainer wrapping the baseline model.
    """
    if explainer_type == "gnn_explainer":
        algorithm = GNNExplainer(epochs=50)
    elif explainer_type == "pg_explainer":
        algorithm = PGExplainer(epochs=30)
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
