import torch
from torch_geometric.data import Data

def test_gnn():
    # Define dummy graph data
    num_nodes = 10
    num_features = 5
    edge_index = torch.randint(0, num_nodes, (2, 20))  # Random edges
    x = torch.rand((num_nodes, num_features))  # Random node features
    batch = torch.zeros(num_nodes, dtype=torch.long)  # Single batch

    # Create graph data object
    data = Data(x=x, edge_index=edge_index, batch=batch)

    # Initialize GNN model
    model = GNN(
        dim_features=num_features,
        dim_target=3,  # Example: predicting 3 classes
        layers=[16, 16, 8],
        pooling='max',
        dropout=0.1,
        conv='GINConv'
    )

    # Run model forward pass
    out, Z, node_embeddings = model(data)

    # Print output shapes
    print(f"Graph-level output shape: {out.shape}")  # Should be (1, dim_target)
    print(f"Node-level embeddings shape: {Z.shape}")  # Should be (num_nodes, dim_target)
    print(f"Final node feature shape: {node_embeddings.shape}")  # Should match (num_nodes, last layer size)

    # Assert expected shapes
    assert out.shape == (1, 3), "Unexpected graph output shape!"
    assert Z.shape[0] == num_nodes, "Mismatch in node embedding count!"
    assert node_embeddings.shape[0] == num_nodes, "Node embeddings shape mismatch!"

    print("✅ GNN test passed!")

if __name__ == "__main__":
    test_gnn()
