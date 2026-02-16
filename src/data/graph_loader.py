import torch

def load_sample_graph():
    """
    Returns a toy graph for testing forward pass.
    X: Node features (N x C)
    A: Adjacency matrix (N x N)
    """
    # 6 nodes, 4 features each
    X = torch.randn(6, 4)
    
    # Sample adjacency matrix
    A = torch.tensor([
        [0,1,0,0,1,0],
        [1,0,1,0,0,0],
        [0,1,0,1,0,0],
        [0,0,1,0,1,0],
        [1,0,0,1,0,1],
        [0,0,0,0,1,0]
    ], dtype=torch.float32)

    return X, A
