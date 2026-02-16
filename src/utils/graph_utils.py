import torch

def sample_graph():
    X = torch.randn(6, 4)  # 6 nodes, 4 features
    A = torch.tensor([
        [0,1,0,0,1,0],
        [1,0,1,0,0,0],
        [0,1,0,1,0,0],
        [0,0,1,0,1,0],
        [1,0,0,1,0,1],
        [0,0,0,0,1,0]
    ], dtype=torch.float32)
    return X, A

def augment_graph_power(A, k=2):
    A_power = A.clone()
    for _ in range(k-1):
        A_power = A_power @ A
    return (A_power > 0).float()  
