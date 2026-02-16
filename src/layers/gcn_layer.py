import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, A):
        I = torch.eye(A.size(0), device=A.device)
        A_hat = A + 2 * I
        D_hat = torch.diag(A_hat.sum(1).pow(-0.5))
        X_new = D_hat @ A_hat @ D_hat @ X
        X_new = self.linear(X_new)
        return torch.relu(X_new)
