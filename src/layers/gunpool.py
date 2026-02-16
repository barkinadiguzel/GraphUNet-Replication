import torch
import torch.nn as nn

class GUnpool(nn.Module):
    def __init__(self):
        super(GUnpool, self).__init__()

    def forward(self, X, idx, N):
        C = X.size(1)
        X_restored = torch.zeros(N, C, device=X.device)
        X_restored[idx, :] = X
        return X_restored
