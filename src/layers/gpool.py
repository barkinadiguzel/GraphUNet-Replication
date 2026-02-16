import torch
import torch.nn as nn

class GPool(nn.Module):
    def __init__(self, in_features, k):
        super(GPool, self).__init__()
        self.k = k
        self.proj = nn.Parameter(torch.randn(in_features))  

    def forward(self, X, A):
        y = X @ self.proj / self.proj.norm()   
        topk = torch.topk(y, self.k)
        idx = topk.indices
        y_selected = torch.sigmoid(topk.values)
        X_new = X[idx, :] * y_selected.unsqueeze(1)  
        A_new = A[idx][:, idx]

        return X_new, A_new, idx
