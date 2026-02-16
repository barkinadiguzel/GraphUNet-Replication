import torch
import torch.nn as nn
from layers.gpool import GPool
from layers.gunpool import GUnpool
from layers.gcn_layer import GCNLayer
from config import TOP_K, NUM_BLOCKS, EMBED_DIM

class GraphUNet(nn.Module):
    def __init__(self, input_dim):
        super(GraphUNet, self).__init__()
        # Initial embedding
        self.embedding = GCNLayer(input_dim, EMBED_DIM)

        # Encoder blocks
        self.enc_gcns = nn.ModuleList([GCNLayer(EMBED_DIM, EMBED_DIM) for _ in range(NUM_BLOCKS)])
        self.enc_pools = nn.ModuleList([GPool(EMBED_DIM, TOP_K) for _ in range(NUM_BLOCKS)])

        # Decoder blocks
        self.dec_unpools = nn.ModuleList([GUnpool() for _ in range(NUM_BLOCKS)])
        self.dec_gcns = nn.ModuleList([GCNLayer(EMBED_DIM, EMBED_DIM) for _ in range(NUM_BLOCKS)])

        # Final GCN for network embedding
        self.final_gcn = GCNLayer(EMBED_DIM, EMBED_DIM)

    def forward(self, X, A):
        X = self.embedding(X, A)
        skip_X, skip_A, idx_list = [], [], []
        
        for gcn, pool in zip(self.enc_gcns, self.enc_pools):
            X = gcn(X, A)
            X, A, idx = pool(X, A)
            skip_X.append(X)  
            skip_A.append(A)
            idx_list.append(idx)

        for gcn, unpool, X_skip, idx, A_enc in zip(
            self.dec_gcns, self.dec_unpools, reversed(skip_X), reversed(idx_list), reversed(skip_A)
        ):
            X = unpool(X, idx, N=A_enc.size(0))  
            X = X + X_skip  
            X = gcn(X, A_enc)

        X = self.final_gcn(X, A)
        return X
