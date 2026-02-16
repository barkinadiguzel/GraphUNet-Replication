import torch
import torch.nn as nn
from layers.gpool import GPool
from layers.gunpool import GUnpool
from layers.gcn_layer import GCNLayer
from config import TOP_K, NUM_BLOCKS, EMBED_DIM

class GraphUNet(nn.Module):
    def __init__(self, input_dim):
        super(GraphUNet, self).__init__()
        self.embedding = GCNLayer(input_dim, EMBED_DIM)

        # Encoder blocks
        self.enc_gcns = nn.ModuleList([GCNLayer(EMBED_DIM, EMBED_DIM) for _ in range(NUM_BLOCKS)])
        self.enc_pools = nn.ModuleList([GPool(EMBED_DIM, TOP_K) for _ in range(NUM_BLOCKS)])

        # Decoder blocks
        self.dec_unpools = nn.ModuleList([GUnpool() for _ in range(NUM_BLOCKS)])
        self.dec_gcns = nn.ModuleList([GCNLayer(EMBED_DIM, EMBED_DIM) for _ in range(NUM_BLOCKS)])

        # Final GCN layer for output
        self.final_gcn = GCNLayer(EMBED_DIM, EMBED_DIM)

    def forward(self, X, A):
        X = self.embedding(X, A)
        idx_list, A_list, X_list = [], [], []

        # Encoder
        for gcn, pool in zip(self.enc_gcns, self.enc_pools):
            X = gcn(X, A)
            X, A, idx = pool(X, A)
            idx_list.append(idx)
            A_list.append(A)
            X_list.append(X)

        # Decoder
        for gcn, unpool, idx in zip(self.dec_gcns, self.dec_unpools, reversed(idx_list)):
            X = unpool(X, idx, N=A_list.pop(0).size(0))
            X = gcn(X, A_list.pop(0))

        X = self.final_gcn(X, A)
        return X
