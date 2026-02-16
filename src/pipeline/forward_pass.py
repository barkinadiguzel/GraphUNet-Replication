import torch
from model.graph_unet import GraphUNet
from utils.graph_utils import sample_graph, augment_graph_power

TOP_K = 3       # Number of nodes to select in gPool
NUM_BLOCKS = 2  
EMBED_DIM = 8   

def main():
    X, A = sample_graph()
    A_aug = augment_graph_power(A, k=2)

    model = GraphUNet(input_dim=X.size(1))

    X_out = model(X, A_aug)

    print("Input Features Shape:", X.shape)
    print("Output Features Shape:", X_out.shape)
    print("Output Features:", X_out)

if __name__ == "__main__":
    main()
