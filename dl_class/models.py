class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Conv2d(1, 8, 3, padding=1)
        # 1  → grayscale image (MNIST / EMNIST)
        # 8  → learn 8 visual feature maps (edges, strokes)
        # 3  → 3×3 local receptive field
        # pad=1 → keep spatial size: 28×28 → 28×28
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)

        # Conv2d(8, 8, 3, padding=1)
        # 8 → input feature maps from previous layer
        # 8 → keep model small for fair CNN vs GNN comparison
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)

        # Global pooling:
        # (8, 28, 28) → (8, 1, 1)
        # removes spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Linear(8 → num_classes)
        # 8 learned features → class scores
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

class SmallGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_classes):
        super().__init__()

        # GCNConv(in_dim → hidden)
        # in_dim = 1
        #   → each node has 1 scalar feature
        #      (pixel intensity / patch mean / superpixel mean)
        # hidden = 32
        #   → node embedding size after message passing
        self.c1 = GCNConv(in_dim, hidden)

        # GCNConv(hidden → hidden)
        # keeps embedding dimension stable
        # allows multi-hop information aggregation
        self.c2 = GCNConv(hidden, hidden)

        # Linear(hidden → num_classes)
        # after graph pooling, one vector represents the whole image
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, data):
        # Node-level message passing
        x = F.relu(self.c1(data.x, data.edge_index))
        x = F.relu(self.c2(x, data.edge_index))

        # Graph-level pooling:
        # aggregates all node embeddings into one vector per image
        x = global_mean_pool(x, data.batch)

        return self.fc(x)
class SmallGAT(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, heads=2):
        super().__init__()

        # GATConv(in_dim → hidden / heads, heads)
        # heads = 2
        #   → use two attention mechanisms per node
        # hidden / heads
        #   → total output dimension remains = hidden
        self.c1 = GATConv(in_dim, hidden // heads, heads=heads)

        # Second attention layer
        # allows nodes to attend to more distant neighbors
        self.c2 = GATConv(hidden, hidden // heads, heads=heads)

        # Linear(hidden → num_classes)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x = F.relu(self.c1(data.x, data.edge_index))
        x = F.relu(self.c2(x, data.edge_index))

        # Graph-level representation
        x = global_mean_pool(x, data.batch)

        return self.fc(x)
