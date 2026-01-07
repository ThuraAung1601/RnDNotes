class PatchGraphDataset(torch.utils.data.Dataset):
    """
    Image → Patch Graph

    Each node = one P×P image patch
    Each edge = spatial adjacency between patches
    """

    def __init__(self, dataset, patch=4):
        self.dataset = dataset
        self.patch = patch  # P = 4 → 4×4 pixel patches

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.squeeze()  # (28, 28)

        P = self.patch

        # Nodes:
        # 28 / 4 = 7 patches per dimension
        # total nodes = 7 × 7 = 49 nodes
        nodes, pos = [], []

        for i in range(0, 28, P):
            for j in range(0, 28, P):
                patch = img[i:i+P, j:j+P]

                # Node feature = mean intensity of the patch
                # → 1 scalar per node
                nodes.append(patch.mean())

                # Patch grid position (row, col)
                pos.append((i // P, j // P))

        # Node feature matrix:
        # shape = (49, 1)
        x = torch.tensor(nodes, dtype=torch.float).unsqueeze(1)

        # Edges:
        # 4-neighborhood (up, down, left, right)
        # preserves spatial structure of the image
        edges = []
        W = 28 // P  # 7

        for i, (r, c) in enumerate(pos):
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < W and 0 <= nc < W:
                    edges.append([i, nr * W + nc])

        edge_index = torch.tensor(edges, dtype=torch.long).t()

        return Data(x=x, edge_index=edge_index, y=torch.tensor(label))

class GridGraphDataset(torch.utils.data.Dataset):
    """
    Image → Pixel Graph

    Each node = one pixel
    Each edge = adjacency between neighboring pixels
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.squeeze()  # (28, 28)

        H, W = img.shape

        # Nodes:
        # 28 × 28 = 784 nodes
        # Node feature = pixel intensity
        x = img.view(-1, 1)  # (784, 1)

        # Edges:
        # Each pixel connects to 4 neighbors
        # Very dense and memory-intensive graph
        edges = []

        for r in range(H):
            for c in range(W):
                i = r * W + c
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        edges.append([i, nr * W + nc])

        edge_index = torch.tensor(edges, dtype=torch.long).t()

        return Data(x=x, edge_index=edge_index, y=torch.tensor(label))

class SuperpixelGraphDataset(torch.utils.data.Dataset):
    """
    Image → Superpixel Graph (SLIC)

    Each node = one superpixel region
    Each edge = adjacency between superpixels
    """

    def __init__(self, dataset, n_segments=50):
        self.dataset = dataset
        self.n_segments = n_segments  # target number of superpixels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.squeeze().numpy()

        # SLIC segmentation:
        # n_segments = 50 → ~50 regions per image
        segments = slic(
            img,
            n_segments=self.n_segments,
            compactness=10,
            start_label=0,
            channel_axis=None
        )

        # Region Adjacency Graph (RAG)
        rag = graph.rag_mean_color(img, segments)

        # Nodes:
        # One node per superpixel
        # Feature = mean intensity of region
        x = []
        for node in rag.nodes:
            mask = segments == node
            x.append(img[mask].mean())

        # Edges:
        # Connect neighboring superpixels
        edges = []
        for u, v in rag.edges:
            edges.append([u, v])
            edges.append([v, u])  # make graph undirected

        return Data(
            x=torch.tensor(x, dtype=torch.float).unsqueeze(1),
            edge_index=torch.tensor(edges, dtype=torch.long).t(),
            y=torch.tensor(label)
        )
