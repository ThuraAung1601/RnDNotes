import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

from skimage.segmentation import slic
from skimage import graph

from tqdm import tqdm

# =====================================================
# Utils
# =====================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_params(n):
    return f"{n/1e3:.1f}K"

# =====================================================
# CNN
# =====================================================
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

# =====================================================
# Patch Graph Dataset
# =====================================================
class PatchGraphDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, patch=4):
        self.dataset = dataset
        self.patch = patch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.squeeze()

        P = self.patch
        nodes, pos = [], []

        for i in range(0, 28, P):
            for j in range(0, 28, P):
                patch = img[i:i+P, j:j+P]
                nodes.append(patch.mean())
                pos.append((i // P, j // P))

        x = torch.tensor(nodes, dtype=torch.float).unsqueeze(1)

        edges = []
        W = 28 // P
        for i, (r, c) in enumerate(pos):
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < W and 0 <= nc < W:
                    edges.append([i, nr * W + nc])

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return Data(x=x, edge_index=edge_index, y=torch.tensor(label))

# =====================================================
# Grid Graph Dataset (Pixel)
# =====================================================
class GridGraphDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.squeeze()
        H, W = img.shape

        x = img.view(-1, 1)

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

# =====================================================
# Superpixel Graph Dataset (SLIC)
# =====================================================
class SuperpixelGraphDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, n_segments=50):
        self.dataset = dataset
        self.n_segments = n_segments

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.squeeze().numpy()

        segments = slic(
            img,
            n_segments=self.n_segments,
            compactness=10,
            start_label=0,
            channel_axis=None   # IMPORTANT
        )

        rag = graph.rag_mean_color(img, segments)

        x = []
        for node in rag.nodes:
            mask = segments == node
            x.append(img[mask].mean())

        edges = []
        for u, v in rag.edges:
            edges.append([u, v])
            edges.append([v, u])

        return Data(
            x=torch.tensor(x, dtype=torch.float).unsqueeze(1),
            edge_index=torch.tensor(edges, dtype=torch.long).t(),
            y=torch.tensor(label)
        )

# =====================================================
# GCN / GAT
# =====================================================
class SmallGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_classes):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x = F.relu(self.c1(data.x, data.edge_index))
        x = F.relu(self.c2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.fc(x)

class SmallGAT(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, heads=2):
        super().__init__()
        self.c1 = GATConv(in_dim, hidden // heads, heads=heads)
        self.c2 = GATConv(hidden, hidden // heads, heads=heads)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x = F.relu(self.c1(data.x, data.edge_index))
        x = F.relu(self.c2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.fc(x)

# =====================================================
# Train / Eval
# =====================================================
def run_epoch(model, loader, opt, crit, device, train=True, desc=""):
    model.train() if train else model.eval()
    correct = total = 0

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc=desc, leave=False):
            if isinstance(batch, Data):
                batch = batch.to(device)
                out = model(batch)
                y = batch.y
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x)

            loss = crit(out, y)

            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    return 100 * correct / total

# =====================================================
# MAIN
# =====================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    datasets = {
        "MNIST": torchvision.datasets.MNIST,
        "EMNIST": lambda root, train, download, transform:
            torchvision.datasets.EMNIST(
                root=root, split="balanced",
                train=train, download=download,
                transform=transform
            )
    }

    EPOCHS = 1

    for name, DS in datasets.items():
        print(f"\n=== {name} ===")

        train_ds = DS("./data", train=True, download=True, transform=T.ToTensor())
        test_ds  = DS("./data", train=False, download=True, transform=T.ToTensor())

        num_classes = len(train_ds.classes)
        crit = nn.CrossEntropyLoss()

        # CNN
        cnn = SmallCNN(num_classes).to(device)
        opt = torch.optim.Adam(cnn.parameters(), 1e-3)
        train_dl = torch.utils.data.DataLoader(train_ds, 128, shuffle=True)
        test_dl  = torch.utils.data.DataLoader(test_ds, 128)

        best = 0
        for ep in range(EPOCHS):
            run_epoch(cnn, train_dl, opt, crit, device, True,
                      desc=f"CNN {name} Epoch {ep+1}/{EPOCHS}")
            best = max(best, run_epoch(cnn, test_dl, opt, crit, device, False,
                                       desc="Eval"))

        results.append(("CNN", name, count_params(cnn), best))
        print(f"CNN  {format_params(count_params(cnn))}  {best:.2f}%")

        # GNN Ablations
        graph_sets = {
            "Patch": PatchGraphDataset(train_ds, 4),
            "Grid": GridGraphDataset(train_ds),
            "Superpixel": SuperpixelGraphDataset(train_ds, 50)
        }
        test_sets = {
            "Patch": PatchGraphDataset(test_ds, 4),
            "Grid": GridGraphDataset(test_ds),
            "Superpixel": SuperpixelGraphDataset(test_ds, 50)
        }

        for gtype in graph_sets:
            bs = 16 if gtype == "Grid" else 64
            train_g = DataLoader(graph_sets[gtype], bs, shuffle=True)
            test_g  = DataLoader(test_sets[gtype], bs)

            for Model, tag in [(SmallGCN, "GCN"), (SmallGAT, "GAT")]:
                model = Model(1, 32, num_classes).to(device)
                opt = torch.optim.Adam(model.parameters(), 1e-3)

                best = 0
                for ep in range(EPOCHS):
                    run_epoch(model, train_g, opt, crit, device, True,
                              desc=f"{tag}-{gtype} {name} Epoch {ep+1}/{EPOCHS}")
                    best = max(best, run_epoch(model, test_g, opt, crit, device, False,
                                               desc="Eval"))

                name_tag = f"{tag}-{gtype}"
                results.append((name_tag, name, count_params(model), best))
                print(f"{name_tag}  {format_params(count_params(model))}  {best:.2f}%")

    # Plot
    plt.figure(figsize=(7,5))
    for m, ds, p, acc in results:
        plt.scatter(p, acc)
        plt.text(p, acc+0.3, f"{m}-{ds}", fontsize=7)

    plt.xscale("log")
    plt.xlabel("# Parameters")
    plt.ylabel("Accuracy (%)")
    plt.title("CNN vs GNN Ablation (MNIST / EMNIST)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
