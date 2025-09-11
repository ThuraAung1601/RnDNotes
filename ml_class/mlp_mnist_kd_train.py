import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np
from collections import defaultdict

# ==============================
# Parse command-line arguments
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")

    # Main settings
    parser.add_argument('--method', type=str, default='vanilla',
                        choices=['vanilla', 'vanilla_kd', 'feature_kd', 'simkd'],
                        help="Training method: vanilla | vanilla_kd | feature_kd | simkd")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")

    # KD hyperparameters
    parser.add_argument('--temperature', type=float, default=4.0, help="Temperature for KD")
    parser.add_argument('--alpha', type=float, default=0.5, help="Weight for KD loss vs CE loss")
    parser.add_argument('--beta', type=float, default=0.5, help="Weight for feature/similarity loss")

    return parser.parse_args()

# ==============================
# Seed and device
# ==============================
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================
# Data loading
# ==============================
def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ==============================
# Models
# ==============================
class TeacherMLP(nn.Module):
    def __init__(self):
        super(TeacherMLP, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 1200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1200, 800),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(800, 400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(400, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.features(x)
        output = self.classifier(features)
        return output, features

class StudentMLP(nn.Module):
    def __init__(self):
        super(StudentMLP, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.features(x)
        output = self.classifier(features)
        return output, features

# ==============================
# Utility functions
# ==============================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

# ==============================
# Training methods
# ==============================
def train_teacher(model, train_loader, test_loader, epochs, lr):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Teacher Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f} - Acc: {evaluate_model(model, test_loader):.4f}")

def train_student_vanilla(student, train_loader, test_loader, epochs, lr):
    student.train()
    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, _ = student(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Student Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f} - Acc: {evaluate_model(student, test_loader):.4f}")

def train_vanilla_kd(teacher, student, train_loader, test_loader, epochs, lr, temperature, alpha):
    teacher.eval()
    student.train()
    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                teacher_output, _ = teacher(data)
                teacher_soft = F.softmax(teacher_output / temperature, dim=1)

            optimizer.zero_grad()
            student_output, _ = student(data)
            student_soft = F.log_softmax(student_output / temperature, dim=1)

            kd_loss = criterion_kd(student_soft, teacher_soft) * (temperature ** 2)
            ce_loss = criterion_ce(student_output, target)
            loss = alpha * kd_loss + (1 - alpha) * ce_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Vanilla KD Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f} - Acc: {evaluate_model(student, test_loader):.4f}")

def train_feature_kd(teacher, student, train_loader, test_loader, epochs, lr, temperature, alpha, beta):
    teacher.eval()
    student.train()
    optimizer = optim.Adam(student.parameters(), lr=lr)
    feat_adapter = nn.Linear(64, 400).to(device)
    feat_optimizer = optim.Adam(feat_adapter.parameters(), lr=lr)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    criterion_feat = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                teacher_output, teacher_features = teacher(data)
                teacher_soft = F.softmax(teacher_output / temperature, dim=1)

            optimizer.zero_grad()
            feat_optimizer.zero_grad()
            student_output, student_features = student(data)
            student_soft = F.log_softmax(student_output / temperature, dim=1)

            adapted_student_features = feat_adapter(student_features)

            kd_loss = criterion_kd(student_soft, teacher_soft) * (temperature ** 2)
            ce_loss = criterion_ce(student_output, target)
            feat_loss = criterion_feat(adapted_student_features, teacher_features)

            loss = alpha * kd_loss + (1 - alpha) * ce_loss + beta * feat_loss
            loss.backward()

            optimizer.step()
            feat_optimizer.step()
            total_loss += loss.item()
        print(f"Feature KD Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f} - Acc: {evaluate_model(student, test_loader):.4f}")

def train_simkd(teacher, student, train_loader, test_loader, epochs, lr, temperature, alpha, beta):
    teacher.eval()
    student.train()
    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')

    def similarity_loss(f_s, f_t):
        f_s = F.normalize(f_s, p=2, dim=1)
        f_t = F.normalize(f_t, p=2, dim=1)
        return F.mse_loss(torch.mm(f_s, f_s.t()), torch.mm(f_t, f_t.t()))

    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                teacher_output, teacher_features = teacher(data)
                teacher_soft = F.softmax(teacher_output / temperature, dim=1)

            optimizer.zero_grad()
            student_output, student_features = student(data)
            student_soft = F.log_softmax(student_output / temperature, dim=1)

            kd_loss = criterion_kd(student_soft, teacher_soft) * (temperature ** 2)
            ce_loss = criterion_ce(student_output, target)
            sim_loss = similarity_loss(student_features, teacher_features)

            loss = alpha * kd_loss + (1 - alpha) * ce_loss + beta * sim_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"SimKD Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f} - Acc: {evaluate_model(student, test_loader):.4f}")

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    args = parse_args()

    print(f"Selected Method: {args.method}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print(f"KD Hyperparams -> Temperature: {args.temperature}, Alpha: {args.alpha}, Beta: {args.beta}")

    # Load data
    train_loader, test_loader = get_dataloaders(args.batch_size)

    # Load models
    teacher = TeacherMLP().to(device)
    student = StudentMLP().to(device)

    # Always train teacher first
    print("\nTraining Teacher Model...")
    train_teacher(teacher, train_loader, test_loader, epochs=3, lr=args.lr)

    # Choose method
    if args.method == "vanilla":
        train_student_vanilla(student, train_loader, test_loader, args.epochs, args.lr)
    elif args.method == "vanilla_kd":
        train_vanilla_kd(teacher, student, train_loader, test_loader,
                         args.epochs, args.lr, args.temperature, args.alpha)
    elif args.method == "feature_kd":
        train_feature_kd(teacher, student, train_loader, test_loader,
                         args.epochs, args.lr, args.temperature, args.alpha, args.beta)
    elif args.method == "simkd":
        train_simkd(teacher, student, train_loader, test_loader,
                    args.epochs, args.lr, args.temperature, args.alpha, args.beta)

    # Final evaluation
    final_acc = evaluate_model(student, test_loader)
    print(f"\nFinal {args.method} Student Accuracy: {final_acc:.4f}")
