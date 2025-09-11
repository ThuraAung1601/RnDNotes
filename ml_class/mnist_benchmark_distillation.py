import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ----------------------------
# Simple CNN for Student Model
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ----------------------------
# Knowledge Distillation Loss
# ----------------------------
def kd_loss_fn(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7):
    """
    Combine hard-label CE loss and soft-label KD loss
    """
    kd_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(student_logits / temperature, dim=1),
                                                  F.softmax(teacher_logits / temperature, dim=1)) * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss

# ----------------------------
# Training Function
# ----------------------------
def train_model(model, dataloader, optimizer, criterion, device, epochs, teacher=None, alpha=0.7, temperature=4.0):
    start_time = time.time()
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # If teacher is provided → KD loss
            if teacher is not None:
                with torch.no_grad():
                    teacher_outputs = teacher(inputs)
                loss = kd_loss_fn(outputs, teacher_outputs, labels, temperature, alpha)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(dataloader):.4f} Acc: {acc:.2f}%")

    elapsed = time.time() - start_time
    return acc, elapsed

# ----------------------------
# Evaluation
# ----------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

# ----------------------------
# Parameter Count
# ----------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Teacher/Student/KD models on MNIST")
    parser.add_argument("--method", type=str, choices=["student", "teacher", "kd"], default="student",
                        help="Training method: student / teacher / kd")
    parser.add_argument("--teacher-epochs", type=int, default=5, help="Epochs for teacher training")
    parser.add_argument("--student-epochs", type=int, default=5, help="Epochs for student training")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=4.0, help="KD temperature")
    parser.add_argument("--alpha", type=float, default=0.7, help="KD loss weight for soft targets")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Teacher: Pretrained ResNet18
    teacher_model = models.resnet18(weights=None)
    teacher_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    teacher_model.fc = nn.Linear(512, 10)

    # Student: Small CNN
    student_model = SimpleCNN(num_classes=10)

    criterion = nn.CrossEntropyLoss()

    print(f"\nMethod Selected: {args.method}")
    print("-" * 50)

    if args.method == "teacher":
        print("Training TEACHER only...")
        optimizer = optim.Adam(teacher_model.parameters(), lr=args.lr)
        teacher_acc, teacher_time = train_model(teacher_model, train_loader, optimizer, criterion, device, args.teacher_epochs)
        final_acc = evaluate_model(teacher_model, test_loader, device)
        print(f"\nTeacher Params: {count_params(teacher_model):,}")
        print(f"Training Time: {teacher_time:.2f}s")
        print(f"Final Teacher Accuracy: {final_acc:.2f}%")

    elif args.method == "student":
        print("Training STUDENT only...")
        optimizer = optim.Adam(student_model.parameters(), lr=args.lr)
        student_acc, student_time = train_model(student_model, train_loader, optimizer, criterion, device, args.student_epochs)
        final_acc = evaluate_model(student_model, test_loader, device)
        print(f"\nStudent Params: {count_params(student_model):,}")
        print(f"Training Time: {student_time:.2f}s")
        print(f"Final Student Accuracy: {final_acc:.2f}%")

    elif args.method == "kd":
        print("Training TEACHER first...")
        optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=args.lr)
        teacher_acc, teacher_time = train_model(teacher_model, train_loader, optimizer_teacher, criterion, device, args.teacher_epochs)
        final_teacher_acc = evaluate_model(teacher_model, test_loader, device)

        print(f"\nTeacher Params: {count_params(teacher_model):,}")
        print(f"Teacher Training Time: {teacher_time:.2f}s")
        print(f"Final Teacher Accuracy: {final_teacher_acc:.2f}%")

        print("\nTraining STUDENT with Knowledge Distillation...")
        optimizer_student = optim.Adam(student_model.parameters(), lr=args.lr)
        student_acc, student_time = train_model(student_model, train_loader, optimizer_student, criterion, device,
                                                args.student_epochs, teacher=teacher_model,
                                                alpha=args.alpha, temperature=args.temperature)
        final_student_acc = evaluate_model(student_model, test_loader, device)

        print(f"\nStudent Params: {count_params(student_model):,}")
        print(f"Student Training Time: {student_time:.2f}s")
        print(f"Final Student Accuracy: {final_student_acc:.2f}%")

if __name__ == "__main__":
    main()
