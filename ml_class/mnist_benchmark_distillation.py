import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

# Teacher CNN (Larger model)
class TeacherCNN(nn.Module):
    def __init__(self):
        super(TeacherCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        features = {}
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        features['conv1'] = x
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        features['conv2'] = x
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        features['conv3'] = x
        
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        features['fc1'] = x
        x = self.dropout2(x)
        
        x = F.relu(self.fc2(x))
        features['fc2'] = x
        
        x = self.fc3(x)
        features['logits'] = x
        
        return x, features

# Student CNN (Smaller model)
class StudentCNN(nn.Module):
    def __init__(self):
        super(StudentCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        features = {}
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        features['conv1'] = x
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        features['conv2'] = x
        
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        features['fc1'] = x
        x = self.dropout2(x)
        
        x = self.fc2(x)
        features['logits'] = x
        
        return x, features

# Knowledge Distillation Loss Functions
class VanillaKDLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super(VanillaKDLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KD loss
        kd_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Student loss
        student_loss = self.ce_loss(student_logits, labels)
        
        return self.alpha * kd_loss + (1 - self.alpha) * student_loss

class FeatureKDLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7, lambda_feat=0.5):
        super(FeatureKDLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.lambda_feat = lambda_feat
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # Feature adaptation layers
        self.adapt_conv1 = nn.Conv2d(16, 64, 1).to(device)
        self.adapt_conv2 = nn.Conv2d(32, 128, 1).to(device)
        self.adapt_fc1 = nn.Linear(64, 512).to(device)
        
    def forward(self, student_logits, teacher_logits, labels, student_features, teacher_features):
        # Vanilla KD loss
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        kd_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Student CE loss
        student_loss = self.ce_loss(student_logits, labels)
        
        # Feature matching loss
        feat_loss = 0
        if 'conv1' in student_features and 'conv1' in teacher_features:
            adapted_student = self.adapt_conv1(student_features['conv1'])
            feat_loss += self.mse_loss(adapted_student, teacher_features['conv1'])
            
        if 'conv2' in student_features and 'conv2' in teacher_features:
            adapted_student = self.adapt_conv2(student_features['conv2'])
            feat_loss += self.mse_loss(adapted_student, teacher_features['conv2'])
            
        if 'fc1' in student_features and 'fc1' in teacher_features:
            adapted_student = self.adapt_fc1(student_features['fc1'])
            feat_loss += self.mse_loss(adapted_student, teacher_features['fc1'])
        
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * student_loss + self.lambda_feat * feat_loss
        return total_loss

class SimKDLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7, lambda_sim=0.5):
        super(SimKDLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.lambda_sim = lambda_sim
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, labels, student_features, teacher_features):
        # Vanilla KD loss
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        kd_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Student CE loss
        student_loss = self.ce_loss(student_logits, labels)
        
        # Similarity loss using cosine similarity
        sim_loss = 0
        count = 0
        for key in ['fc1', 'fc2']:
            if key in student_features and key in teacher_features:
                student_feat = F.normalize(student_features[key], dim=1)
                teacher_feat = F.normalize(teacher_features[key], dim=1)
                
                # Cosine similarity matrix
                student_sim = torch.mm(student_feat, student_feat.t())
                teacher_sim = torch.mm(teacher_feat, teacher_feat.t())
                
                sim_loss += F.mse_loss(student_sim, teacher_sim)
                count += 1
        
        if count > 0:
            sim_loss /= count
        
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * student_loss + self.lambda_sim * sim_loss
        return total_loss

# Training and evaluation functions
def train_model(model, train_loader, criterion, optimizer, device, model_type='student', teacher_model=None, 
                student_features=None, teacher_features=None, epoch=0):
    model.train()
    if teacher_model:
        teacher_model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    forward_time = 0
    backward_time = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Forward pass timing
        start_time = time.time()
        if model_type == 'teacher':
            output, _ = model(data)
            loss = criterion(output, target)
        else:  # student with KD
            student_output, student_feat = model(data)
            if teacher_model:
                with torch.no_grad():
                    teacher_output, teacher_feat = teacher_model(data)
                
                if isinstance(criterion, VanillaKDLoss):
                    loss = criterion(student_output, teacher_output, target)
                else:  # Feature KD or Sim KD
                    loss = criterion(student_output, teacher_output, target, student_feat, teacher_feat)
            else:
                loss = F.cross_entropy(student_output, target)
            output = student_output
        
        forward_time += time.time() - start_time
        
        # Backward pass timing
        start_time = time.time()
        loss.backward()
        optimizer.step()
        backward_time += time.time() - start_time
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    avg_forward_time = forward_time / len(train_loader)
    avg_backward_time = backward_time / len(train_loader)
    
    return avg_loss, accuracy, avg_forward_time, avg_backward_time

def test_model(model, test_loader, device, model_type='student'):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    inference_time = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            start_time = time.time()
            if model_type == 'teacher':
                output, _ = model(data)
            else:
                output, _ = model(data)
            inference_time += time.time() - start_time
            
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= total
    accuracy = 100. * correct / total
    avg_inference_time = inference_time / len(test_loader)
    
    return test_loss, accuracy, avg_inference_time

# Training function for teacher
def train_teacher():
    print("Training Teacher Model...")
    teacher = TeacherCNN().to(device)
    teacher_optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    teacher_criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(3):
        train_loss, train_acc, forward_time, backward_time = train_model(
            teacher, train_loader, teacher_criterion, teacher_optimizer, device, 'teacher'
        )
        test_loss, test_acc, inference_time = test_model(teacher, test_loader, device, 'teacher')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(teacher.state_dict(), 'teacher_best.pth')
        
        print(f'Teacher Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    # Load best model
    teacher.load_state_dict(torch.load('teacher_best.pth'))
    return teacher

# Benchmark function
def run_benchmark():
    results = []
    
    # Train teacher first
    teacher = train_teacher()
    
    # Test configurations for ablation study
    temperatures = [1.0, 4.0, 8.0, 16.0]
    alphas = [0.3, 0.5, 0.7, 0.9]
    lambdas = [0.1, 0.5, 1.0, 2.0]
    
    # Models to test
    models_config = [
        ('Vanilla Student', None, None),
        ('Vanilla KD', VanillaKDLoss, {}),
        ('Feature KD', FeatureKDLoss, {}),
        ('Sim KD', SimKDLoss, {})
    ]
    
    # Base configuration test
    print("\n=== Base Configuration Benchmark ===")
    for model_name, loss_class, loss_kwargs in models_config:
        print(f"\nTraining {model_name}...")
        
        student = StudentCNN().to(device)
        optimizer = optim.Adam(student.parameters(), lr=0.001)
        
        if loss_class:
            criterion = loss_class(**loss_kwargs)
            use_teacher = True
        else:
            criterion = nn.CrossEntropyLoss()
            use_teacher = False
        
        # Training
        train_times = []
        test_accs = []
        
        for epoch in range(3):
            train_loss, train_acc, forward_time, backward_time = train_model(
                student, train_loader, criterion, optimizer, device, 'student',
                teacher if use_teacher else None
            )
            test_loss, test_acc, inference_time = test_model(student, test_loader, device)
            
            train_times.append(forward_time + backward_time)
            test_accs.append(test_acc)
        
        # Final evaluation
        final_test_loss, final_test_acc, final_inference_time = test_model(student, test_loader, device)
        
        result = {
            'Model': model_name,
            'Temperature': 4.0 if loss_class else 'N/A',
            'Alpha': 0.7 if loss_class else 'N/A',
            'Lambda': 0.5 if 'KD' in model_name and model_name != 'Vanilla KD' else 'N/A',
            'Final_Accuracy': final_test_acc,
            'Top1_Accuracy': final_test_acc,  # Same for MNIST
            'Avg_Train_Time': np.mean(train_times),
            'Inference_Time': final_inference_time,
            'Forward_Time': forward_time,
            'Backward_Time': backward_time
        }
        results.append(result)
        
        print(f'{model_name} - Final Accuracy: {final_test_acc:.2f}%')
    
    # Temperature Ablation
    print("\n=== Temperature Ablation Study ===")
    for temp in temperatures:
        print(f"\nTesting Temperature: {temp}")
        
        student = StudentCNN().to(device)
        optimizer = optim.Adam(student.parameters(), lr=0.001)
        criterion = VanillaKDLoss(temperature=temp, alpha=0.7)
        
        for epoch in range(3):  # Reduced epochs for ablation
            train_model(student, train_loader, criterion, optimizer, device, 'student', teacher)
        
        test_loss, test_acc, inference_time = test_model(student, test_loader, device)
        
        result = {
            'Model': f'Vanilla KD (T={temp})',
            'Temperature': temp,
            'Alpha': 0.7,
            'Lambda': 'N/A',
            'Final_Accuracy': test_acc,
            'Top1_Accuracy': test_acc,
            'Avg_Train_Time': 0,  # Not measured in ablation
            'Inference_Time': inference_time,
            'Forward_Time': 0,
            'Backward_Time': 0
        }
        results.append(result)
        
        print(f'Temperature {temp} - Accuracy: {test_acc:.2f}%')
    
    # Alpha Ablation
    print("\n=== Alpha Ablation Study ===")
    for alpha in alphas:
        print(f"\nTesting Alpha: {alpha}")
        
        student = StudentCNN().to(device)
        optimizer = optim.Adam(student.parameters(), lr=0.001)
        criterion = VanillaKDLoss(temperature=4.0, alpha=alpha)
        
        for epoch in range(3):
            train_model(student, train_loader, criterion, optimizer, device, 'student', teacher)
        
        test_loss, test_acc, inference_time = test_model(student, test_loader, device)
        
        result = {
            'Model': f'Vanilla KD (α={alpha})',
            'Temperature': 4.0,
            'Alpha': alpha,
            'Lambda': 'N/A',
            'Final_Accuracy': test_acc,
            'Top1_Accuracy': test_acc,
            'Avg_Train_Time': 0,
            'Inference_Time': inference_time,
            'Forward_Time': 0,
            'Backward_Time': 0
        }
        results.append(result)
        
        print(f'Alpha {alpha} - Accuracy: {test_acc:.2f}%')
    
    # Lambda Ablation (for Feature KD)
    print("\n=== Lambda Ablation Study (Feature KD) ===")
    for lam in lambdas:
        print(f"\nTesting Lambda: {lam}")
        
        student = StudentCNN().to(device)
        optimizer = optim.Adam(student.parameters(), lr=0.001)
        criterion = FeatureKDLoss(temperature=4.0, alpha=0.7, lambda_feat=lam)
        
        for epoch in range(3):
            train_model(student, train_loader, criterion, optimizer, device, 'student', teacher)
        
        test_loss, test_acc, inference_time = test_model(student, test_loader, device)
        
        result = {
            'Model': f'Feature KD (λ={lam})',
            'Temperature': 4.0,
            'Alpha': 0.7,
            'Lambda': lam,
            'Final_Accuracy': test_acc,
            'Top1_Accuracy': test_acc,
            'Avg_Train_Time': 0,
            'Inference_Time': inference_time,
            'Forward_Time': 0,
            'Backward_Time': 0
        }
        results.append(result)
        
        print(f'Lambda {lam} - Accuracy: {test_acc:.2f}%')
    
    return results, teacher

# Run benchmark and create results
print("Starting MNIST Knowledge Distillation Benchmark...")
results, trained_teacher = run_benchmark()

# Create results DataFrame
df_results = pd.DataFrame(results)
print("\n=== BENCHMARK RESULTS ===")
print(df_results.to_string(index=False))

# Save results
df_results.to_csv('mnist_kd_benchmark_results.csv', index=False)

# Create visualizations
plt.figure(figsize=(15, 10))

# Accuracy comparison
plt.subplot(2, 3, 1)
base_models = df_results[df_results['Model'].str.contains(r'^(Vanilla Student|Vanilla KD|Feature KD|Sim KD)$', regex=True)]
plt.bar(base_models['Model'], base_models['Final_Accuracy'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45)

# Temperature ablation
plt.subplot(2, 3, 2)
temp_results = df_results[df_results['Model'].str.contains('T=')]
plt.plot(temp_results['Temperature'], temp_results['Final_Accuracy'], 'bo-')
plt.title('Temperature Ablation')
plt.xlabel('Temperature')
plt.ylabel('Accuracy (%)')

# Alpha ablation
plt.subplot(2, 3, 3)
alpha_results = df_results[df_results['Model'].str.contains('α=')]
plt.plot(alpha_results['Alpha'], alpha_results['Final_Accuracy'], 'ro-')
plt.title('Alpha Ablation')
plt.xlabel('Alpha')
plt.ylabel('Accuracy (%)')

# Lambda ablation
plt.subplot(2, 3, 4)
lambda_results = df_results[df_results['Model'].str.contains('λ=')]
plt.plot(lambda_results['Lambda'], lambda_results['Final_Accuracy'], 'go-')
plt.title('Lambda Ablation (Feature KD)')
plt.xlabel('Lambda')
plt.ylabel('Accuracy (%)')

# Timing comparison
plt.subplot(2, 3, 5)
timing_models = base_models[base_models['Forward_Time'] > 0]
x = range(len(timing_models))
plt.bar([i-0.2 for i in x], timing_models['Forward_Time'], 0.4, label='Forward', alpha=0.7)
plt.bar([i+0.2 for i in x], timing_models['Backward_Time'], 0.4, label='Backward', alpha=0.7)
plt.title('Training Time Comparison')
plt.ylabel('Time (s)')
plt.xticks(x, timing_models['Model'], rotation=45)
plt.legend()

# Inference time
plt.subplot(2, 3, 6)
plt.bar(base_models['Model'], base_models['Inference_Time'])
plt.title('Inference Time Comparison')
plt.ylabel('Time (s)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('mnist_kd_benchmark_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Print model parameter counts
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

teacher_params = count_parameters(TeacherCNN())
student_params = count_parameters(StudentCNN())

print(f"\n=== MODEL COMPLEXITY ===")
print(f"Teacher Model Parameters: {teacher_params:,}")
print(f"Student Model Parameters: {student_params:,}")
print(f"Compression Ratio: {teacher_params/student_params:.2f}x")

print(f"\nBenchmark completed! Results saved to 'mnist_kd_benchmark_results.csv'")
print(f"Plots saved to 'mnist_kd_benchmark_plots.png'")
