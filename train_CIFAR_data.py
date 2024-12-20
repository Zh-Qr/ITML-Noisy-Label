import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet34
import pandas as pd
from torch.utils.data import DataLoader
import os

print("Environment setup is done")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据集的保存路径
data_path = 'autodl-tmp/CIFAR-100N'

# 数据变换
transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 使用 torchvision 下载 CIFAR-100 数据集
train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms_train)
test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transforms_test)

# 加载噪声标签文件
noise_file = torch.load('autodl-tmp/CIFAR-100N/cifar-100-python/CIFAR-100_human.pt')
noise_labels = {key: noise_file[key] for key in noise_file.keys()}
clean_label = noise_file['clean_label']

# 定义 ResNet 模型
def initialize_model():
    model = resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)  # CIFAR-100 类别数
    return model.to(device)

# 定义损失函数和优化器
def initialize_optimizer(model):
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return optimizer, scheduler

# 训练函数
def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets).mean()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# ITLM 训练函数
def ITLM_train(model, train_loader, criterion, optimizer, prune_ratio):
    model.train()
    prune_losses = []
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets).mean()
        prune_losses.append(loss.item())
    
    num_prune = int(prune_ratio * len(prune_losses))
    prune_indices = sorted(range(len(prune_losses)), key=lambda i: prune_losses[i], reverse=True)[:num_prune]
    
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx not in prune_indices:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets).mean()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / (len(train_loader) - num_prune)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# 测试函数
def test(model, test_loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets).mean()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# 文件夹创建检查
os.makedirs('log', exist_ok=True)
os.makedirs('weight', exist_ok=True)

# 主训练循环
batch_size = 256
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

for key, noise_label in noise_labels.items():
    if key=='noisy_label' :
#         print(f"-----------------直接训练开始，noise标签为 {key}-----------------")
#         noisy_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms_train)
#         noisy_dataset.targets = noise_label
#         train_loader_noisy = DataLoader(noisy_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

#         model = initialize_model()
#         optimizer, scheduler = initialize_optimizer(model)
#         criterion = nn.CrossEntropyLoss(reduction='none')

#         train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []

#         # 直接训练
#         for epoch in range(100):
#             train_loss, train_accuracy = train(model, train_loader_noisy, criterion, optimizer)
#             test_loss, test_accuracy = test(model, test_loader, criterion)
#             scheduler.step()

#             train_losses.append(train_loss)
#             train_accuracies.append(train_accuracy)
#             test_losses.append(test_loss)
#             test_accuracies.append(test_accuracy)

#             print(f'Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
#                   f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
#             if train_accuracy > 99.999:
#                 break

#         # 保存训练日志
#         pd.DataFrame({
#             'Epoch': range(1, len(train_losses) + 1),
#             'Train Loss': train_losses,
#             'Train Accuracy': train_accuracies,
#             'Test Loss': test_losses,
#             'Test Accuracy': test_accuracies
#         }).to_csv(f'log/raw_{key}.csv', index=False)

#         torch.save(model.state_dict(), f'weight/resnet34_cifar100_raw_{key}.pth')
#         print(f"训练过程已保存到 'log/raw_{key}.csv'，模型权重保存为 'weight/resnet34_cifar100_raw_{key}.pth'")

        # ITLM 训练
        print(f"-----------------ITLM训练开始，noise标签为 {key}-----------------")
        prune_ratio = sum(1 for i in range(len(noise_label)) if noise_label[i] != clean_label[i]) / len(noise_label)
        noisy_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms_train)
        noisy_dataset.targets = noise_label
        train_loader_noisy = DataLoader(noisy_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        model = initialize_model()
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer, scheduler = initialize_optimizer(model)

        train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []

        for epoch in range(100):
            train_loss, train_accuracy = ITLM_train(model, train_loader_noisy, criterion, optimizer, prune_ratio*0.5)
            test_loss, test_accuracy = test(model, test_loader, criterion)
            scheduler.step()

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            print(f'Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
            if train_accuracy > 99.999:
                break

        # 保存 ITLM 训练日志
        pd.DataFrame({
            'Epoch': range(1, len(train_losses) + 1),
            'Train Loss': train_losses,
            'Train Accuracy': train_accuracies,
            'Test Loss': test_losses,
            'Test Accuracy': test_accuracies
        }).to_csv(f'log/ITLM_{key}.csv', index=False)

        torch.save(model.state_dict(), f'weight/resnet34_cifar100_ITLM_{key}.pth')
        print(f"ITLM 训练过程已保存到 'log/ITLM_{key}.csv'，模型权重保存为 'weight/resnet34_cifar100_ITLM_{key}.pth'")
