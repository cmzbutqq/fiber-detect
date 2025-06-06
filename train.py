#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纤维图像分类训练脚本
使用ResNet-50进行6类植物纤维分类
数据集：optical + electron 文件夹
类别：鼠皮树、粽叶芦、怀槐、蒙古栎、长叶水麻、硬头黄
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import time
from tqdm import tqdm

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class FiberDataset(Dataset):
    """纤维图像数据集类"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            # 处理不同格式的图像
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个黑色图像作为备用
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_dataset(data_dir, microscope_type='both'):
    """加载数据集
    
    Args:
        data_dir: 数据目录路径
        microscope_type: 显微镜类型 ('optical', 'electron', 'both')
    """
    # 定义类别映射
    class_names = [
        '20CS19764 鼠皮树',
        'DH001 粽叶芦', 
        'JL001 怀槐',
        'JL005 蒙古栎',
        'LiYLXWZW002 长叶水麻',
        'XZYGX023 硬头黄'
    ]
    
    image_paths = []
    labels = []
    
    # 支持的图像格式
    supported_formats = ('.tiff', '.tif', '.png', '.jpg', '.jpeg')
    
    # 根据microscope_type确定要处理的文件夹
    if microscope_type == 'both':
        folders = ['optical', 'electron']
    elif microscope_type in ['optical', 'electron']:
        folders = [microscope_type]
    else:
        raise ValueError("microscope_type must be 'optical', 'electron', or 'both'")
    
    print(f"加载 {microscope_type} 显微镜数据...")
    
    # 遍历指定的文件夹
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist")
            continue
            
        print(f"处理文件夹: {folder}")
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(folder_path, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist")
                continue
                
            # 遍历类别文件夹中的所有图像
            for item in os.listdir(class_dir):
                item_path = os.path.join(class_dir, item)
                
                # 跳过子文件夹
                if os.path.isdir(item_path):
                    continue
                    
                # 检查文件格式
                if item.lower().endswith(supported_formats):
                    image_paths.append(item_path)
                    labels.append(class_idx)
    
    print(f"总共加载了 {len(image_paths)} 张图像")
    
    # 统计每个类别的图像数量
    class_counts = Counter(labels)
    for class_idx, class_name in enumerate(class_names):
        count = class_counts.get(class_idx, 0)
        print(f"{class_name}: {count} 张图像")
    
    return image_paths, labels, class_names

def get_transforms():
    """定义数据预处理和增强"""
    
    # 训练时的数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])
    
    # 验证和测试时的预处理
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class FiberClassifier(nn.Module):
    """基于ResNet-50的纤维分类器"""
    
    def __init__(self, num_classes=6, pretrained=True):
        super(FiberClassifier, self).__init__()
        
        # 加载预训练的ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # 替换最后的全连接层
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_labels

def plot_training_history(train_losses, train_accs, val_losses, val_accs, microscope_type=''):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title(f'Training and Validation Loss - {microscope_type.capitalize()}' if microscope_type else 'Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_title(f'Training and Validation Accuracy - {microscope_type.capitalize()}' if microscope_type else 'Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    filename = f'training_history_{microscope_type}.png' if microscope_type else 'training_history.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, microscope_type=''):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    title = f'Confusion Matrix - {microscope_type.capitalize()}' if microscope_type else 'Confusion Matrix'
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    filename = f'confusion_matrix_{microscope_type}.png' if microscope_type else 'confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def train_single_model(config, microscope_type):
    """训练单个模型
    
    Args:
        config: 配置参数
        microscope_type: 显微镜类型 ('optical' 或 'electron')
    """
    print(f"\n{'='*60}")
    print(f"开始训练 {microscope_type.upper()} 显微镜模型")
    print(f"{'='*60}")
    
    # 检查设备 - 强制使用CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用！请确保已安装支持CUDA的PyTorch版本并且系统有可用的GPU。")
    
    device = torch.device('cuda')
    print(f"使用设备: {device}")
    print(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载指定类型的数据集
    print(f"\n加载 {microscope_type} 数据集...")
    image_paths, labels, class_names = load_dataset(config['data_dir'], microscope_type)
    
    if len(image_paths) == 0:
        print(f"错误：没有找到任何 {microscope_type} 图像文件！")
        return
    
    # 划分数据集
    print("\n划分数据集...")
    # 首先划分出测试集
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, 
        test_size=config['test_size'], 
        random_state=42, 
        stratify=labels
    )
    
    # 再从训练验证集中划分出验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=config['val_size']/(1-config['test_size']),
        random_state=42,
        stratify=train_val_labels
    )
    
    print(f"训练集: {len(train_paths)} 张图像")
    print(f"验证集: {len(val_paths)} 张图像")
    print(f"测试集: {len(test_paths)} 张图像")
    
    # 获取数据变换
    train_transform, val_transform = get_transforms()
    
    # 创建数据集
    train_dataset = FiberDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = FiberDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = FiberDataset(test_paths, test_labels, transform=val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建模型
    print("\n创建模型...")
    model = FiberClassifier(num_classes=len(class_names))
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['learning_rate'], 
                          weight_decay=config['weight_decay'])
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # 训练历史记录
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 40)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc, val_predictions, val_true = validate_epoch(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
    
    training_time = time.time() - start_time
    print(f"\n训练完成！总用时: {training_time/60:.2f} 分钟")
    
    # 加载最佳模型进行测试
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 在测试集上评估
    print("\n在测试集上评估...")
    test_loss, test_acc, test_predictions, test_true = validate_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # 生成分类报告
    print("\n分类报告:")
    print(classification_report(test_true, test_predictions, target_names=class_names))
    
    # 保存模型
    if config['save_model']:
        model_filename = f"fiber_classifier_resnet50_{microscope_type}.pth"
        torch.save({
            'model_state_dict': best_model_state,
            'class_names': class_names,
            'config': config,
            'microscope_type': microscope_type,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc
        }, model_filename)
        print(f"\n模型已保存为: {model_filename}")
    
    # 绘制训练历史
    print("\n绘制训练历史...")
    plot_training_history(train_losses, train_accs, val_losses, val_accs, microscope_type)
    
    # 绘制混淆矩阵
    print("绘制混淆矩阵...")
    plot_confusion_matrix(test_true, test_predictions, class_names, microscope_type)
    
    print(f"\n{microscope_type.upper()} 模型训练完成！")
    return test_acc

def main():
    """主训练函数"""
    # 首先检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用！请确保已安装支持CUDA的PyTorch版本并且系统有可用的GPU。")
    
    print("=" * 60)
    print("纤维图像分类训练开始 - 分别训练光镜和电镜模型")
    print("=" * 60)
    print(f"检测到CUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)
    
    # 配置参数
    config = {
        'data_dir': './data',
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'test_size': 0.2,
        'val_size': 0.2,
        'num_workers': 4,
        'save_model': True,
        'model_name': 'fiber_classifier_resnet50.pth'
    }
    
    # 保存配置
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 训练结果记录
    results = {}
    
    # 训练光镜模型
    print("\n开始训练光镜(Optical)模型...")
    optical_acc = train_single_model(config, 'optical')
    results['optical'] = optical_acc
    
    # 训练电镜模型
    print("\n开始训练电镜(Electron)模型...")
    electron_acc = train_single_model(config, 'electron')
    results['electron'] = electron_acc
    
    # 输出最终结果
    print("\n" + "=" * 60)
    print("所有模型训练完成！")
    print("=" * 60)
    print(f"光镜模型测试准确率: {results['optical']:.2f}%")
    print(f"电镜模型测试准确率: {results['electron']:.2f}%")
    print("\n模型文件:")
    print("- fiber_classifier_resnet50_optical.pth")
    print("- fiber_classifier_resnet50_electron.pth")
    print("\n图表文件:")
    print("- training_history_optical.png")
    print("- training_history_electron.png")
    print("- confusion_matrix_optical.png")
    print("- confusion_matrix_electron.png")
    
    # 保存训练结果摘要
    summary = {
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': config,
        'results': results,
        'model_files': {
            'optical': 'fiber_classifier_resnet50_optical.pth',
            'electron': 'fiber_classifier_resnet50_electron.pth'
        }
    }
    
    with open('training_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n训练摘要已保存为: training_summary.json")

if __name__ == '__main__':
    main()