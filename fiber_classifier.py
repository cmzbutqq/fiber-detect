import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
from tqdm import tqdm

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
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个黑色图像作为备用
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_fiber_data(data_dir):
    """
    加载纤维图像数据
    将光镜集和电镜集统一处理
    """
    classes = ['20CS19764 鼠皮树', 'DH001 粽叶芦', 'JL001 怀槐', 
               'JL005 蒙古栎', 'LiYLXWZW002 长叶水麻', 'XZYGX023 硬头黄']
    
    image_paths = []
    labels = []
    
    # 处理光镜集
    light_dir = os.path.join(data_dir, '光镜集(2)')
    if os.path.exists(light_dir):
        print("Loading optical microscope images...")
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(light_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(class_dir, img_file))
                        labels.append(class_idx)
    
    # 处理电镜集
    electron_dir = os.path.join(data_dir, '电镜集')
    if os.path.exists(electron_dir):
        print("Loading electron microscope images...")
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(electron_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(class_dir, img_file))
                        labels.append(class_idx)
    
    print(f"Total images loaded: {len(image_paths)}")
    print(f"Class distribution: {Counter(labels)}")
    
    return image_paths, labels, classes

class FiberClassifier(nn.Module):
    """基于ResNet-50的纤维分类器"""
    def __init__(self, num_classes=6, pretrained=True):
        super(FiberClassifier, self).__init__()
        
        # 加载预训练的ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # 替换分类头
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

def get_transforms():
    """获取数据预处理变换"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

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
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_labels

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(train_losses, train_accs, val_losses, val_accs, 
                y_true, y_pred, class_names):
    """保存训练结果"""
    results = {
        'train_losses': train_losses,
        'train_accuracies': train_accs,
        'val_losses': val_losses,
        'val_accuracies': val_accs,
        'classification_report': classification_report(y_true, y_pred, 
                                                     target_names=class_names, 
                                                     output_dict=True)
    }
    
    with open('training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def main():
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据路径
    data_dir = 'data'
    
    # 加载数据
    print("Loading data...")
    image_paths, labels, class_names = load_fiber_data(data_dir)
    
    if len(image_paths) == 0:
        print("No images found! Please check the data directory.")
        return
    
    # 划分训练集和验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # 获取数据变换
    train_transform, val_transform = get_transforms()
    
    # 创建数据集
    train_dataset = FiberDataset(train_paths, train_labels, train_transform)
    val_dataset = FiberDataset(val_paths, val_labels, val_transform)
    
    # 创建数据加载器
    batch_size = 16  # 根据GPU内存调整
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # 创建模型
    model = FiberClassifier(num_classes=len(class_names))
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 训练参数
    num_epochs = 30
    best_val_acc = 0.0
    
    # 记录训练历史
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
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
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'class_names': class_names
            }, 'best_fiber_classifier.pth')
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # 绘制训练历史
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(val_true, val_predictions, class_names)
    
    # 保存结果
    save_results(train_losses, train_accs, val_losses, val_accs, 
                val_true, val_predictions, class_names)
    
    print("\nTraining results saved to 'training_results.json'")
    print("Model saved to 'best_fiber_classifier.pth'")
    print("Plots saved as 'training_history.png' and 'confusion_matrix.png'")

if __name__ == '__main__':
    main()