# 纤维图像分类项目

基于ResNet-50的植物纤维显微镜图像分类系统，支持光镜和电镜图像的统一分类。

## 项目概述

本项目使用深度学习技术对6种植物纤维进行自动分类：
- 20CS19764 鼠皮树
- DH001 粽叶芦
- JL001 怀槐
- JL005 蒙古栎
- LiYLXWZW002 长叶水麻
- XZYGX023 硬头黄

## 数据集结构

```
data/
├── 光镜集(2)/
│   ├── 20CS19764 鼠皮树/
│   ├── DH001 粽叶芦/
│   ├── JL001 怀槐/
│   ├── JL005 蒙古栎/
│   ├── LiYLXWZW002 长叶水麻/
│   └── XZYGX023 硬头黄/
└── 电镜集/
    ├── 20CS19764 鼠皮树/
    ├── DH001 粽叶芦/
    ├── JL001 怀槐/
    ├── JL005 蒙古栎/
    ├── LiYLXWZW002 长叶水麻/
    └── XZYGX023 硬头黄/
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 系统要求

- Python 3.7+
- CUDA支持的GPU（推荐，可选）
- 至少8GB内存

## 使用方法

### 训练模型

```bash
python fiber_classifier.py
```

训练过程将：
1. 自动加载光镜集和电镜集数据
2. 划分训练集和验证集（8:2比例）
3. 使用预训练ResNet-50进行迁移学习
4. 保存最佳模型到 `best_fiber_classifier.pth`
5. 生成训练曲线和混淆矩阵图表
6. 保存详细结果到 `training_results.json`

### 模型特性

- **骨干网络**: 预训练ResNet-50
- **分类头**: 全连接层 (2048 → 512 → 6)
- **数据增强**: 随机翻转、旋转、颜色抖动
- **优化器**: Adam (lr=0.001, weight_decay=1e-4)
- **学习率调度**: StepLR (step_size=10, gamma=0.1)
- **批次大小**: 16
- **训练轮数**: 30

### 输出文件

训练完成后将生成以下文件：

- `best_fiber_classifier.pth`: 最佳模型权重
- `training_history.png`: 训练损失和准确率曲线
- `confusion_matrix.png`: 混淆矩阵热力图
- `training_results.json`: 详细训练结果和分类报告

## 模型架构

```python
FiberClassifier(
  (backbone): ResNet(
    # ResNet-50 layers
    (fc): Sequential(
      (0): Dropout(p=0.5)
      (1): Linear(in_features=2048, out_features=512)
      (2): ReLU()
      (3): Dropout(p=0.3)
      (4): Linear(in_features=512, out_features=6)
    )
  )
)
```

## 数据预处理

### 训练时增强
- 尺寸调整: 256×256 → 随机裁剪224×224
- 随机水平/垂直翻转
- 随机旋转 (±15度)
- 颜色抖动
- ImageNet标准化

### 验证时处理
- 尺寸调整: 224×224
- ImageNet标准化

## 性能监控

训练过程中会显示：
- 实时损失和准确率
- 每个epoch的训练/验证指标
- 最佳模型保存提示

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size (默认16)
   - 减少num_workers

2. **数据加载错误**
   - 检查data目录结构
   - 确认图像文件格式支持

3. **训练速度慢**
   - 确保使用GPU训练
   - 检查数据加载器的num_workers设置

### 调试模式

如需调试，可以修改以下参数：
- 减少num_epochs进行快速测试
- 增加batch_size提高训练效率
- 调整学习率和权重衰减

## 扩展功能

### 自定义配置

可以修改 `fiber_classifier.py` 中的参数：
- 网络架构
- 训练超参数
- 数据增强策略
- 损失函数

### 模型推理

训练完成后，可以使用保存的模型进行单张图像预测：

```python
# 加载模型
checkpoint = torch.load('best_fiber_classifier.pth')
model = FiberClassifier(num_classes=6)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 预测单张图像
# (需要实现预测函数)
```

## 许可证

本项目仅用于学术研究目的。