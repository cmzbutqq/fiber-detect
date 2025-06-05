# 纤维分类器配置文件

# 数据配置
DATA_CONFIG = {
    'data_dir': 'data',
    'test_size': 0.2,  # 验证集比例
    'random_state': 42,
    'supported_formats': ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
}

# 模型配置
MODEL_CONFIG = {
    'backbone': 'resnet50',  # 骨干网络
    'pretrained': True,      # 是否使用预训练权重
    'num_classes': 6,        # 分类数量
    'dropout_rate': 0.5,     # Dropout率
    'hidden_dim': 512        # 隐藏层维度
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 16,
    'num_epochs': 30,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'num_workers': 4,
    'pin_memory': True
}

# 优化器配置
OPTIMIZER_CONFIG = {
    'type': 'adam',  # 'adam', 'sgd', 'adamw'
    'momentum': 0.9,  # 仅用于SGD
    'betas': (0.9, 0.999),  # 仅用于Adam/AdamW
}

# 学习率调度器配置
SCHEDULER_CONFIG = {
    'type': 'step',  # 'step', 'cosine', 'plateau'
    'step_size': 10,  # StepLR参数
    'gamma': 0.1,     # StepLR参数
    'T_max': 30,      # CosineAnnealingLR参数
    'patience': 5,    # ReduceLROnPlateau参数
    'factor': 0.5     # ReduceLROnPlateau参数
}

# 数据增强配置
AUGMENTATION_CONFIG = {
    'resize_size': 256,
    'crop_size': 224,
    'horizontal_flip_prob': 0.5,
    'vertical_flip_prob': 0.5,
    'rotation_degrees': 15,
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    },
    'normalize': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}

# 保存配置
SAVE_CONFIG = {
    'model_save_path': 'best_fiber_classifier.pth',
    'results_save_path': 'training_results.json',
    'plots_dir': './',
    'save_every_n_epochs': 5  # 每N个epoch保存一次检查点
}

# 设备配置
DEVICE_CONFIG = {
    'use_cuda': True,
    'cuda_device': 0,  # GPU设备ID
    'mixed_precision': False  # 是否使用混合精度训练
}

# 日志配置
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'training.log',
    'console_output': True
}

# 早停配置
EARLY_STOPPING_CONFIG = {
    'enabled': True,
    'patience': 10,
    'min_delta': 0.001,
    'monitor': 'val_acc'  # 'val_acc' 或 'val_loss'
}

# 类别名称
CLASS_NAMES = [
    '20CS19764 鼠皮树',
    'DH001 粽叶芦', 
    'JL001 怀槐',
    'JL005 蒙古栎',
    'LiYLXWZW002 长叶水麻',
    'XZYGX023 硬头黄'
]

# 高级配置
ADVANCED_CONFIG = {
    'gradient_clipping': {
        'enabled': False,
        'max_norm': 1.0
    },
    'label_smoothing': {
        'enabled': False,
        'smoothing': 0.1
    },
    'mixup': {
        'enabled': False,
        'alpha': 0.2
    },
    'cutmix': {
        'enabled': False,
        'alpha': 1.0
    }
}