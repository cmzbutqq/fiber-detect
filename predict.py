#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纤维图像分类预测脚本
加载训练好的模型对单张图像进行预测
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import argparse
import json
import os

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

def load_model(model_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    model = FiberClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names, checkpoint.get('config', {})

def get_transform():
    """获取预测时的图像变换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def predict_image(model, image_path, class_names, transform, device):
    """对单张图像进行预测"""
    try:
        # 加载和预处理图像
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 应用变换
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        # 获取所有类别的概率
        all_probs = {}
        for i, class_name in enumerate(class_names):
            all_probs[class_name] = probabilities[i].item()
        
        return predicted_class, confidence_score, all_probs
        
    except Exception as e:
        print(f"预测图像时出错: {e}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description='纤维图像分类预测')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model', type=str, default='fiber_classifier_resnet50.pth', 
                       help='模型文件路径')
    parser.add_argument('--top_k', type=int, default=3, help='显示前k个预测结果')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.image):
        print(f"错误：图像文件 {args.image} 不存在")
        return
    
    if not os.path.exists(args.model):
        print(f"错误：模型文件 {args.model} 不存在")
        return
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model, class_names, config = load_model(args.model, device)
    
    # 获取变换
    transform = get_transform()
    
    # 预测
    print(f"\n预测图像: {args.image}")
    predicted_class, confidence, all_probs = predict_image(
        model, args.image, class_names, transform, device
    )
    
    if predicted_class is not None:
        print(f"\n预测结果:")
        print(f"类别: {predicted_class}")
        print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # 显示前k个预测结果
        print(f"\n前 {args.top_k} 个预测结果:")
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for i, (class_name, prob) in enumerate(sorted_probs[:args.top_k]):
            print(f"{i+1}. {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    else:
        print("预测失败")

if __name__ == '__main__':
    main()