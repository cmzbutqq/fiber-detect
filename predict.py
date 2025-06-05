import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import argparse
import os
import json

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

def load_model(model_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取类别名称
    class_names = checkpoint.get('class_names', [
        '20CS19764 鼠皮树', 'DH001 粽叶芦', 'JL001 怀槐', 
        'JL005 蒙古栎', 'LiYLXWZW002 长叶水麻', 'XZYGX023 硬头黄'
    ])
    
    # 创建模型
    model = FiberClassifier(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names

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
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        # 获取所有类别的概率
        all_probs = {class_names[i]: probabilities[i].item() 
                    for i in range(len(class_names))}
        
        return predicted_class, confidence_score, all_probs
        
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        return None, None, None

def predict_batch(model, image_dir, class_names, transform, device, output_file=None):
    """对目录中的所有图像进行批量预测"""
    results = []
    
    # 支持的图像格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp')
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print(f"No supported image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images to predict...")
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        predicted_class, confidence, all_probs = predict_image(
            model, img_path, class_names, transform, device
        )
        
        if predicted_class is not None:
            result = {
                'image': img_file,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': all_probs
            }
            results.append(result)
            
            print(f"{img_file}: {predicted_class} (置信度: {confidence:.3f})")
        else:
            print(f"Failed to predict {img_file}")
    
    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='纤维图像分类预测')
    parser.add_argument('--model', type=str, default='best_fiber_classifier.pth',
                       help='模型文件路径')
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--dir', type=str, help='图像目录路径（批量预测）')
    parser.add_argument('--output', type=str, help='结果输出文件路径（JSON格式）')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 设备配置
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Please train the model first by running: python fiber_classifier.py")
        return
    
    # 加载模型
    print("Loading model...")
    model, class_names = load_model(args.model, device)
    transform = get_transform()
    
    print(f"Model loaded successfully. Classes: {class_names}")
    
    # 单张图像预测
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image file not found: {args.image}")
            return
            
        print(f"\nPredicting image: {args.image}")
        predicted_class, confidence, all_probs = predict_image(
            model, args.image, class_names, transform, device
        )
        
        if predicted_class is not None:
            print(f"\n预测结果:")
            print(f"类别: {predicted_class}")
            print(f"置信度: {confidence:.3f}")
            print(f"\n所有类别概率:")
            for class_name, prob in sorted(all_probs.items(), 
                                         key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {prob:.3f}")
        else:
            print("预测失败")
    
    # 批量预测
    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"Directory not found: {args.dir}")
            return
            
        print(f"\nBatch predicting images in: {args.dir}")
        results = predict_batch(
            model, args.dir, class_names, transform, device, args.output
        )
        
        if results:
            # 统计预测结果
            class_counts = {}
            for result in results:
                pred_class = result['predicted_class']
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
            
            print(f"\n预测统计:")
            for class_name, count in sorted(class_counts.items()):
                print(f"  {class_name}: {count} 张图像")
    
    else:
        print("Please specify either --image for single prediction or --dir for batch prediction")
        print("Use --help for more information")

if __name__ == '__main__':
    main()