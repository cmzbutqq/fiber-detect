import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter, defaultdict
import json
from tqdm import tqdm
import pandas as pd

def analyze_dataset(data_dir):
    """
    分析数据集的基本信息
    """
    classes = ['20CS19764 鼠皮树', 'DH001 粽叶芦', 'JL001 怀槐', 
               'JL005 蒙古栎', 'LiYLXWZW002 长叶水麻', 'XZYGX023 硬头黄']
    
    analysis_results = {
        'total_images': 0,
        'class_distribution': {},
        'image_sizes': [],
        'file_formats': Counter(),
        'microscope_types': {'optical': 0, 'electron': 0},
        'magnifications': defaultdict(int),
        'class_details': {}
    }
    
    # 分析光镜集
    optical_dir = os.path.join(data_dir, '光镜集(2)')
    if os.path.exists(optical_dir):
        print("Analyzing optical microscope images...")
        analyze_microscope_type(optical_dir, 'optical', classes, analysis_results)
    
    # 分析电镜集
    electron_dir = os.path.join(data_dir, '电镜集')
    if os.path.exists(electron_dir):
        print("Analyzing electron microscope images...")
        analyze_microscope_type(electron_dir, 'electron', classes, analysis_results)
    
    return analysis_results

def analyze_microscope_type(microscope_dir, microscope_type, classes, analysis_results):
    """
    分析特定显微镜类型的图像
    """
    for class_name in classes:
        class_dir = os.path.join(microscope_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        class_images = []
        class_sizes = []
        class_formats = Counter()
        class_magnifications = defaultdict(int)
        
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    # 获取图像信息
                    with Image.open(img_path) as img:
                        width, height = img.size
                        class_sizes.append((width, height))
                        analysis_results['image_sizes'].append((width, height))
                    
                    # 文件格式统计
                    file_ext = os.path.splitext(img_file)[1].lower()
                    class_formats[file_ext] += 1
                    analysis_results['file_formats'][file_ext] += 1
                    
                    # 提取放大倍率信息
                    magnification = extract_magnification(img_file)
                    if magnification:
                        class_magnifications[magnification] += 1
                        analysis_results['magnifications'][magnification] += 1
                    
                    class_images.append(img_file)
                    analysis_results['total_images'] += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # 更新类别统计
        if class_name not in analysis_results['class_distribution']:
            analysis_results['class_distribution'][class_name] = 0
        analysis_results['class_distribution'][class_name] += len(class_images)
        
        # 更新显微镜类型统计
        analysis_results['microscope_types'][microscope_type] += len(class_images)
        
        # 保存类别详细信息
        analysis_results['class_details'][f"{class_name}_{microscope_type}"] = {
            'count': len(class_images),
            'sizes': class_sizes,
            'formats': dict(class_formats),
            'magnifications': dict(class_magnifications)
        }

def extract_magnification(filename):
    """
    从文件名中提取放大倍率信息
    """
    import re
    
    # 常见的放大倍率模式
    patterns = [
        r'(\d+)x',  # 200x, 400x等
        r'-(\d+)-',  # -200-, -400-等
        r'(\d{3,4})(?!\d)',  # 200, 400, 600等（3-4位数字）
    ]
    
    filename_lower = filename.lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, filename_lower)
        if matches:
            # 过滤合理的放大倍率范围
            for match in matches:
                mag = int(match)
                if 50 <= mag <= 10000:  # 合理的放大倍率范围
                    return f"{mag}x"
    
    return None

def plot_class_distribution(analysis_results):
    """
    绘制类别分布图
    """
    class_dist = analysis_results['class_distribution']
    
    plt.figure(figsize=(12, 6))
    
    # 条形图
    plt.subplot(1, 2, 1)
    classes = list(class_dist.keys())
    counts = list(class_dist.values())
    
    bars = plt.bar(range(len(classes)), counts, color='skyblue', alpha=0.7)
    plt.xlabel('植物纤维类别')
    plt.ylabel('图像数量')
    plt.title('各类别图像数量分布')
    plt.xticks(range(len(classes)), [c.split(' ')[1] for c in classes], rotation=45)
    
    # 在条形图上添加数值标签
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    # 饼图
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=[c.split(' ')[1] for c in classes], autopct='%1.1f%%', startangle=90)
    plt.title('各类别图像比例分布')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_microscope_distribution(analysis_results):
    """
    绘制显微镜类型分布图
    """
    microscope_types = analysis_results['microscope_types']
    
    plt.figure(figsize=(10, 5))
    
    # 显微镜类型分布
    plt.subplot(1, 2, 1)
    types = list(microscope_types.keys())
    counts = list(microscope_types.values())
    
    plt.bar(types, counts, color=['lightcoral', 'lightblue'])
    plt.xlabel('显微镜类型')
    plt.ylabel('图像数量')
    plt.title('显微镜类型分布')
    
    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha='center', va='bottom')
    
    # 放大倍率分布
    plt.subplot(1, 2, 2)
    magnifications = analysis_results['magnifications']
    if magnifications:
        mags = list(magnifications.keys())
        mag_counts = list(magnifications.values())
        
        plt.bar(range(len(mags)), mag_counts, color='lightgreen')
        plt.xlabel('放大倍率')
        plt.ylabel('图像数量')
        plt.title('放大倍率分布')
        plt.xticks(range(len(mags)), mags, rotation=45)
        
        for i, count in enumerate(mag_counts):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('microscope_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_image_size_distribution(analysis_results):
    """
    绘制图像尺寸分布图
    """
    sizes = analysis_results['image_sizes']
    if not sizes:
        print("No image size data available")
        return
    
    widths = [size[0] for size in sizes]
    heights = [size[1] for size in sizes]
    
    plt.figure(figsize=(15, 5))
    
    # 宽度分布
    plt.subplot(1, 3, 1)
    plt.hist(widths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('图像宽度 (像素)')
    plt.ylabel('频次')
    plt.title('图像宽度分布')
    plt.grid(True, alpha=0.3)
    
    # 高度分布
    plt.subplot(1, 3, 2)
    plt.hist(heights, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('图像高度 (像素)')
    plt.ylabel('频次')
    plt.title('图像高度分布')
    plt.grid(True, alpha=0.3)
    
    # 宽高比分布
    plt.subplot(1, 3, 3)
    aspect_ratios = [w/h for w, h in sizes]
    plt.hist(aspect_ratios, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('宽高比')
    plt.ylabel('频次')
    plt.title('图像宽高比分布')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('image_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(analysis_results):
    """
    生成数据集摘要报告
    """
    report = []
    report.append("=" * 60)
    report.append("纤维图像数据集分析报告")
    report.append("=" * 60)
    
    # 基本统计
    report.append(f"\n总图像数量: {analysis_results['total_images']}")
    report.append(f"类别数量: {len(analysis_results['class_distribution'])}")
    
    # 类别分布
    report.append("\n类别分布:")
    for class_name, count in analysis_results['class_distribution'].items():
        percentage = (count / analysis_results['total_images']) * 100
        report.append(f"  {class_name}: {count} 张 ({percentage:.1f}%)")
    
    # 显微镜类型分布
    report.append("\n显微镜类型分布:")
    for mic_type, count in analysis_results['microscope_types'].items():
        percentage = (count / analysis_results['total_images']) * 100
        report.append(f"  {mic_type}: {count} 张 ({percentage:.1f}%)")
    
    # 文件格式分布
    report.append("\n文件格式分布:")
    for format_type, count in analysis_results['file_formats'].items():
        percentage = (count / analysis_results['total_images']) * 100
        report.append(f"  {format_type}: {count} 张 ({percentage:.1f}%)")
    
    # 放大倍率分布
    if analysis_results['magnifications']:
        report.append("\n放大倍率分布:")
        for mag, count in sorted(analysis_results['magnifications'].items()):
            percentage = (count / analysis_results['total_images']) * 100
            report.append(f"  {mag}: {count} 张 ({percentage:.1f}%)")
    
    # 图像尺寸统计
    if analysis_results['image_sizes']:
        sizes = analysis_results['image_sizes']
        widths = [size[0] for size in sizes]
        heights = [size[1] for size in sizes]
        
        report.append("\n图像尺寸统计:")
        report.append(f"  宽度范围: {min(widths)} - {max(widths)} 像素")
        report.append(f"  高度范围: {min(heights)} - {max(heights)} 像素")
        report.append(f"  平均宽度: {np.mean(widths):.1f} 像素")
        report.append(f"  平均高度: {np.mean(heights):.1f} 像素")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)

def save_analysis_results(analysis_results, output_file='dataset_analysis.json'):
    """
    保存分析结果到JSON文件
    """
    # 转换不可序列化的对象
    serializable_results = {
        'total_images': analysis_results['total_images'],
        'class_distribution': analysis_results['class_distribution'],
        'file_formats': dict(analysis_results['file_formats']),
        'microscope_types': analysis_results['microscope_types'],
        'magnifications': dict(analysis_results['magnifications']),
        'class_details': {}
    }
    
    # 处理类别详细信息
    for class_key, details in analysis_results['class_details'].items():
        serializable_results['class_details'][class_key] = {
            'count': details['count'],
            'formats': details['formats'],
            'magnifications': details['magnifications'],
            'size_stats': {
                'count': len(details['sizes']),
                'avg_width': np.mean([s[0] for s in details['sizes']]) if details['sizes'] else 0,
                'avg_height': np.mean([s[1] for s in details['sizes']]) if details['sizes'] else 0
            }
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"Analysis results saved to {output_file}")

def main():
    """
    主函数
    """
    data_dir = 'data'
    
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found!")
        print("Please make sure the data directory exists and contains the image datasets.")
        return
    
    print("Starting dataset analysis...")
    
    # 分析数据集
    analysis_results = analyze_dataset(data_dir)
    
    if analysis_results['total_images'] == 0:
        print("No images found in the dataset!")
        return
    
    # 生成并打印摘要报告
    summary_report = generate_summary_report(analysis_results)
    print(summary_report)
    
    # 保存摘要报告
    with open('dataset_summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(summary_report)
    
    # 保存详细分析结果
    save_analysis_results(analysis_results)
    
    # 生成可视化图表
    print("\nGenerating visualization plots...")
    plot_class_distribution(analysis_results)
    plot_microscope_distribution(analysis_results)
    plot_image_size_distribution(analysis_results)
    
    print("\nDataset analysis completed!")
    print("Generated files:")
    print("  - dataset_summary_report.txt")
    print("  - dataset_analysis.json")
    print("  - class_distribution.png")
    print("  - microscope_distribution.png")
    print("  - image_size_distribution.png")

if __name__ == '__main__':
    main()