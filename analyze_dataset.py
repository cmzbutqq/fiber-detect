#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集分析脚本
分析纤维图像数据集的统计信息和分布
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

def analyze_dataset(data_dir):
    """分析数据集"""
    
    # 定义类别
    class_names = [
        '20CS19764 鼠皮树',
        'DH001 粽叶芦', 
        'JL001 怀槐',
        'JL005 蒙古栎',
        'LiYLXWZW002 长叶水麻',
        'XZYGX023 硬头黄'
    ]
    
    # 支持的图像格式
    supported_formats = ('.tiff', '.tif', '.png', '.jpg', '.jpeg')
    
    # 统计信息
    stats = {
        'total_images': 0,
        'optical_images': 0,
        'electron_images': 0,
        'class_distribution': defaultdict(lambda: {'optical': 0, 'electron': 0, 'total': 0}),
        'image_formats': Counter(),
        'image_sizes': [],
        'file_sizes': [],
        'errors': []
    }
    
    print("分析数据集...")
    print("=" * 50)
    
    # 遍历optical和electron文件夹
    for folder in ['optical', 'electron']:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            print(f"警告: {folder_path} 不存在")
            continue
            
        print(f"\n分析 {folder} 文件夹...")
        
        for class_name in class_names:
            class_dir = os.path.join(folder_path, class_name)
            if not os.path.exists(class_dir):
                print(f"警告: {class_dir} 不存在")
                continue
                
            # 遍历类别文件夹中的所有文件
            files = [f for f in os.listdir(class_dir) 
                    if os.path.isfile(os.path.join(class_dir, f))]
            
            for filename in tqdm(files, desc=f"{folder}/{class_name}"):
                file_path = os.path.join(class_dir, filename)
                
                # 检查文件格式
                if filename.lower().endswith(supported_formats):
                    try:
                        # 统计图像信息
                        stats['total_images'] += 1
                        if folder == 'optical':
                            stats['optical_images'] += 1
                        else:
                            stats['electron_images'] += 1
                        
                        stats['class_distribution'][class_name][folder] += 1
                        stats['class_distribution'][class_name]['total'] += 1
                        
                        # 文件格式统计
                        ext = os.path.splitext(filename)[1].lower()
                        stats['image_formats'][ext] += 1
                        
                        # 文件大小
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        stats['file_sizes'].append(file_size)
                        
                        # 图像尺寸（采样部分图像以节省时间）
                        if len(stats['image_sizes']) < 100 or np.random.random() < 0.1:
                            try:
                                with Image.open(file_path) as img:
                                    stats['image_sizes'].append(img.size)
                            except Exception as e:
                                stats['errors'].append(f"无法读取图像 {file_path}: {e}")
                        
                    except Exception as e:
                        stats['errors'].append(f"处理文件 {file_path} 时出错: {e}")
    
    return stats, class_names

def print_statistics(stats, class_names):
    """打印统计信息"""
    print("\n" + "=" * 50)
    print("数据集统计信息")
    print("=" * 50)
    
    print(f"总图像数量: {stats['total_images']}")
    print(f"光镜图像: {stats['optical_images']}")
    print(f"电镜图像: {stats['electron_images']}")
    
    print("\n类别分布:")
    print("-" * 30)
    for class_name in class_names:
        optical_count = stats['class_distribution'][class_name]['optical']
        electron_count = stats['class_distribution'][class_name]['electron']
        total_count = stats['class_distribution'][class_name]['total']
        print(f"{class_name}:")
        print(f"  光镜: {optical_count}, 电镜: {electron_count}, 总计: {total_count}")
    
    print("\n文件格式分布:")
    print("-" * 20)
    for ext, count in stats['image_formats'].items():
        percentage = (count / stats['total_images']) * 100
        print(f"{ext}: {count} ({percentage:.1f}%)")
    
    if stats['file_sizes']:
        print("\n文件大小统计 (MB):")
        print("-" * 20)
        file_sizes = np.array(stats['file_sizes'])
        print(f"平均: {np.mean(file_sizes):.2f}")
        print(f"中位数: {np.median(file_sizes):.2f}")
        print(f"最小: {np.min(file_sizes):.2f}")
        print(f"最大: {np.max(file_sizes):.2f}")
    
    if stats['image_sizes']:
        print("\n图像尺寸统计:")
        print("-" * 15)
        widths = [size[0] for size in stats['image_sizes']]
        heights = [size[1] for size in stats['image_sizes']]
        print(f"宽度 - 平均: {np.mean(widths):.0f}, 范围: {np.min(widths)}-{np.max(widths)}")
        print(f"高度 - 平均: {np.mean(heights):.0f}, 范围: {np.min(heights)}-{np.max(heights)}")
        
        # 统计常见尺寸
        size_counter = Counter(stats['image_sizes'])
        print("\n最常见的图像尺寸:")
        for size, count in size_counter.most_common(5):
            print(f"  {size[0]}x{size[1]}: {count} 张")
    
    if stats['errors']:
        print(f"\n错误数量: {len(stats['errors'])}")
        if len(stats['errors']) <= 10:
            print("错误详情:")
            for error in stats['errors']:
                print(f"  {error}")
        else:
            print("前10个错误:")
            for error in stats['errors'][:10]:
                print(f"  {error}")

def plot_visualizations(stats, class_names):
    """绘制可视化图表"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 类别分布条形图
    ax1 = plt.subplot(2, 3, 1)
    class_totals = [stats['class_distribution'][name]['total'] for name in class_names]
    optical_counts = [stats['class_distribution'][name]['optical'] for name in class_names]
    electron_counts = [stats['class_distribution'][name]['electron'] for name in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, optical_counts, width, label='光镜', alpha=0.8)
    bars2 = ax1.bar(x + width/2, electron_counts, width, label='电镜', alpha=0.8)
    
    ax1.set_xlabel('植物类别')
    ax1.set_ylabel('图像数量')
    ax1.set_title('各类别图像数量分布')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.split(' ')[1] for name in class_names], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 在条形图上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 2. 总体分布饼图
    ax2 = plt.subplot(2, 3, 2)
    sizes = [stats['optical_images'], stats['electron_images']]
    labels = ['光镜图像', '电镜图像']
    colors = ['lightblue', 'lightcoral']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('光镜vs电镜图像分布')
    
    # 3. 文件格式分布
    ax3 = plt.subplot(2, 3, 3)
    formats = list(stats['image_formats'].keys())
    counts = list(stats['image_formats'].values())
    
    ax3.bar(formats, counts, color='lightgreen', alpha=0.8)
    ax3.set_xlabel('文件格式')
    ax3.set_ylabel('数量')
    ax3.set_title('文件格式分布')
    ax3.grid(True, alpha=0.3)
    
    # 4. 文件大小分布
    if stats['file_sizes']:
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(stats['file_sizes'], bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('文件大小 (MB)')
        ax4.set_ylabel('频次')
        ax4.set_title('文件大小分布')
        ax4.grid(True, alpha=0.3)
    
    # 5. 图像尺寸散点图
    if stats['image_sizes']:
        ax5 = plt.subplot(2, 3, 5)
        widths = [size[0] for size in stats['image_sizes']]
        heights = [size[1] for size in stats['image_sizes']]
        ax5.scatter(widths, heights, alpha=0.6, color='purple')
        ax5.set_xlabel('宽度 (像素)')
        ax5.set_ylabel('高度 (像素)')
        ax5.set_title('图像尺寸分布')
        ax5.grid(True, alpha=0.3)
    
    # 6. 类别不平衡可视化
    ax6 = plt.subplot(2, 3, 6)
    class_totals_sorted = sorted([(name.split(' ')[1], total) for name, total in 
                                 zip(class_names, class_totals)], key=lambda x: x[1])
    names_sorted, totals_sorted = zip(*class_totals_sorted)
    
    bars = ax6.barh(names_sorted, totals_sorted, color='skyblue', alpha=0.8)
    ax6.set_xlabel('图像数量')
    ax6.set_title('类别数量排序')
    ax6.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax6.text(width, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_analysis_report(stats, class_names, output_file='dataset_analysis_report.json'):
    """保存分析报告为JSON文件"""
    
    # 准备报告数据
    report = {
        'summary': {
            'total_images': stats['total_images'],
            'optical_images': stats['optical_images'],
            'electron_images': stats['electron_images'],
            'num_classes': len(class_names),
            'class_names': class_names
        },
        'class_distribution': dict(stats['class_distribution']),
        'file_formats': dict(stats['image_formats']),
        'file_size_stats': {},
        'image_size_stats': {},
        'errors': stats['errors']
    }
    
    # 文件大小统计
    if stats['file_sizes']:
        file_sizes = np.array(stats['file_sizes'])
        report['file_size_stats'] = {
            'mean_mb': float(np.mean(file_sizes)),
            'median_mb': float(np.median(file_sizes)),
            'min_mb': float(np.min(file_sizes)),
            'max_mb': float(np.max(file_sizes)),
            'std_mb': float(np.std(file_sizes))
        }
    
    # 图像尺寸统计
    if stats['image_sizes']:
        widths = [size[0] for size in stats['image_sizes']]
        heights = [size[1] for size in stats['image_sizes']]
        size_counter = Counter(stats['image_sizes'])
        
        report['image_size_stats'] = {
            'width_stats': {
                'mean': float(np.mean(widths)),
                'min': int(np.min(widths)),
                'max': int(np.max(widths))
            },
            'height_stats': {
                'mean': float(np.mean(heights)),
                'min': int(np.min(heights)),
                'max': int(np.max(heights))
            },
            'common_sizes': [(f"{size[0]}x{size[1]}", count) 
                           for size, count in size_counter.most_common(10)]
        }
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n分析报告已保存为: {output_file}")

def main():
    """主函数"""
    data_dir = './data'
    
    if not os.path.exists(data_dir):
        print(f"错误：数据目录 {data_dir} 不存在")
        return
    
    # 分析数据集
    stats, class_names = analyze_dataset(data_dir)
    
    # 打印统计信息
    print_statistics(stats, class_names)
    
    # 绘制可视化图表
    print("\n生成可视化图表...")
    plot_visualizations(stats, class_names)
    
    # 保存分析报告
    save_analysis_report(stats, class_names)
    
    print("\n数据集分析完成！")

if __name__ == '__main__':
    main()