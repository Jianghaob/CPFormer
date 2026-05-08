#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计HU18数据集中每个类别的样本数
"""
import os
import numpy as np
import scipy.io as sio
import h5py

# HU18数据集的类别名称
CLASS_NAMES = [
    "Healthy grass",
    "Stressed grass",
    "Synthetic grass",
    "Evergreen trees",
    "Deciduous trees",
    "Soil",
    "Water",
    "Residential buildings",
    "Non-residential buildings",
    "Roads",
    "Sidewalks",
    "Crosswalks",
    "Major thoroughfares",
    "Highways",
    "Railways",
    "Paved parking lots",
    "Unpaved parking lots",
    "Cars",
    "Trains",
    "Stadium seats"
]


def load_hu18_labels(data_path):
    """
    加载HU18数据集的标签文件
    返回: labels数组 (H, W)
    """
    label_mat_path = os.path.join(data_path, 'HoustonU_gt.mat')
    
    # 读取标签
    try:
        labels = sio.loadmat(label_mat_path)['houstonU_gt']
    except NotImplementedError:
        # v7.3 情况，使用 h5py 读取 HDF5 结构
        with h5py.File(label_mat_path, "r") as f_hu18_gt:
            if 'houstonU_gt' in f_hu18_gt.keys():
                labels = f_hu18_gt['houstonU_gt'][:]
            elif 'HoustonU_gt' in f_hu18_gt.keys():
                labels = f_hu18_gt['HoustonU_gt'][:]
            else:
                first_key = list(f_hu18_gt.keys())[0]
                labels = f_hu18_gt[first_key][:]
    
    # 统一标签形状：去掉多余维度，统一为 (H, W)
    labels = np.array(labels)
    if labels.ndim > 2:
        labels = np.squeeze(labels)
    
    return labels


def count_samples_by_class(labels, class_names):
    """
    统计每个类别的样本数
    
    参数:
        labels: 标签数组 (H, W)
        class_names: 类别名称列表
    
    返回:
        class_counts: 字典，{类别ID: 样本数}
        total_valid: 总的有效样本数（排除类别0）
    """
    labels_flat = labels.reshape(-1)
    
    # 获取所有类别ID（排除0）
    unique_classes = np.unique(labels_flat)
    unique_classes = unique_classes[unique_classes > 0]  # 排除背景类别0
    
    # 统计每个类别的样本数
    class_counts = {}
    for cls_id in unique_classes:
        count = np.sum(labels_flat == cls_id)
        class_counts[int(cls_id)] = count
    
    # 计算总的有效样本数（排除类别0）
    total_valid = np.sum(labels_flat > 0)
    
    return class_counts, total_valid


def print_statistics(class_counts, total_valid, class_names):
    """
    打印统计结果
    """
    print("=" * 80)
    print("HU18数据集样本统计")
    print("=" * 80)
    print()
    
    # 打印每个类别的样本数
    print("各类别样本数:")
    print("-" * 80)
    print(f"{'类别ID':<8} {'类别名称':<30} {'样本数':<12} {'占比':<10}")
    print("-" * 80)
    
    total_samples = sum(class_counts.values())
    
    for cls_id in sorted(class_counts.keys()):
        count = class_counts[cls_id]
        percentage = (count / total_valid * 100) if total_valid > 0 else 0
        
        # 获取类别名称（类别ID从1开始，索引从0开始）
        if 1 <= cls_id <= len(class_names):
            class_name = class_names[cls_id - 1]
        else:
            class_name = f"Unknown Class {cls_id}"
        
        print(f"{cls_id:<8} {class_name:<30} {count:<12} {percentage:>6.2f}%")
    
    print("-" * 80)
    print(f"{'总计':<8} {'有效样本总数':<30} {total_valid:<12} {'100.00%':<10}")
    print()
    
    # 打印背景样本数（类别0）
    labels_flat = None  # 需要重新加载来计算背景
    print("=" * 80)
    print("其他统计信息:")
    print("-" * 80)
    print(f"有效样本数（类别1-{max(class_counts.keys())}）: {total_valid:,}")
    print(f"类别总数: {len(class_counts)}")
    print(f"最小类别样本数: {min(class_counts.values()):,}")
    print(f"最大类别样本数: {max(class_counts.values()):,}")
    print(f"平均类别样本数: {np.mean(list(class_counts.values())):,.0f}")
    print("=" * 80)


def main():
    """
    主函数
    """
    # 获取数据路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data')
    
    # 如果data目录不存在，尝试使用绝对路径
    if not os.path.exists(data_path):
        data_path = os.path.join(os.getcwd(), 'data')
    
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据目录: {data_path}")
        print("请确保数据文件位于项目根目录下的 data/ 文件夹中")
        return
    
    label_file = os.path.join(data_path, 'HoustonU_gt.mat')
    if not os.path.exists(label_file):
        print(f"错误: 找不到标签文件: {label_file}")
        return
    
    print(f"正在加载标签文件: {label_file}")
    print()
    
    # 加载标签
    try:
        labels = load_hu18_labels(data_path)
        print(f"标签形状: {labels.shape}")
        print()
    except Exception as e:
        print(f"错误: 加载标签文件失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 统计样本数
    class_counts, total_valid = count_samples_by_class(labels, CLASS_NAMES)
    
    # 打印统计结果
    print_statistics(class_counts, total_valid, CLASS_NAMES)
    
    # 额外信息：背景样本数
    labels_flat = labels.reshape(-1)
    background_count = np.sum(labels_flat == 0)
    total_pixels = labels_flat.size
    print()
    print("=" * 80)
    print("背景样本统计:")
    print("-" * 80)
    print(f"背景样本数（类别0）: {background_count:,}")
    print(f"总像素数: {total_pixels:,}")
    print(f"背景占比: {background_count / total_pixels * 100:.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()

