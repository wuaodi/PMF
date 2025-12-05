#!/usr/bin/env python3
"""
检查SpaceSense数据集中是否有NaN或异常值
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

def check_dataset(data_root):
    """检查数据集中的NaN和异常值"""
    
    print("=" * 80)
    print("SpaceSense数据集完整性检查")
    print("=" * 80)
    
    data_path = Path(data_root)
    issues_found = []
    
    # 检查点云文件
    print("\n检查点云文件 (.bin)...")
    bin_files = list(data_path.rglob("*.bin"))
    for bin_file in tqdm(bin_files, desc="Checking point clouds"):
        try:
            pc = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
            
            if np.isnan(pc).any():
                issues_found.append(f"NaN in point cloud: {bin_file.relative_to(data_path)}")
            
            if np.isinf(pc).any():
                issues_found.append(f"Inf in point cloud: {bin_file.relative_to(data_path)}")
            
            if (np.abs(pc) > 1e6).any():
                issues_found.append(f"Extreme values in point cloud: {bin_file.relative_to(data_path)}")
            
            if len(pc) == 0:
                issues_found.append(f"Empty point cloud: {bin_file.relative_to(data_path)}")
                
        except Exception as e:
            issues_found.append(f"Error reading {bin_file.relative_to(data_path)}: {e}")
    
    # 检查标签文件
    print("\n检查标签文件 (.label)...")
    label_files = list(data_path.rglob("*.label"))
    for label_file in tqdm(label_files, desc="Checking labels"):
        try:
            labels = np.fromfile(label_file, dtype=np.uint32)
            sem_labels = labels & 0xFFFF
            
            if len(labels) == 0:
                issues_found.append(f"Empty labels: {label_file.relative_to(data_path)}")
            
            unique_labels = np.unique(sem_labels)
            if np.any(unique_labels > 6):
                issues_found.append(f"Invalid label values {unique_labels} in: {label_file.relative_to(data_path)}")
                
        except Exception as e:
            issues_found.append(f"Error reading {label_file.relative_to(data_path)}: {e}")
    
    # 检查图像文件
    print("\n检查图像文件 (.png)...")
    png_files = [f for f in data_path.rglob("*.png") if "image_2" in str(f)]
    for png_file in tqdm(png_files, desc="Checking images"):
        try:
            img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
            
            if img is None:
                issues_found.append(f"Failed to load image: {png_file.relative_to(data_path)}")
                continue
            
            if np.isnan(img).any():
                issues_found.append(f"NaN in image: {png_file.relative_to(data_path)}")
            
            if img.size == 0:
                issues_found.append(f"Empty image: {png_file.relative_to(data_path)}")
                
        except Exception as e:
            issues_found.append(f"Error reading {png_file.relative_to(data_path)}: {e}")
    
    # 打印结果
    print("\n" + "=" * 80)
    print("检查完成!")
    print("=" * 80)
    print(f"总文件数: {len(bin_files)} 点云, {len(label_files)} 标签, {len(png_files)} 图像")
    print(f"发现问题: {len(issues_found)}")
    print("=" * 80)
    
    if issues_found:
        print("\n发现以下问题:")
        for i, issue in enumerate(issues_found[:50], 1):  # 最多显示50个
            print(f"{i}. {issue}")
        if len(issues_found) > 50:
            print(f"... 还有 {len(issues_found) - 50} 个问题")
        
        # 保存完整列表到文件
        with open("dataset_issues.txt", "w") as f:
            for issue in issues_found:
                f.write(f"{issue}\n")
        print(f"\n完整问题列表已保存到: dataset_issues.txt")
    else:
        print("\n✅ 未发现数据问题！")
    
    return issues_found

if __name__ == "__main__":
    data_root = "/home/wuaodi/projects/PMF/data/spacesense_minimal_kitti/sequences"
    issues = check_dataset(data_root)
    
    if issues:
        print("\n⚠️  建议: 修复或删除有问题的数据文件")
    else:
        print("\n✅ 数据集完整性良好")

