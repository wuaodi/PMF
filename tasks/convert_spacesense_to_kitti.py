#!/usr/bin/env python3
"""
将SpaceSense数据转换为Semantic-KITTI格式
"""
import os
import shutil
from pathlib import Path

def create_default_calib():
    """创建真实的calib.txt文件"""
    calib_content = """P2: 1097.98754330 0 512.0 0 0 1097.98754330 512.0 0 0 0 1 0
Tr: 0 1 0 0 0 0 1 0 1 0 0 0
"""
    return calib_content

def convert_spacesense_to_kitti(source_root, target_root):
    """
    将SpaceSense数据转换为Semantic-KITTI格式
    每颗卫星对应一个sequence，该卫星的所有轨迹数据合并
    
    Args:
        source_root: SpaceSense数据根目录
        target_root: 输出目录（将创建sequences文件夹）
    """
    source_root = Path(source_root)
    target_root = Path(target_root)
    
    # 找到所有卫星文件夹（如Gaia等）
    satellite_dirs = sorted([d for d in source_root.iterdir() if d.is_dir()])
    print(f"找到 {len(satellite_dirs)} 颗卫星")
    
    # 创建sequences目录
    sequences_dir = target_root / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每颗卫星创建一个sequence
    for seq_idx, satellite_dir in enumerate(satellite_dirs):
        seq_id = f"{seq_idx:02d}"
        print(f"\n处理 sequence {seq_id}: {satellite_dir.name}")
        
        # 创建sequence目录结构
        seq_dir = sequences_dir / seq_id
        velodyne_dir = seq_dir / "velodyne"
        labels_dir = seq_dir / "labels"
        image_dir = seq_dir / "image_2"
        
        velodyne_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建calib.txt
        calib_path = seq_dir / "calib.txt"
        with open(calib_path, 'w') as f:
            f.write(create_default_calib())
        
        # 获取该卫星下的所有轨迹文件夹
        trajectory_dirs = sorted([d for d in satellite_dir.iterdir() if d.is_dir()])
        print(f"  找到 {len(trajectory_dirs)} 条轨迹")
        
        frame_counter = 0
        
        # 遍历每条轨迹
        for traj_dir in trajectory_dirs:
            print(f"  处理轨迹: {traj_dir.name}")
            
            src_velodyne = traj_dir / "velodyne"
            src_labels = traj_dir / "labels"
            src_images = traj_dir / "image_2"
            
            if not src_velodyne.exists():
                print(f"    警告: {src_velodyne} 不存在，跳过")
                continue
            
            # 获取bin文件列表并排序（按文件名）
            bin_files = sorted(src_velodyne.glob("*.bin"))
            print(f"    {len(bin_files)} 帧")
            
            # 复制并重命名文件
            for bin_file in bin_files:
                new_name = f"{frame_counter:06d}"
                timestamp = bin_file.stem
                
                # 复制velodyne文件
                shutil.copy2(bin_file, velodyne_dir / f"{new_name}.bin")
                
                # 复制label文件
                label_file = src_labels / f"{timestamp}.label"
                if label_file.exists():
                    shutil.copy2(label_file, labels_dir / f"{new_name}.label")
                
                # 复制image文件
                image_file = src_images / f"{timestamp}.png"
                if image_file.exists():
                    shutil.copy2(image_file, image_dir / f"{new_name}.png")
                
                frame_counter += 1
        
        print(f"  完成: 共转换了 {frame_counter} 帧数据")
    
    print(f"\n转换完成! 数据已保存到: {sequences_dir}")
    print(f"共创建了 {len(satellite_dirs)} 个sequences (00-{len(satellite_dirs)-1:02d})")

if __name__ == "__main__":
    # 设置路径
    source_root = "/home/wuaodi/projects/PMF/data/spacesense_minimal"
    target_root = "/home/wuaodi/projects/PMF/data/spacesense_kitti"
    
    print("SpaceSense 到 Semantic-KITTI 格式转换工具")
    print("=" * 50)
    print(f"源目录: {source_root}")
    print(f"目标目录: {target_root}")
    print("=" * 50)
    
    convert_spacesense_to_kitti(source_root, target_root)

