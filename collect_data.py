
import os
import shutil
from pathlib import Path

def collect_and_rename_images(source_root, target_folder):
    """
    遍历 source_root 下的所有子文件夹，提取所有图片，
    并按顺序重新命名放入 target_folder 中。
    """
    source_path = Path(source_root)
    target_path = Path(target_folder)
    
    # 如果目标文件夹不存在，自动创建
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图片后缀名 (可以根据你的实际情况添加)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    print(f"正在扫描文件夹: {source_path.absolute()} ...")
    
    count = 1
    # rglob('*') 会递归遍历该文件夹及其所有子文件夹下的所有文件
    for file_path in source_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            # 构造新的文件名: 比如 0001.jpg, 0002.png
            new_filename = f"{count:04d}{file_path.suffix.lower()}"
            new_filepath = target_path / new_filename
            
            # 复制文件
            shutil.copy2(file_path, new_filepath)
            print(f"提取成功: {file_path.name}  --->  {new_filename}")
            count += 1
            
    print("-" * 30)
    print(f"🎉 提取完毕！一共找到了 {count - 1} 张图片。")
    print(f"它们现在全都整齐地躺在: {target_path.absolute()}")

if __name__ == '__main__':
    # ================= 修改这里 =================
    # 填入你那个包含各种子文件夹的“总文件夹”路径
    SOURCE_DIRECTORY = r"D:\data" 
    
    # 你想把提取出来的图片集中存放到哪里？
    TARGET_DIRECTORY = r"D:\raw_data"
    # ============================================
    
    collect_and_rename_images(SOURCE_DIRECTORY, TARGET_DIRECTORY)
