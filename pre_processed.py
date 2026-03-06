
import cv2
import numpy as np
import os
import shutil
from pathlib import Path

# =====================================================
#         你只需要修改这里的参数！
# =====================================================

# 你那些原始铝制品图片所在的文件夹（散落的图片请先用collect脚本汇总到这里）
INPUT_FOLDER = r"D:\raw_data"

# 处理后的图片保存路径（会自动创建）
# ours_dataset  --> 送进 Anomalib "你们的方法" 用的数据集
OUTPUT_FOLDER = r"D:\machine learning\dataset"

# 圆形掩膜缩放比例：0.95 = 切掉最外圈5%的强反光边缘（建议范围 0.90 ~ 1.00）
MASK_SHRINK_RATIO = 0.95

# CLAHE 对比度限制：越大对比度越强（建议范围 2.0 ~ 4.0）
CLAHE_CLIP_LIMIT = 2.5

# CLAHE 分块大小：越小细节越丰富（一般不用改）
CLAHE_TILE_GRID = (8, 8)

# =====================================================


def extract_cylinder_mask(gray_img, shrink_ratio=0.95):
    """
    从灰度图中提取铝制圆柱体的圆形掩膜（Mask）。
    原理：找到图中面积最大的轮廓，拟合最小包围圆，生成 ROI。
    """
    # 高斯模糊：消除图片本身的噪点和细小反光点的干扰
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 2)

    # Otsu 自适应二值化：自动找到最佳阈值，把铝制品和背景分开
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学闭运算：填补铝制品表面因高光导致的"白色空洞"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 寻找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    # 选面积最大的轮廓（也就是铝制品本身）
    max_contour = max(contours, key=cv2.contourArea)

    # 拟合最小包围圆
    (cx, cy), radius = cv2.minEnclosingCircle(max_contour)

    # 稍微缩小半径，切掉边缘的强反光圈（核心创新点之一）
    radius = int(radius * shrink_ratio)
    center = (int(cx), int(cy))

    # 生成圆形掩膜
    mask = np.zeros_like(gray_img)
    cv2.circle(mask, center, radius, 255, thickness=-1)

    return mask, center, radius


def apply_clahe_enhancement(bgr_img, mask, clip_limit=2.5, tile_grid=(8, 8)):
    """
    在 LAB 色彩空间中，仅对亮度通道（L）进行 CLAHE 均衡化。
    优点：不改变颜色，只让光照均匀，把暗处的划痕变得更清晰。
    """
    # 先用掩膜把背景抠掉，避免背景干扰 CLAHE 计算
    roi_img = cv2.bitwise_and(bgr_img, bgr_img, mask=mask)

    # 转换到 LAB 色彩空间
    lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 创建 CLAHE 对象，只对亮度通道 L 进行均衡化（核心创新点之二）
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l)

    # 合并通道并转回 BGR
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced_img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 再次用掩膜压制背景（CLAHE处理后背景可能出现微弱灰色，这里清干净）
    final_img = cv2.bitwise_and(enhanced_img, enhanced_img, mask=mask)

    return final_img


def process_single_image(image_path, output_path):
    """
    处理单张图片的完整流水线：
    原图 -> 圆形ROI掩膜（去背景+去边缘强反光）-> CLAHE均衡化（抗反光） -> 保存
    """
    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [跳过] 无法读取图片: {image_path.name}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 步骤1：提取圆形掩膜
    mask, center, radius = extract_cylinder_mask(gray, shrink_ratio=MASK_SHRINK_RATIO)

    if mask is None:
        print(f"  [警告] 未能找到产品轮廓，将直接保存原图: {image_path.name}")
        # 找不到轮廓时，退化为：只做 CLAHE，不裁圆
        mask = np.ones_like(gray) * 255
        center, radius = None, None

    # 步骤2：CLAHE 抗反光增强
    final_img = apply_clahe_enhancement(img, mask,
                                         clip_limit=CLAHE_CLIP_LIMIT,
                                         tile_grid=CLAHE_TILE_GRID)

    # 保存处理后的图片
    cv2.imwrite(str(output_path), final_img)
    return True


def batch_process(input_dir, output_dir):
    """
    批量处理主函数：
    遍历 input_dir 下所有图片（包括子文件夹），处理后保存到 output_dir。
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"[错误] 输入文件夹不存在: {input_path.absolute()}")
        print("请先把你的图片放入该文件夹，或修改脚本顶部的 INPUT_FOLDER 路径。")
        return

    # 自动创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    # 递归搜索所有图片
    all_images = [f for f in input_path.rglob('*')
                  if f.is_file() and f.suffix.lower() in valid_extensions]

    if not all_images:
        print(f"[警告] 在 {input_path} 中没有找到任何图片文件！")
        return

    print(f"共找到 {len(all_images)} 张图片，开始处理...\n")
    print("-" * 50)

    success_count = 0
    fail_count = 0

    for i, img_path in enumerate(all_images, 1):
        # 输出文件名：保留原文件名，前面加编号方便整理
        out_name = f"{i:04d}_{img_path.name}"
        out_path = output_path / out_name

        print(f"[{i}/{len(all_images)}] 正在处理: {img_path.name}")

        if process_single_image(img_path, out_path):
            success_count += 1
        else:
            fail_count += 1

    print("-" * 50)
    print(f"\n✅ 全部处理完成！")
    print(f"   成功: {success_count} 张")
    print(f"   失败/跳过: {fail_count} 张")
    print(f"\n📁 处理后的图片保存在:\n   {output_path.absolute()}")
    print(f"\n下一步：把 [{output_path.name}] 文件夹作为 '你们的方法' 数据集送入 Anomalib！")


if __name__ == "__main__":
    batch_process(INPUT_FOLDER, OUTPUT_FOLDER)
