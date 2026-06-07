"""
Aluminum Alloy Cylinder High Reflective Metal Surface Data Augmentation Script v2
=====================================================================================
  ① resize from INTER_LINEAR → INTER_CUBIC (preserve details)
  ② Augmentation pipeline A.Resize specified INTER_CUBIC
  ③ Added Unsharp Mask (USM): sharpen before resize to prevent pit loss
"""

import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import random

# ============================================================
# Configuration Section
# ============================================================
CONFIG = {
    "normal_input_dir" : r"D:\MLProject\raw_data\normal",
    "defect_input_dir" : r"D:\MLProject\raw_data\defect",
    "output_root"      : r"D:\MLProject\augmented01",

    "aug_times_normal" : 5,
    "aug_times_defect" : 8,

    "img_size"         : 256,
    "keep_original"    : True,
    "seed"             : 42,

    "sharpen_strength" : 1.2,
    "sharpen_sigma"    : 1.5,
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# ============================================================
# Support Chinese Path Image Reading
# ============================================================
def imread_cn(path):
    try:
        buf   = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"\n  [Read Error] {path}\n  Reason: {e}")
        return None


def imwrite_cn(path, image):
    try:
        encode_ext = ".png"
        ret, buf   = cv2.imencode(encode_ext, image)
        if ret:
            save_path = os.path.splitext(path)[0] + ".png"
            buf.tofile(save_path)
            return True
        else:
            print(f"\n  [Save Error] Encoding failed: {path}")
            return False
    except Exception as e:
        print(f"\n  [Save Error] {path}\n  Reason: {e}")
        return False


# ============================================================
# CLAHE Illumination Normalization
# ============================================================
def apply_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab             = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b         = cv2.split(lab)
    clahe           = cv2.createCLAHE(clipLimit=clip_limit,
                                       tileGridSize=tile_grid_size)
    l_clahe         = clahe.apply(l)
    lab_clahe       = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


# ============================================================
#   Unsharp Mask (USM)
#   Specifically for pits/small defects: sharpen edges before resize
#   Principle: sharpen = original + strength × (original - blurred)
# ============================================================
def apply_unsharp_mask(image_bgr, strength=1.2, sigma=1.5):
    """
    Args:
        strength : Sharpening strength, 1.0~2.0
                   1.0 = slight, 1.5 = obvious, 2.0 = strong
        sigma    : Gaussian blur radius, smaller values sharpen finer details
                   Recommended 1.0~2.0 for pits
    """
    img_f    = image_bgr.astype(np.float32)
    blurred  = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma)
    sharpened = img_f + strength * (img_f - blurred)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened


# ============================================================
# Unified Preprocessing (Sharpen → CLAHE → INTER_CUBIC resize)
# ============================================================
def preprocess_image(image_bgr, img_size,
                     apply_clahe_flag=True,
                     sharpen_strength=1.2,
                     sharpen_sigma=1.5):
    """
    Processing order:
      1. Unsharp Mask (USM)  ← ★ New, enhance details before resize
      2. CLAHE illumination normalization
      3. INTER_CUBIC resize   ← ★ Fix, replaced original INTER_LINEAR
    """
    # Step1: Sharpen first (best effect at original resolution)
    image = apply_unsharp_mask(image_bgr,
                                strength=sharpen_strength,
                                sigma=sharpen_sigma)

    # Step2: CLAHE
    if apply_clahe_flag:
        image = apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))

    #   INTER_CUBIC replaces INTER_LINEAR
    #   INTER_LINEAR : bilinear, fast but blurs details
    #   INTER_CUBIC  : bicubic, preserves high-frequency details (pits, scratches)
    #   INTER_LANCZOS4: highest quality, but slowest (optional)
    image = cv2.resize(image, (img_size, img_size),
                       interpolation=cv2.INTER_CUBIC)   # ← Fix point

    return image


# ============================================================

def get_normal_augmentation_pipeline(img_size):
    """
    Normal sample augmentation pipeline (conservative)
    Fix: Final A.Resize specifies interpolation=cv2.INTER_CUBIC
    """
    return A.Compose([
        # --- Geometric transformations ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(
            limit=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4
        ),
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.85, 1.0),
            ratio=(0.95, 1.05),
            p=0.4
        ),
        # --- Slight illumination perturbation ---
        A.RandomBrightnessContrast(
            brightness_limit=(-0.08, 0.08),
            contrast_limit=(-0.08, 0.08),
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(90, 110), p=0.3),
        # --- Very slight blur (more conservative than before) ---
        # ★ Note: Sharpening added, blur p reduced from 0.2 to 0.1
        A.GaussianBlur(blur_limit=(3, 3), p=0.1),
        # --- Very slight noise ---
        A.GaussNoise(std_range=(0.001, 0.005), p=0.2),
        # --- ★ Fix: Specify INTER_CUBIC ---
        A.Resize(img_size, img_size,
                 interpolation=cv2.INTER_CUBIC),
    ])


def get_defect_augmentation_pipeline(img_size):
    """
    Defect sample augmentation pipeline (more conservative, protect defect integrity)
    Final A.Resize specifies interpolation=cv2.INTER_CUBIC
    """
    return A.Compose([
        # --- Geometric transformations ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(
            limit=8,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3
        ),
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.90, 1.0),
            ratio=(0.97, 1.03),
            p=0.3
        ),
        # --- Illumination perturbation ---
        A.RandomBrightnessContrast(
            brightness_limit=(-0.06, 0.06),
            contrast_limit=(-0.06, 0.06),
            p=0.4
        ),
        A.RandomGamma(gamma_limit=(93, 107), p=0.2),
        # --- ★ Fix: Specify INTER_CUBIC ---
        A.Resize(img_size, img_size,
                 interpolation=cv2.INTER_CUBIC),
    ])


# ============================================================
# Single Image Augmentation
# ============================================================
def augment_image(image_bgr, pipeline, num_augmentations):
    augmented_images = []
    for _ in range(num_augmentations):
        result = pipeline(image=image_bgr)
        augmented_images.append(result["image"])
    return augmented_images


# ============================================================
# Highlight Region Detection
# ============================================================
def detect_highlight_mask(image_bgr, threshold=240):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask


# ============================================================
# Batch Process One Category (unchanged, only pass new parameters)
# ============================================================
def process_category(
    input_dir,
    output_dir,
    pipeline,
    aug_times,
    img_size,
    apply_clahe_flag=True,
    keep_original=True,
    category_name="unknown",
    sharpen_strength=1.2,
    sharpen_sigma=1.5,
):
    os.makedirs(output_dir, exist_ok=True)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    image_paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_ext:
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        print(f"  [Warning] No images found under {input_dir}, skipping.")
        return 0

    expected = len(image_paths) * (aug_times + (1 if keep_original else 0))
    print(f"\n{'='*55}")
    print(f"  [{category_name}]")
    print(f"  Original image count : {len(image_paths)}")
    print(f"  Augmentation factor  : {aug_times}x")
    print(f"  Expected output total: {expected}")
    print(f"  Resize interpolation : INTER_CUBIC ★")
    print(f"  Sharpen strength      : {sharpen_strength} (sigma={sharpen_sigma}) ★")
    print(f"  Output directory      : {output_dir}")
    print(f"{'='*55}")

    aug_count  = 0
    skip_count = 0

    for img_path in tqdm(image_paths, desc=f"  Augmenting [{category_name}]"):
        image = imread_cn(img_path)
        if image is None:
            skip_count += 1
            continue

        # ★ Pass sharpening parameters
        image = preprocess_image(
            image,
            img_size,
            apply_clahe_flag=apply_clahe_flag,
            sharpen_strength=sharpen_strength,
            sharpen_sigma=sharpen_sigma,
        )

        prefix    = category_name.split()[0].lower()
        base_name = f"{prefix}_{str(image_paths.index(img_path)+1).zfill(5)}"

        if keep_original:
            orig_path = os.path.join(output_dir, f"{base_name}_orig.png")
            imwrite_cn(orig_path, image)

        aug_list = augment_image(image, pipeline, aug_times)
        for i, aug_img in enumerate(aug_list):
            save_path = os.path.join(output_dir, f"{base_name}_aug{i+1:03d}.png")
            imwrite_cn(save_path, aug_img)
            aug_count += 1

    print(f"\n  ✅ [{category_name}] Completed!")
    print(f"      Successfully generated : {aug_count} images")
    if skip_count > 0:
        print(f"      Skipped (read failed) : {skip_count}")

    return aug_count


# ============================================================
#   Visualize Sharpness Comparison Before/After
#   Helps intuitively confirm sharpening effect
# ============================================================
def visualize_sharpness_compare(
    input_dir,
    img_size,
    sharpen_strength=1.2,
    sharpen_sigma=1.5,
    save_path="sharpness_compare.png",
    num_samples=3,
):
    """
    Compare three columns: Original | INTER_LINEAR (old) | INTER_CUBIC+USM (new)
    """
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_ext:
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        print("  [Comparison Visualization] No images, skipping.")
        return

    sample_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
    rows = []

    for img_path in sample_paths:
        image = imread_cn(img_path)
        if image is None:
            continue

        # Column 1: Original (only resize, no processing)
        col1 = cv2.resize(image, (img_size, img_size),
                          interpolation=cv2.INTER_LINEAR)

        # Column 2: Old method CLAHE + INTER_LINEAR (blurry)
        col2 = apply_clahe(image)
        col2 = cv2.resize(col2, (img_size, img_size),
                          interpolation=cv2.INTER_LINEAR)

        # Column 3: New method USM + CLAHE + INTER_CUBIC (sharp)
        col3 = preprocess_image(image, img_size,
                                apply_clahe_flag=True,
                                sharpen_strength=sharpen_strength,
                                sharpen_sigma=sharpen_sigma)

        # Add labels
        def add_label(img, text):
            out = img.copy()
            cv2.putText(out, text, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA)
            return out

        col1 = add_label(col1, "Original (LINEAR)")
        col2 = add_label(col2, "Old: CLAHE+LINEAR")
        col3 = add_label(col3, "New: USM+CLAHE+CUBIC")

        row = np.concatenate([col1, col2, col3], axis=1)
        rows.append(row)

    if rows:
        preview = np.concatenate(rows, axis=0)
        imwrite_cn(save_path, preview)
        print(f"  [Comparison Visualization] Sharpness comparison saved: {save_path}")
        print(f"  Left=Original  Middle=Old Method(Blurry)  Right=New Method(Sharp)")


# ============================================================
# Augmentation Effect Visualization (unchanged)
# ============================================================
def visualize_augmentation_samples(
    input_dir, pipeline, img_size,
    num_samples=3, num_aug=4, save_path="preview.png",
    sharpen_strength=1.2, sharpen_sigma=1.5,
):
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_ext:
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        return

    sample_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
    rows = []

    for img_path in sample_paths:
        image = imread_cn(img_path)
        if image is None:
            continue
        image    = preprocess_image(image, img_size,
                                    apply_clahe_flag=True,
                                    sharpen_strength=sharpen_strength,
                                    sharpen_sigma=sharpen_sigma)
        aug_list = augment_image(image, pipeline, num_aug)
        row      = np.concatenate([image] + aug_list, axis=1)
        rows.append(row)

    if rows:
        preview = np.concatenate(rows, axis=0)
        imwrite_cn(save_path, preview)
        print(f"  [Visualization] Preview saved: {save_path}")


# ============================================================
# Highlight Quality Check
# ============================================================
def check_highlight_ratio(image_dir, threshold=240, max_ratio=0.15):
    valid_ext     = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    warning_count = 0
    total_count   = 0

    for root, _, files in os.walk(image_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_ext:
                img_path = os.path.join(root, f)
                image    = imread_cn(img_path)
                if image is None:
                    continue
                mask          = detect_highlight_mask(image, threshold=threshold)
                ratio         = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
                total_count  += 1
                if ratio > max_ratio:
                    warning_count += 1
                    print(f"  [Highlight Warning] {f} | Highlight ratio: {ratio:.2%}")

    print(f"\n  [Highlight Check] Checked {total_count} images, "
          f"{warning_count} exceeded {max_ratio:.0%} threshold.")
    if warning_count == 0:
        print("  [Highlight Check] ✅ All augmented images have normal highlight areas!")
    else:
        print("  [Highlight Check] ⚠️  Please reduce brightness_limit and re-augment.")


# ============================================================
# Main Function
# ============================================================
def main():
    cfg      = CONFIG
    img_size = cfg["img_size"]
    s_str    = cfg["sharpen_strength"]
    s_sig    = cfg["sharpen_sigma"]

    normal_input  = cfg["normal_input_dir"]
    defect_input  = cfg["defect_input_dir"]
    normal_output = os.path.join(cfg["output_root"], "normal")
    defect_output = os.path.join(cfg["output_root"], "defect")

    normal_pipeline = get_normal_augmentation_pipeline(img_size)
    defect_pipeline = get_defect_augmentation_pipeline(img_size)

    print("\n" + "=" * 55)
    print("    Aluminum Alloy High Reflective Metal Surface Data Augmentation Script v2")
    print("    ★ Fix: INTER_CUBIC + Unsharp Mask (USM)")
    print("=" * 55)
    print(f"  Normal input   : {normal_input}")
    print(f"  Defect input   : {defect_input}")
    print(f"  Output root    : {cfg['output_root']}")
    print(f"  Image size     : {img_size} × {img_size}")
    print(f"  Resize         : INTER_CUBIC (was LINEAR) ★")
    print(f"  Sharpen strength: {s_str} sigma={s_sig} ★")
    print("=" * 55)

    # ---------- First generate sharpness comparison to confirm effect ----------
    if os.path.exists(normal_input):
        print("\n  [Pre-check] Generating sharpness comparison (recommended to view before augmentation)...")
        visualize_sharpness_compare(
            input_dir        = normal_input,
            img_size         = img_size,
            sharpen_strength = s_str,
            sharpen_sigma    = s_sig,
            save_path        = os.path.join(cfg["output_root"], "sharpness_compare.png"),
            num_samples      = 3,
        )

    # ---------- Process normal samples ----------
    if os.path.exists(normal_input):
        process_category(
            input_dir        = normal_input,
            output_dir       = normal_output,
            pipeline         = normal_pipeline,
            aug_times        = cfg["aug_times_normal"],
            img_size         = img_size,
            apply_clahe_flag = True,
            keep_original    = cfg["keep_original"],
            category_name    = "normal samples",
            sharpen_strength = s_str,
            sharpen_sigma    = s_sig,
        )
        visualize_augmentation_samples(
            input_dir        = normal_input,
            pipeline         = normal_pipeline,
            img_size         = img_size,
            num_samples      = 3,
            num_aug          = 4,
            save_path        = os.path.join(cfg["output_root"], "preview_normal.png"),
            sharpen_strength = s_str,
            sharpen_sigma    = s_sig,
        )
    else:
        print(f"\n[Error] Normal directory does not exist: {normal_input}")

    # ---------- Process defect samples ----------
    if os.path.exists(defect_input):
        process_category(
            input_dir        = defect_input,
            output_dir       = defect_output,
            pipeline         = defect_pipeline,
            aug_times        = cfg["aug_times_defect"],
            img_size         = img_size,
            apply_clahe_flag = True,
            keep_original    = cfg["keep_original"],
            category_name    = "defect samples",
            sharpen_strength = s_str,
            sharpen_sigma    = s_sig,
        )
        visualize_augmentation_samples(
            input_dir        = defect_input,
            pipeline         = defect_pipeline,
            img_size         = img_size,
            num_samples      = 3,
            num_aug          = 4,
            save_path        = os.path.join(cfg["output_root"], "preview_defect.png"),
            sharpen_strength = s_str,
            sharpen_sigma    = s_sig,
        )
    else:
        print(f"\n[Error] Defect directory does not exist: {defect_input}")

    # ---------- Highlight quality check ----------
    print("\n\n[Quality Verification] Checking highlight ratio in augmented images...")
    print("\n  >> Normal augmented images check:")
    check_highlight_ratio(normal_output, threshold=240, max_ratio=0.15)
    print("\n  >> Defect augmented images check:")
    check_highlight_ratio(defect_output, threshold=240, max_ratio=0.15)

    # ---------- Completion summary ----------
    print("\n" + "=" * 55)
    print("  ✅ All augmentation tasks completed!")
    print(f"  Sharpness comparison → {cfg['output_root']}\\sharpness_compare.png")
    print(f"  Normal augmented images → {normal_output}")
    print(f"  Defect augmented images → {defect_output}")
    print("=" * 55)


if __name__ == "__main__":
    main()