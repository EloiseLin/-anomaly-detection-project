"""
Preprocessing Ablation Study
Verify the contribution of each component: CLAHE + USM + BICUBIC
This corresponds to Table 1 in the paper

Fix: Patchcore (lowercase c) in anomalib 2.5.0
"""

import os
import cv2
import time
import torch
import inspect
import warnings
import numpy as np
from PIL import Image
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    # ★ raw_data:
    #   raw_data/
    #     train/good/       ← Good samples for training
    #     test/good/        ← Good samples for testing
    #     test/defect/      ← Defective samples for testing
    "dataset_root" : r"D:\MLProject\RawData\raw",
    "dataset_name" : "raw",
    "output_dir"   : r"D:\MLProject\results\ablation",
    "image_size"   : 256,
    "num_workers"  : 4,
}

# ============================================================
# Preprocessing Components
# ============================================================
def apply_clahe(image_np):
    """CLAHE local contrast enhancement (RGB input/output)"""
    lab          = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b      = cv2.split(lab)
    clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(l)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_usm(image_np, strength=1.2, sigma=1.5):
    """Unsharp Mask (RGB input/output)"""
    img_f     = image_np.astype(np.float32)
    blurred   = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma)
    sharpened = np.clip(img_f + strength * (img_f - blurred), 0, 255)
    return sharpened.astype(np.uint8)


# ============================================================
# 4 Preprocessing Schemes
# ============================================================
ABLATION_CONFIGS = [
    {
        "name"         : "No Preprocessing (Baseline)",
        "use_clahe"    : False,
        "use_usm"      : False,
        "interpolation": cv2.INTER_LINEAR,
        "interp_name"  : "LINEAR",
    },
    {
        "name"         : "CLAHE Only",
        "use_clahe"    : True,
        "use_usm"      : False,
        "interpolation": cv2.INTER_LINEAR,
        "interp_name"  : "LINEAR",
    },
    {
        "name"         : "USM + CUBIC Only",
        "use_clahe"    : False,
        "use_usm"      : True,
        "interpolation": cv2.INTER_CUBIC,
        "interp_name"  : "CUBIC",
    },
    {
        "name"         : "Full Pipeline (CLAHE + USM + CUBIC)",
        "use_clahe"    : True,
        "use_usm"      : True,
        "interpolation": cv2.INTER_CUBIC,
        "interp_name"  : "CUBIC",
    },
]

# ============================================================
# Custom Preprocessing Transform
# ============================================================
class CustomPreprocess:
    def __init__(self, img_size, use_clahe, use_usm, interpolation):
        self.img_size      = img_size
        self.use_clahe     = use_clahe
        "use_usm"         = use_usm
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img_np = np.array(img.convert("RGB")).astype(np.uint8)

        if self.use_usm:
            img_np = apply_usm(img_np, strength=1.2, sigma=1.5)

        if self.use_clahe:
            img_np = apply_clahe(img_np)

        img_np = cv2.resize(img_np,
                            (self.img_size, self.img_size),
                            interpolation=self.interpolation)

        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        mean  = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std   = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (img_t - mean) / std


# ============================================================
# Build DataModule (Compatible with anomalib 2.5.0)
# ============================================================
def build_datamodule(cfg, preprocess):
    from anomalib.data import Folder

    sig    = inspect.signature(Folder.__init__)
    params = set(sig.parameters.keys())

    print(f"  [DataModule] Folder supported parameters: {[p for p in params if p != 'self']}")

    kwargs = {
        "name"             : cfg["dataset_name"],
        "root"             : cfg["dataset_root"],
        "normal_dir"       : "train/good",
        "abnormal_dir"     : "test/defect",
        "normal_test_dir"  : "test/good",
        "train_batch_size" : 32,
        "eval_batch_size"  : 32,
        "num_workers"      : cfg["num_workers"],
    }

    # Dynamically pass parameters based on version
    if "image_size" in params:
        kwargs["image_size"] = (cfg["image_size"], cfg["image_size"])

    if "transform" in params:
        kwargs["transform"] = preprocess
    elif "train_transform" in params:
        kwargs["train_transform"] = preprocess
        kwargs["eval_transform"]  = preprocess

    # anomalib 2.x may use task parameter
    if "task" in params:
        from anomalib.data.utils import TaskType
        kwargs["task"] = TaskType.CLASSIFICATION

    return Folder(**kwargs)


# ============================================================
# ★ Core Fix: Auto-adapt model loading for different anomalib versions
# ============================================================
def get_patchcore_model():
    """
    anomalib 2.5.0 → Patchcore (lowercase c)
    anomalib 0.x   → PatchCore (uppercase C)
    """
    # Try 2.5.0 style first
    try:
        from anomalib.models import Patchcore          # ← Correct for 2.5.0
        print("  [Model Loading] ✅ anomalib 2.5.0: Using Patchcore")
        return Patchcore()
    except ImportError:
        pass

    # Fallback to older style
    try:
        from anomalib.models import PatchCore          # ← Older style
        print("  [Model Loading] ✅ Older anomalib: Using PatchCore")
        return PatchCore()
    except ImportError:
        pass

    # Last resort: string-based loading
    try:
        from anomalib.models import get_model
        print("  [Model Loading] ✅ Using get_model('patchcore')")
        return get_model("patchcore")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load Patchcore model using all methods: {e}")


# ============================================================
# Run a single ablation experiment
# ============================================================
def run_one_ablation(cfg, ablation_cfg, run_idx):
    from anomalib.engine import Engine

    name    = ablation_cfg["name"]
    run_dir = os.path.join(cfg["output_dir"], f"exp{run_idx:02d}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Experiment [{run_idx}/4]: {name}")
    print(f"  CLAHE={ablation_cfg['use_clahe']}  "
          f"USM={ablation_cfg['use_usm']}  "
          f"Interp={ablation_cfg['interp_name']}")
    print(f"{'='*60}")

    preprocess = CustomPreprocess(
        img_size      = cfg["image_size"],
        use_clahe     = ablation_cfg["use_clahe"],
        use_usm       = ablation_cfg["use_usm"],
        interpolation = ablation_cfg["interpolation"],
    )

    datamodule = build_datamodule(cfg, preprocess)
    datamodule.setup()

    # ★ Use the fixed model loading function
    model = get_patchcore_model()

    engine = Engine(
        default_root_dir = run_dir,
        accelerator      = "gpu",
        devices          = 1,
        max_epochs       = 1,
    )

    t0      = time.time()
    engine.fit(model=model, datamodule=datamodule)
    results = engine.test(model=model, datamodule=datamodule)
    t_total = time.time() - t0

    # Compatible with different result formats
    metrics = {}
    if results:
        if isinstance(results, list):
            metrics = results[0]
        elif isinstance(results, dict):
            metrics = results

    # Compatible with different metric key names
    auroc = float(
        metrics.get("image_AUROC",
        metrics.get("AUROC",
        metrics.get("auroc", 0)))
    )
    f1 = float(
        metrics.get("image_F1Score",
        metrics.get("F1Score",
        metrics.get("f1_score", 0)))
    )

    print(f"\n  ✅ Completed: AUROC={auroc:.4f}  F1={f1:.4f}  "
          f"Time={t_total/60:.1f}min")
    print(f"  Full metrics: {metrics}")   # Print all for debugging

    return {
        "name"    : name,
        "CLAHE"   : "✅" if ablation_cfg["use_clahe"] else "❌",
        "USM"     : "✅" if ablation_cfg["use_usm"]   else "❌",
        "Interp"  : ablation_cfg["interp_name"],
        "AUROC"   : auroc,
        "F1Score" : f1,
        "time_min": round(t_total / 60, 1),
    }


# ============================================================
# Print Ablation Result Table
# ============================================================
def print_ablation_table(results):
    valid = [r for r in results if r["AUROC"] > 0]
    if not valid:
        print("\n  ⚠️  All experiments failed, please check error messages")
        return

    print("\n" + "=" * 78)
    print("  📊 Ablation Study Results Table (Paper Table 1)")
    print("=" * 78)
    print(f"  {'Scheme':<30} {'CLAHE':^6} {'USM':^6} {'Interp':^8} "
          f"{'AUROC':^8} {'F1':^8}")
    print("-" * 78)

    baseline_auroc = results[0]["AUROC"]
    baseline_f1    = results[0]["F1Score"]

    for r in results:
        delta_a   = r["AUROC"]   - baseline_auroc
        delta_str = f"(+{delta_a:.4f})" if delta_a > 0 else "        "
        tag       = " ★ Proposed Method" if "Full Pipeline" in r["name"] else ""
        print(f"  {r['name']:<30} {r['CLAHE']:^6} {r['USM']:^6} "
              f"{r['Interp']:^8} {r['AUROC']:^8.4f} "
              f"{r['F1Score']:^8.4f}  {delta_str}{tag}")

    print("=" * 78)
    if valid:
        best = max(valid, key=lambda x: x["AUROC"])
        print(f"\n  Best Scheme  : {best['name']}")
        print(f"  AUROC Gain   : +{best['AUROC']-baseline_auroc:.4f}")
        print(f"  F1 Gain      : +{best['F1Score']-baseline_f1:.4f}")


# ============================================================
# Plot Ablation Results
# ============================================================
def plot_ablation(results, output_dir):
    valid = [r for r in results if r["AUROC"] > 0]
    if len(valid) < 2:
        print("  ⚠️  Less than 2 valid results, skipping plot")
        return

    try:
        names  = [r["name"] for r in results]
        aurocs = [r["AUROC"]   for r in results]
        f1s    = [r["F1Score"] for r in results]
        x      = np.arange(len(names))

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Ablation Study Results", fontsize=14, fontweight="bold")

        colors = ["#95A5A6"] * (len(results) - 1) + ["#E74C3C"]

        for ax, values, title, ylabel, best_line in [
            (axes[0], aurocs, "AUROC Comparison",   "AUROC",    0.95),
            (axes[1], f1s,    "F1Score Comparison", "F1 Score", 0.98),
        ]:
            bars = ax.bar(x, values, color=colors, width=0.5,
                          edgecolor="white", linewidth=1.5)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{val:.4f}", ha="center",
                        fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=9, rotation=10, ha="right")
            y_min = max(0.85, min(v for v in values if v > 0) - 0.03)
            ax.set_ylim([y_min, 1.02])
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=12)
            ax.axhline(y=best_line, color="green",
                       linestyle="--", alpha=0.6,
                       label=f"Excellent Line ({best_line})")
            ax.legend(fontsize=10)
            ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, "ablation_result.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  ✅ Ablation plot saved: {save_path}")

    except Exception as e:
        print(f"  ⚠️  Plotting failed: {e}")


# ============================================================
# Dataset Structure Check (Early Problem Detection)
# ============================================================
def check_dataset_structure(dataset_root):
    print("\n  [Directory Check] Verifying dataset structure...")
    required = [
        "train/good",
        "test/good",
        "test/defect",
    ]
    all_ok = True
    for rel_path in required:
        full_path = os.path.join(dataset_root, rel_path)
        exists    = os.path.isdir(full_path)
        files     = len(os.listdir(full_path)) if exists else 0
        status    = f"✅ ({files} files)" if exists and files > 0 else "❌ Missing or empty"
        print(f"    {rel_path:<20} → {status}")
        if not exists or files == 0:
            all_ok = False

    if not all_ok:
        print("\n  ❌ Dataset directory structure incomplete!")
        print("  Please ensure raw_data has the following structure:")
        print("""
  raw_data/
  ├── train/
  │   └── good/       ← Good samples for training (at least 10 images)
  └── test/
      ├── good/       ← Good samples for testing
      └── defect/     ← Defective samples for testing
        """)
        return False

    print("  ✅ Dataset structure verified!\n")
    return True


# ============================================================
# Main Workflow
# ============================================================
def main():
    cfg = CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)

    import anomalib
    print("=" * 60)
    print("   Preprocessing Ablation Study — Paper Table 1 Generator")
    print(f"  anomalib version : {anomalib.__version__}")
    print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    print(f"  Number of experiments : 4")
    print(f"  Estimated time   : 60~80 minutes")
    print("=" * 60)

    # ── Check directory structure first ──
    if not check_dataset_structure(cfg["dataset_root"]):
        print("  Please fix the directory structure and run again.")
        return

    all_results = []

    for i, ab_cfg in enumerate(ABLATION_CONFIGS):
        try:
            result = run_one_ablation(cfg, ab_cfg, run_idx=i + 1)
            all_results.append(result)
        except Exception as e:
            import traceback
            print(f"\n  ❌ Experiment failed: {ab_cfg['name']}")
            print(f"  Error: {e}")
            traceback.print_exc()          
            all_results.append({
                "name"    : ab_cfg["name"],
                "CLAHE"   : "✅" if ab_cfg["use_clahe"] else "❌",
                "USM"     : "✅" if ab_cfg["use_usm"]   else "❌",
                "Interp"  : ab_cfg["interp_name"],
                "AUROC"   : 0.0,
                "F1Score" : 0.0,
                "time_min": 0,
            })

    print_ablation_table(all_results)
    plot_ablation(all_results, cfg["output_dir"])

    print("\n" + "=" * 60)
    print("  📋 Next Steps:")
    print("  1. Paste the table above into your paper as Table 1")
    print("  2. Include ablation_result.png in your paper as a figure")
    print("  3. Send me the results, and I will help write the experimental analysis")
    print("=" * 60)


if __name__ == "__main__":
    main()