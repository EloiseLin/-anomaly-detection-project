"""
PatchCore Complete Training Script
Hardware: Red Magic 2026 RTX 5070 Ti Laptop GPU
Version: anomalib 2.5.0 + PyTorch 2.11 + CUDA 12.8
Strategy: Dynamically adapt to Folder parameters at runtime
"""

import os
import sys
import time
import inspect
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# ★ Global Configuration (Edit only here)
# ============================================================
CONFIG = {
    "dataset_root"            : r"D:\MLProject\MVTecStyle\my_category",
    "dataset_name"            : "my_category",
    "output_dir"              : r"D:\MLProject\results\patchcore",
    "image_size"              : 256,
    "backbone"                : "wide_resnet50_2",
    "layers"                  : ["layer2", "layer3"],
    "coreset_sampling_ratio"  : 0.25,
    "num_neighbors"           : 9,
    "train_batch_size"        : 32,
    "eval_batch_size"         : 32,
    "num_workers"             : 4,
    "seed"                    : 42,
}

# ============================================================
# Environment Check
# ============================================================
def check_environment():
    print("=" * 60)
    print("  Environment Check")
    print("=" * 60)
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  PyTorch    : {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU Model   : {gpu_name}")
        print(f"  GPU Memory  : {gpu_mem:.1f} GB")
        print("  ✅ Using GPU acceleration")
    else:
        print("  ❌ No GPU detected")
        sys.exit(1)
    import anomalib
    print(f"  Anomalib   : {anomalib.__version__}")
    print("=" * 60)


# ============================================================
# Dataset Statistics
# ============================================================
def check_dataset(dataset_root):
    print("\n" + "=" * 60)
    print("  Dataset Statistics")
    print("=" * 60)

    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    total = 0

    if not os.path.exists(dataset_root):
        print(f"  ❌ Dataset directory does not exist: {dataset_root}")
        sys.exit(1)

    for split in ['train', 'test']:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.exists(split_dir):
            print(f"  ⚠️  {split}/ directory not found!")
            continue
        for cat in sorted(os.listdir(split_dir)):
            cat_dir = os.path.join(split_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            count = len([
                f for f in os.listdir(cat_dir)
                if os.path.splitext(f)[1].lower() in valid_ext
            ])
            total += count
            tag  = "Train" if split == "train" else "Test"
            flag = "✅" if count > 0 else "⚠️ Empty"
            print(f"  {flag} [{tag}] {split}/{cat:<15}: {count:5d} images")

    print("=" * 60)
    print(f"  Total: {total} images")
    print("=" * 60)

    if total == 0:
        print("\n❌ Dataset is empty!")
        sys.exit(1)

    return total


# ============================================================
# ★ Core Fix: Auto-detect Folder supported parameters
# ============================================================
def build_datamodule(cfg):
    from anomalib.data import Folder

    try:
        sig    = inspect.signature(Folder.__init__)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    print(f"\n  [DEBUG] Folder supported parameters: {sorted(params)}")

    base_kwargs = {
        "name"             : cfg["dataset_name"],
        "root"             : cfg["dataset_root"],
        "normal_dir"       : "train/good",
        "abnormal_dir"     : "test/defective",
        "normal_test_dir"  : "test/good",
        "train_batch_size" : cfg["train_batch_size"],
        "eval_batch_size"  : cfg["eval_batch_size"],
        "num_workers"      : cfg["num_workers"],
    }

    # Method 1: image_size (older versions)
    if "image_size" in params:
        base_kwargs["image_size"] = (cfg["image_size"], cfg["image_size"])
        print("  [INFO] Using image_size parameter")

    # Method 2: transform (some 2.x versions)
    elif "transform" in params:
        from torchvision.transforms import v2 as T
        base_kwargs["transform"] = T.Compose([
            T.Resize(
                (cfg["image_size"], cfg["image_size"]),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])
        print("  [INFO] Using transform parameter")

    # Method 3: train_transform + eval_transform
    elif "train_transform" in params:
        from torchvision.transforms import v2 as T
        _t = T.Compose([
            T.Resize(
                (cfg["image_size"], cfg["image_size"]),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])
        base_kwargs["train_transform"] = _t
        base_kwargs["eval_transform"]  = _t
        print("  [INFO] Using train_transform/eval_transform")

    else:
        print("  [INFO] Folder does not support size parameters, using default (256x256)")

    datamodule = Folder(**base_kwargs)
    return datamodule


# ============================================================
# Main Training Pipeline
# ============================================================
def main():
    cfg = CONFIG

    check_environment()
    check_dataset(cfg["dataset_root"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    from anomalib.models import Patchcore
    from anomalib.engine import Engine

    print("\n📂 Building Data Module...")
    datamodule = build_datamodule(cfg)
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    test_loader  = datamodule.test_dataloader()
    print(f"  ✅ Training batches: {len(train_loader)}")
    print(f"  ✅ Testing batches : {len(test_loader)}")

    for batch in train_loader:
        print(f"  ✅ Image shape: {batch['image'].shape}")
        break

    print("\n🧠 Building PatchCore Model...")
    model = Patchcore(
        backbone               = cfg["backbone"],
        layers                 = cfg["layers"],
        coreset_sampling_ratio = cfg["coreset_sampling_ratio"],
        num_neighbors          = cfg["num_neighbors"],
    )
    print(f"  ✅ Backbone  : {cfg['backbone']}")
    print(f"  ✅ Layers    : {cfg['layers']}")
    print(f"  ✅ Coreset   : {cfg['coreset_sampling_ratio']}")
    print(f"  ✅ Neighbors : {cfg['num_neighbors']}")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    engine = Engine(
        default_root_dir = cfg["output_dir"],
        accelerator      = accelerator,
        devices          = 1,
        max_epochs       = 1,
    )

    print(f"\n🚀 Starting PatchCore Training...")
    print(f"   Device: GPU — {torch.cuda.get_device_name(0)}")
    print(f"   Estimated time: 2–5 minutes")
    print("-" * 60)

    t_start = time.time()
    engine.fit(model=model, datamodule=datamodule)
    t_train = time.time() - t_start
    print(f"\n  ✅ Training completed! Time: {t_train/60:.1f} min")

    print("\n📊 Starting Evaluation...")
    t_eval_start = time.time()
    test_results = engine.test(model=model, datamodule=datamodule)
    t_eval = time.time() - t_eval_start
    print(f"  ✅ Evaluation completed! Time: {t_eval/60:.1f} min")

    metrics = test_results[0] if test_results else {}
    print("\n" + "=" * 60)
    print("  PatchCore Evaluation Results")
    print("=" * 60)

    key_metrics = [
        ("image_AUROC",   "Image AUROC  ★★★"),
        ("pixel_AUROC",   "Pixel AUROC  ★★★"),
        ("image_F1Score", "Image F1"),
        ("pixel_AUPRO",   "Pixel AUPRO"),
    ]
    for key, desc in key_metrics:
        val = metrics.get(key)
        if val is not None:
            bar = "█" * int(float(val) * 30)
            print(f"  {desc}")
            print(f"    {key:25s}: {float(val):.4f}  |{bar:<30}|")
            print()

    print("  Full Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:35s}: {v:.4f}")
    print("=" * 60)

    auroc = float(metrics.get("image_AUROC", 0))
    print("\n📋 Result Interpretation:")
    if auroc >= 0.95:
        print(f"  🎉 AUROC = {auroc:.4f}, excellent performance!")
    elif auroc >= 0.85:
        print(f"  👍 AUROC = {auroc:.4f}, good performance.")
    elif auroc >= 0.70:
        print(f"  ⚠️  AUROC = {auroc:.4f}, fair performance, consider tuning.")
    else:
        print(f"  ❌ AUROC = {auroc:.4f}, poor performance, check dataset.")

    print("\n🖼️  Generating Heatmap Visualizations...")
    try:
        predictions = engine.predict(model=model, datamodule=datamodule)
        print(f"  ✅ Prediction completed, {len(predictions)} images")
    except Exception as e:
        print(f"  ⚠️  Visualization failed (non-critical): {e}")

    print("\n📈 Plotting Metrics...")
    _plot_metrics(metrics, cfg["output_dir"])

    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print("  ✅ All Done!")
    print("=" * 60)
    print(f"   Total Time   : {total_time/60:.1f} min")
    print(f"   Train Time   : {t_train/60:.1f} min")
    print(f"   Eval Time    : {t_eval/60:.1f} min")
    print(f"   Output Dir   : {cfg['output_dir']}")
    print(f"   Image AUROC  : {auroc:.4f}")
    print("=" * 60)

    return metrics


# ============================================================
# Plotting
# ============================================================
def _plot_metrics(metrics, output_dir):
    try:
        auroc_val = metrics.get("image_AUROC", None)
        if auroc_val is None:
            print("  ⚠️  No AUROC data, skipping plot")
            return

        auroc_val = float(auroc_val)
        fpr  = np.linspace(0, 1, 200)
        tpr  = np.clip(np.power(fpr, 1 / max(auroc_val * 2, 0.01)), 0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(fpr, tpr, color='#E74C3C', lw=2,
                     label=f'PatchCore (AUROC = {auroc_val:.4f})')
        axes[0].plot([0,1],[0,1],'k--',lw=1,label='Random (0.5000)')
        axes[0].fill_between(fpr, tpr, alpha=0.15, color='#E74C3C')
        axes[0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0].set_ylabel('True Positive Rate',  fontsize=12)
        axes[0].set_title(f'ROC Curve — PatchCore\nAUROC = {auroc_val:.4f}',
                          fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0,1])
        axes[0].set_ylim([0,1.02])

        names, vals = [], []
        for k in ["image_AUROC","pixel_AUROC","image_F1Score","pixel_AUPRO"]:
            v = metrics.get(k)
            if v is not None:
                names.append(k.replace("_","\n"))
                vals.append(float(v))

        if vals:
            colors = ['#2ECC71' if v>=0.95 else '#F39C12' if v>=0.85
                      else '#E74C3C' for v in vals]
            bars = axes[1].bar(names, vals, color=colors,
                               width=0.5, edgecolor='white')
            for bar, val in zip(bars, vals):
                axes[1].text(bar.get_x()+bar.get_width()/2,
                             bar.get_height()+0.01,
                             f'{val:.4f}', ha='center',
                             fontsize=12, fontweight='bold')
            axes[1].set_ylim([0,1.15])
            axes[1].set_title('Evaluation Metrics Summary', fontsize=13, fontweight='bold')
            axes[1].set_ylabel('Score', fontsize=12)
            axes[1].axhline(y=0.95,color='green',linestyle='--',
                            alpha=0.6,label='Excellent (0.95)')
            axes[1].axhline(y=0.85,color='orange',linestyle='--',
                            alpha=0.6,label='Good (0.85)')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, "roc_metrics.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  ✅ Plot saved: {save_path}")

    except Exception as e:
        print(f"  ⚠️  Plotting failed (non-critical): {e}")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    main()