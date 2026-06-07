"""
PatchCore Automatic Hyperparameter Tuning Script
Automatically iterates through all parameter combinations to find the optimal configuration
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
# ★ Fixed Configuration
# ============================================================
CONFIG = {
    "dataset_root"   : r"D:\MLProject\augmention\augmention",
    "dataset_name"   : "augmention",
    "output_dir"     : r"D:\MLProject\results\patchcore_tune",
    "image_size"     : 256,
    "backbone"       : "wide_resnet50_2",
    "layers"         : ["layer2", "layer3"],
    "train_batch_size" : 32,
    "eval_batch_size"  : 32,
    "num_workers"      : 4,
}

# ============================================================
# ★ Parameter Grid
# ============================================================
PARAM_GRID = [
    {"coreset_sampling_ratio": 0.1,  "num_neighbors": 1 },
    {"coreset_sampling_ratio": 0.1,  "num_neighbors": 3 },
    {"coreset_sampling_ratio": 0.1,  "num_neighbors": 9 },   # Current best
    {"coreset_sampling_ratio": 0.1,  "num_neighbors": 19},
    {"coreset_sampling_ratio": 0.25, "num_neighbors": 1 },
    {"coreset_sampling_ratio": 0.25, "num_neighbors": 3 },
    {"coreset_sampling_ratio": 0.25, "num_neighbors": 9 },
    {"coreset_sampling_ratio": 0.25, "num_neighbors": 19},
]

# ============================================================
# Build Data Module (Auto-adapts to anomalib 2.5.0)
# ============================================================
def build_datamodule(cfg):
    from anomalib.data import Folder

    sig    = inspect.signature(Folder.__init__)
    params = set(sig.parameters.keys())

    base_kwargs = {
        "name"             : cfg["dataset_name"],
        "root"             : cfg["dataset_root"],
        "normal_dir"       : "train/good",
        "abnormal_dir"     : "test/defect",
        "normal_test_dir"  : "test/good",
        "train_batch_size" : cfg["train_batch_size"],
        "eval_batch_size"  : cfg["eval_batch_size"],
        "num_workers"      : cfg["num_workers"],
    }

    if "image_size" in params:
        base_kwargs["image_size"] = (cfg["image_size"], cfg["image_size"])
    elif "transform" in params:
        from torchvision.transforms import v2 as T
        base_kwargs["transform"] = T.Compose([
            T.Resize((cfg["image_size"], cfg["image_size"]),
                     interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])
    elif "train_transform" in params:
        from torchvision.transforms import v2 as T
        _t = T.Compose([
            T.Resize((cfg["image_size"], cfg["image_size"]),
                     interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])
        base_kwargs["train_transform"] = _t
        base_kwargs["eval_transform"]  = _t

    return Folder(**base_kwargs)


# ============================================================
# Single Training + Evaluation Run
# ============================================================
def run_once(cfg, coreset_ratio, num_neighbors, run_idx, total):
    from anomalib.models import Patchcore
    from anomalib.engine import Engine

    print(f"\n{'='*60}")
    print(f"  [{run_idx}/{total}] coreset={coreset_ratio}  neighbors={num_neighbors}")
    print(f"{'='*60}")

    run_dir = os.path.join(
        cfg["output_dir"],
        f"coreset{coreset_ratio}_neighbors{num_neighbors}"
    )
    os.makedirs(run_dir, exist_ok=True)

    # Data module
    datamodule = build_datamodule(cfg)
    datamodule.setup()

    # Model
    model = Patchcore(
        backbone               = cfg["backbone"],
        layers                 = cfg["layers"],
        coreset_sampling_ratio = coreset_ratio,
        num_neighbors          = num_neighbors,
    )

    # Engine
    engine = Engine(
        default_root_dir = run_dir,
        accelerator      = "gpu",
        devices          = 1,
        max_epochs       = 1,
    )

    # Training
    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    t_train = time.time() - t0

    # Evaluation
    results  = engine.test(model=model, datamodule=datamodule)
    t_total  = time.time() - t0
    metrics  = results[0] if results else {}

    auroc  = float(metrics.get("image_AUROC",   0))
    f1     = float(metrics.get("image_F1Score", 0))

    print(f"\n  ✅ coreset={coreset_ratio}  neighbors={num_neighbors}")
    print(f"     image_AUROC  : {auroc:.4f}")
    print(f"     image_F1Score: {f1:.4f}")
    print(f"     Time elapsed  : {t_total/60:.1f} minutes")

    return {
        "coreset_ratio" : coreset_ratio,
        "num_neighbors" : num_neighbors,
        "image_AUROC"   : auroc,
        "image_F1Score" : f1,
        "time_min"      : round(t_total / 60, 1),
    }


# ============================================================
# Summary Printing + Plotting
# ============================================================
def print_summary(all_results):
    print("\n" + "=" * 70)
    print("  🏆 Hyperparameter Tuning Results Summary")
    print("=" * 70)
    print(f"  {'coreset':>8}  {'neighbors':>10}  {'AUROC':>8}  {'F1':>8}  {'Time':>6}")
    print("-" * 70)

    sorted_results = sorted(all_results, key=lambda x: x["image_AUROC"], reverse=True)
    for i, r in enumerate(sorted_results):
        tag = " <- 🏆 Best" if i == 0 else ""
        print(f"  {r['coreset_ratio']:>8}  {r['num_neighbors']:>10}  "
              f"{r['image_AUROC']:>8.4f}  {r['image_F1Score']:>8.4f}  "
              f"{r['time_min']:>5.1f}m{tag}")

    best = sorted_results[0]
    print("=" * 70)
    print(f"\n  ✅ Optimal Configuration:")
    print(f"     coreset_sampling_ratio = {best['coreset_ratio']}")
    print(f"     num_neighbors          = {best['num_neighbors']}")
    print(f"     image_AUROC            = {best['image_AUROC']:.4f}")
    print(f"     image_F1Score          = {best['image_F1Score']:.4f}")

    return best


def plot_results(all_results, output_dir):
    try:
        import pandas as pd
        df = pd.DataFrame(all_results)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot lines grouped by coreset_ratio
        for ratio in sorted(df["coreset_ratio"].unique()):
            sub = df[df["coreset_ratio"] == ratio].sort_values("num_neighbors")
            axes[0].plot(sub["num_neighbors"], sub["image_AUROC"],
                         marker='o', linewidth=2,
                         label=f'coreset={ratio}')
            axes[1].plot(sub["num_neighbors"], sub["image_F1Score"],
                         marker='s', linewidth=2,
                         label=f'coreset={ratio}')

        for ax, title, metric in zip(
            axes,
            ["image_AUROC vs num_neighbors", "image_F1Score vs num_neighbors"],
            ["AUROC", "F1 Score"]
        ):
            ax.set_xlabel("num_neighbors", fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0.95, color='green', linestyle='--',
                       alpha=0.6, label='0.95 baseline')

        plt.tight_layout()
        save_path = os.path.join(output_dir, "tune_results.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\n  ✅ Tuning plot saved: {save_path}")

    except Exception as e:
        print(f"  ⚠️  Plotting failed: {e}")


# ============================================================
# Main Workflow
# ============================================================
def main():
    cfg   = CONFIG
    grid  = PARAM_GRID
    total = len(grid)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("=" * 60)
    print(f"  🔍 PatchCore Automatic Hyperparameter Tuning")
    print(f"  Total {total} parameter combinations")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Estimated total time: {total * 4 // 60}~{total * 6 // 60} minutes")
    print("=" * 60)

    all_results = []

    for i, params in enumerate(grid, 1):
        try:
            result = run_once(
                cfg,
                coreset_ratio  = params["coreset_sampling_ratio"],
                num_neighbors  = params["num_neighbors"],
                run_idx        = i,
                total          = total,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ❌ Combination {i} failed: {e}")
            continue

    if not all_results:
        print("❌ All combinations failed!")
        return

    best = print_summary(all_results)
    plot_results(all_results, cfg["output_dir"])

    # Prompt to save optimal config
    print(f"\n  📝 Copy the optimal config into train.py CONFIG:")
    print(f"""
CONFIG = {{
    ...
    "coreset_sampling_ratio"  : {best['coreset_ratio']},
    "num_neighbors"           : {best['num_neighbors']},
    ...
}}
""")


if __name__ == "__main__":
    main()