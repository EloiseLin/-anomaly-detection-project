
"""
Ablation Study - PatchCore Runner
====================================
Runs PatchCore on all 5 ablation groups and produces Table 1.

Group | Training data                          | Preprocessing       | Status
------+----------------------------------------+---------------------+--------
  1   | RawData/raw          (original)        | None                | skip (use cached)
  2   | ablation_datasets/aug_only             | Aug only            | run
  3   | ablation_datasets/aug_clahe            | Aug + CLAHE         | run
  4   | ablation_datasets/aug_usm              | Aug + USM + CUBIC   | run
  5   | MVTecStyle/my_category (full pipeline) | Aug + CLAHE + USM   | skip (use cached)
"""

import os
import time
import warnings
import torch
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ============================================================
# Configuration — edit only this block
# ============================================================
RESULTS_DIR = r"D:\MLProject\results\ablation_v2"

ALL_GROUPS = [
    {
        "id"          : 1,
        "name"        : "Baseline (no aug, no preprocess)",
        "aug"         : False,
        "clahe"       : False,
        "usm"         : False,
        "interpolation": "LINEAR",
        "dataset_root": r"D:\MLProject\RawData\raw",
        "dataset_name": "raw",
        "cached_auroc": 0.9435,   # already have this result
        "cached_f1"   : 0.9855,   # already have this result
        "skip"        : True,     # set True = use cached result, skip training
    },
    {
        "id"          : 2,
        "name"        : "Aug only (no preprocess)",
        "aug"         : True,
        "clahe"       : False,
        "usm"         : False,
        "interpolation": "LINEAR",
        "dataset_root": r"D:\MLProject\ablation_datasets\aug_only",
        "dataset_name": "aug_only",
        "cached_auroc": None,
        "cached_f1"   : None,
        "skip"        : False,
    },
    {
        "id"          : 3,
        "name"        : "Aug + CLAHE",
        "aug"         : True,
        "clahe"       : True,
        "usm"         : False,
        "interpolation": "LINEAR",
        "dataset_root": r"D:\MLProject\ablation_datasets\aug_clahe",
        "dataset_name": "aug_clahe",
        "cached_auroc": None,
        "cached_f1"   : None,
        "skip"        : False,
    },
    {
        "id"          : 4,
        "name"        : "Aug + USM + CUBIC",
        "aug"         : True,
        "clahe"       : False,
        "usm"         : True,
        "interpolation": "CUBIC",
        "dataset_root": r"D:\MLProject\ablation_datasets\aug_usm",
        "dataset_name": "aug_usm",
        "cached_auroc": None,
        "cached_f1"   : None,
        "skip"        : False,
    },
    {
        "id"          : 5,
        "name"        : "Full Pipeline — Ours (Aug + CLAHE + USM + CUBIC)",
        "aug"         : True,
        "clahe"       : True,
        "usm"         : True,
        "interpolation": "CUBIC",
        "dataset_root": r"D:\MLProject\MVTecStyle\my_category",
        "dataset_name": "my_category",
        "cached_auroc": 0.9644,   # already have this result
        "cached_f1"   : 0.9815,   # already have this result
        "skip"        : True,     # set True = use cached result, skip training
    },
]

# ============================================================
# Build DataModule (compatible with anomalib 2.5.0)
# ============================================================
def build_datamodule(dataset_root, dataset_name, num_workers=4):
    import inspect
    from anomalib.data import Folder

    sig    = inspect.signature(Folder.__init__)
    params = set(sig.parameters.keys())

    kwargs = {
        "name"             : dataset_name,
        "root"             : dataset_root,
        "normal_dir"       : "train/good",
        "abnormal_dir"     : "test/defect",
        "normal_test_dir"  : "test/good",
        "train_batch_size" : 32,
        "eval_batch_size"  : 32,
        "num_workers"      : num_workers,
    }

    if "image_size" in params:
        kwargs["image_size"] = (256, 256)
    if "task" in params:
        try:
            from anomalib.data.utils import TaskType
            kwargs["task"] = TaskType.CLASSIFICATION
        except ImportError:
            pass

    return Folder(**kwargs)


# ============================================================
# Load PatchCore model (compatible with anomalib 2.5.0)
# ============================================================
def load_patchcore():
    try:
        from anomalib.models import Patchcore
        return Patchcore()
    except ImportError:
        pass
    try:
        from anomalib.models import PatchCore
        return PatchCore()
    except ImportError:
        pass
    from anomalib.models import get_model
    return get_model("patchcore")


# ============================================================
# Run one group
# ============================================================
def run_group(group):
    from anomalib.engine import Engine

    gid      = group["id"]
    name     = group["name"]
    run_dir  = os.path.join(RESULTS_DIR, f"group{gid:02d}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Group [{gid}/5] : {name}")
    print(f"  Dataset       : {group['dataset_root']}")
    print(f"{'='*65}")

    datamodule = build_datamodule(group["dataset_root"], group["dataset_name"])
    datamodule.setup()

    model  = load_patchcore()
    engine = Engine(
        default_root_dir = run_dir,
        accelerator      = "gpu",
        devices          = 1,
        max_epochs       = 1,
    )

    t0      = time.time()
    engine.fit(model=model, datamodule=datamodule)
    results = engine.test(model=model, datamodule=datamodule)
    elapsed = time.time() - t0

    # Parse results (compatible with different anomalib versions)
    metrics = {}
    if results:
        metrics = results[0] if isinstance(results, list) else results

    auroc = float(metrics.get("image_AUROC",   metrics.get("AUROC",    0.0)))
    f1    = float(metrics.get("image_F1Score", metrics.get("F1Score",  0.0)))

    print(f"\n  Result : AUROC={auroc:.4f}  F1={f1:.4f}  "
          f"Time={elapsed/60:.1f}min")
    print(f"  Full metrics: {metrics}")

    return auroc, f1


# ============================================================
# Print result table  (Paper Table 1)
# ============================================================
def print_table(final_results):
    print("\n")
    print("=" * 90)
    print("  ABLATION STUDY RESULTS  —  Paper Table 1")
    print("=" * 90)
    print(f"  {'Group':<2}  {'Method':<45} {'Aug':^5} {'CLAHE':^6} "
          f"{'USM':^5} {'Interp':^7} {'AUROC':^8} {'F1':^8}")
    print("-" * 90)

    baseline_auroc = final_results[0]["auroc"]
    baseline_f1    = final_results[0]["f1"]

    for r in final_results:
        aug_mark   = "yes" if r["aug"]   else "no"
        clahe_mark = "yes" if r["clahe"] else "no"
        usm_mark   = "yes" if r["usm"]   else "no"

        delta_a = r["auroc"] - baseline_auroc
        delta_f = r["f1"]    - baseline_f1
        delta_str = f"+{delta_a:+.4f}" if delta_a != 0 else "  —    "
        star = " <-- OURS" if r["id"] == 5 else ""

        print(f"  {r['id']:<2}  {r['name']:<45} {aug_mark:^5} {clahe_mark:^6} "
              f"{usm_mark:^5} {r['interpolation']:^7} "
              f"{r['auroc']:^8.4f} {r['f1']:^8.4f} {delta_str}{star}")

    print("=" * 90)

    best = max(final_results, key=lambda x: x["auroc"])
    print(f"\n  Best AUROC : Group {best['id']} — {best['name']}")
    print(f"  AUROC gain over Baseline : +{best['auroc'] - baseline_auroc:.4f}")
    print(f"  F1    gain over Baseline : +{best['f1']    - baseline_f1:.4f}")


# ============================================================
# Plot bar chart  (Paper Figure)
# ============================================================
def plot_results(final_results, output_dir):
    try:
        names  = [f"G{r['id']}" for r in final_results]
        labels = [r["name"] for r in final_results]
        aurocs = [r["auroc"] for r in final_results]
        f1s    = [r["f1"]    for r in final_results]

        # Highlight the last bar (our full method)
        colors = ["#5B9BD5"] * (len(final_results) - 1) + ["#E74C3C"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Ablation Study — Preprocessing Component Contribution",
                     fontsize=14, fontweight="bold")

        x = np.arange(len(names))

        for ax, values, title, ylabel in [
            (axes[0], aurocs, "Image-level AUROC", "AUROC"),
            (axes[1], f1s,    "Image-level F1 Score", "F1 Score"),
        ]:
            bars = ax.bar(x, values, color=colors, width=0.55,
                          edgecolor="white", linewidth=1.5)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 0.002,
                        f"{val:.4f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=11)
            ax.set_ylim([max(0.85, min(values) - 0.03), 1.02])
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, axis="y", alpha=0.3)

            # Add legend for group names
            for xi, label in zip(x, labels):
                ax.text(xi, max(0.85, min(values) - 0.03) + 0.002,
                        f"G{xi+1}", ha="center", fontsize=9,
                        color="white", fontweight="bold")

        # Add a legend box explaining group IDs
        legend_text = "\n".join(
            [f"G{r['id']}: {r['name'][:40]}" for r in final_results]
        )
        fig.text(0.01, 0.01, legend_text,
                 fontsize=7, verticalalignment="bottom",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        plt.tight_layout(rect=[0, 0.12, 1, 1])
        save_path = os.path.join(output_dir, "ablation_table1.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Figure saved: {save_path}")

    except Exception as e:
        print(f"  [Plot Error] {e}")


# ============================================================
# Main
# ============================================================
def main():
    import anomalib

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("  Ablation Study Runner — Paper Table 1")
    print(f"  anomalib version : {anomalib.__version__}")
    print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    print(f"  Total groups     : {len(ALL_GROUPS)}")
    print(f"  Groups to run    : {sum(1 for g in ALL_GROUPS if not g['skip'])}")
    print(f"  Groups cached    : {sum(1 for g in ALL_GROUPS if g['skip'])}")
    print("=" * 65)

    # Verify all dataset directories exist before starting
    print("\n  [Pre-check] Verifying dataset directories...")
    all_ok = True
    for g in ALL_GROUPS:
        exists = os.path.isdir(g["dataset_root"])
        status = "OK    " if exists else "MISSING"
        print(f"    [{status}] Group {g['id']} — {g['dataset_root']}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n  ERROR: Some dataset directories are missing.")
        print("  Please run generate_ablation_datasets.py first.")
        return

    print("\n  All directories verified. Starting experiments...\n")

    final_results = []

    for group in ALL_GROUPS:
        if group["skip"]:
            # Use cached result
            print(f"\n  Group [{group['id']}/5] : {group['name']}")
            print(f"  [Cached] AUROC={group['cached_auroc']:.4f}  "
                  f"F1={group['cached_f1']:.4f}")
            final_results.append({
                "id"           : group["id"],
                "name"         : group["name"],
                "aug"          : group["aug"],
                "clahe"        : group["clahe"],
                "usm"          : group["usm"],
                "interpolation": group["interpolation"],
                "auroc"        : group["cached_auroc"],
                "f1"           : group["cached_f1"],
            })
        else:
            # Actually run PatchCore
            try:
                auroc, f1 = run_group(group)
                final_results.append({
                    "id"           : group["id"],
                    "name"         : group["name"],
                    "aug"          : group["aug"],
                    "clahe"        : group["clahe"],
                    "usm"          : group["usm"],
                    "interpolation": group["interpolation"],
                    "auroc"        : auroc,
                    "f1"           : f1,
                })
            except Exception as e:
                import traceback
                print(f"\n  ERROR in Group {group['id']}: {e}")
                traceback.print_exc()
                final_results.append({
                    "id"           : group["id"],
                    "name"         : group["name"],
                    "aug"          : group["aug"],
                    "clahe"        : group["clahe"],
                    "usm"          : group["usm"],
                    "interpolation": group["interpolation"],
                    "auroc"        : 0.0,
                    "f1"           : 0.0,
                })

    # Output table and figure
    print_table(final_results)
    plot_results(final_results, RESULTS_DIR)

    print("\n" + "=" * 65)
    print("  All done!")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("  Next: paste the table into your paper as Table 1")
    print("=" * 65)


if __name__ == "__main__":
    main()
