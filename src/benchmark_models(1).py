
"""
Multi-Model Benchmark — Paper Table 2 (Fixed Version)
=======================================================
Fixes:
  1. PaDiM / STFPM  : force HF mirror + local cache fallback
  2. EfficientAd    : separate datamodule with train_batch_size=1
  3. Better error messages with retry logic

Run download_weights.py FIRST if network errors persist.
"""
# ── Patch timm to load resnet18 from local file, bypass HuggingFace ──
import timm.models._hub as _timm_hub
import torch, os

_LOCAL_WEIGHTS = r"D:\MLModels\model.safetensors" 

_orig_load = _timm_hub.load_state_dict_from_hf

def _patched_load(model_id, *args, **kwargs):
    if "resnet18" in model_id and os.path.exists(_LOCAL_WEIGHTS):
        print(f"  [Patch] Loading resnet18 from local: {_LOCAL_WEIGHTS}")
        return torch.load(_LOCAL_WEIGHTS, weights_only=True, map_location="cpu")
    return _orig_load(model_id, *args, **kwargs)

_timm_hub.load_state_dict_from_hf = _patched_load
# ── End patch ──
import os
import time
import warnings
import traceback
import torch
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt

# ── Must set BEFORE any huggingface import ──
os.environ["HF_ENDPOINT"]              = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "warning"

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
DATASET_ROOT = r"D:\MLProject\MVTecStyle\my_category"
DATASET_NAME = "my_category"
RESULTS_DIR  = r"D:\MLProject\results\benchmark"
NUM_WORKERS  = 4
IMAGE_SIZE   = 256


MODEL_CONFIGS = [
    # PatchCore result cached from HyperparameterTuning.py
# Best config: coreset_sampling_ratio=0.25, num_neighbors=9
# Retraining skipped to avoid redundant computation
    {
        "id"          : 1,
        "name"        : "PatchCore",
        "class_name"  : "Patchcore",
        "batch_size"  : 32,
        "note"        : "Our method",
        "cached_auroc": 0.9644,
        "cached_f1"   : 0.9815,
        "skip"        : True,
    },
    {
        "id"          : 2,
        "name"        : "ReverseDistillation",
        "class_name"  : "ReverseDistillation",
        "batch_size"  : 32,
        "note"        : "Teacher-student reverse distillation",
        "cached_auroc": 0.7474,    # already ran
        "cached_f1"   : 0.9768,
        "skip"        : True,      # use cached result
    },
    {
        "id"          : 3,
        "name"        : "PaDiM",
        "class_name"  : "Padim",
        "batch_size"  : 32,
        "note"        : "Patch distribution modeling",
        "cached_auroc": None,
        "cached_f1"   : None,
        "skip"        : False,
    },
    {
        "id"          : 4,
        "name"        : "STFPM",
        "class_name"  : "Stfpm",
        "batch_size"  : 32,
        "note"        : "Student-teacher feature pyramid matching",
        "cached_auroc": None,
        "cached_f1"   : None,
        "skip"        : False,
    },
    {
        "id"          : 5,
        "name"        : "EfficientAd",
        "class_name"  : "EfficientAd",
        "batch_size"  : 1,         # ← EfficientAd MUST be 1
        "note"        : "Efficient anomaly detection",
        "cached_auroc": 0.3607,
        "cached_f1"   : 0.9762,
        "skip"        : True,
    },
]

# ============================================================
# Build DataModule — batch_size as parameter
# ============================================================
def build_datamodule(train_batch_size=32):
    import inspect
    from anomalib.data import Folder

    sig    = inspect.signature(Folder.__init__)
    params = set(sig.parameters.keys())

    kwargs = {
        "name"             : DATASET_NAME,
        "root"             : DATASET_ROOT,
        "normal_dir"       : "train/good",
        "abnormal_dir"     : "test/defect",
        "normal_test_dir"  : "test/good",
        "train_batch_size" : train_batch_size,
        "eval_batch_size"  : 32,
        "num_workers"      : NUM_WORKERS,
    }

    if "image_size" in params:
        kwargs["image_size"] = (IMAGE_SIZE, IMAGE_SIZE)
    if "task" in params:
        try:
            from anomalib.data.utils import TaskType
            kwargs["task"] = TaskType.CLASSIFICATION
        except ImportError:
            pass

    return Folder(**kwargs)


# ============================================================
# Load model
# ============================================================
def load_model(class_name):
    import anomalib.models as model_module

    if hasattr(model_module, class_name):
        cls = getattr(model_module, class_name)
        return cls()

    for attr_name in dir(model_module):
        if attr_name.lower() == class_name.lower():
            cls = getattr(model_module, attr_name)
            return cls()

    try:
        from anomalib.models import get_model
        return get_model(class_name.lower())
    except Exception:
        pass

    raise ImportError(
        f"Cannot find model '{class_name}'.\n"
        f"Available: {[x for x in dir(model_module) if not x.startswith('_')]}"
    )


# ============================================================
# Run one model
# ============================================================
def run_model(model_cfg):
    from anomalib.engine import Engine

    mid         = model_cfg["id"]
    name        = model_cfg["name"]
    batch_size  = model_cfg["batch_size"]
    run_dir     = os.path.join(RESULTS_DIR, f"model{mid:02d}_{name}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Model [{mid}/{len(MODEL_CONFIGS)}] : {name}")
    print(f"  Note        : {model_cfg['note']}")
    print(f"  Batch size  : {batch_size}")
    print(f"{'='*65}")

    # Each model gets its own datamodule with correct batch_size
    datamodule = build_datamodule(train_batch_size=batch_size)
    datamodule.setup()

    model  = load_model(model_cfg["class_name"])
    print(f"  Loaded : {model.__class__.__name__}")

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

    metrics = {}
    if results:
        metrics = results[0] if isinstance(results, list) else results

    auroc = float(metrics.get("image_AUROC",   metrics.get("AUROC",   0.0)))
    f1    = float(metrics.get("image_F1Score", metrics.get("F1Score", 0.0)))

    print(f"\n  Result  : AUROC={auroc:.4f}  F1={f1:.4f}  "
          f"Time={elapsed/60:.1f}min")
    print(f"  Full metrics : {metrics}")

    return auroc, f1, round(elapsed / 60, 1)


# ============================================================
# Print Table 2
# ============================================================
def print_table(final_results):
    print("\n")
    print("=" * 72)
    print("  MULTI-MODEL BENCHMARK  —  Paper Table 2")
    print(f"  Dataset : Full Pipeline (Aug + CLAHE + USM + CUBIC)")
    print("=" * 72)
    print(f"  {'#':<3} {'Model':<26} {'AUROC':^10} {'F1':^10} {'vs PatchCore':^14}")
    print("-" * 72)

    pc_auroc = next(r["auroc"] for r in final_results if r["id"] == 1)
    pc_f1    = next(r["f1"]    for r in final_results if r["id"] == 1)

    for r in final_results:
        delta     = r["auroc"] - pc_auroc
        delta_str = "  (baseline)" if r["id"] == 1 else f"  ({delta:+.4f})"
        star      = "  <-- OURS"   if r["id"] == 1 else ""
        auroc_str = f"{r['auroc']:.4f}" if r["auroc"] > 0 else "  ERROR"
        f1_str    = f"{r['f1']:.4f}"    if r["f1"]    > 0 else "  ERROR"
        print(f"  {r['id']:<3} {r['name']:<26} {auroc_str:^10} "
              f"{f1_str:^10} {delta_str}{star}")

    print("=" * 72)


# ============================================================
# Plot
# ============================================================
def plot_results(final_results, output_dir):
    try:
        valid   = [r for r in final_results if r["auroc"] > 0]
        names   = [r["name"]  for r in valid]
        aurocs  = [r["auroc"] for r in valid]
        f1s     = [r["f1"]    for r in valid]
        colors  = ["#E74C3C" if r["id"] == 1 else "#5B9BD5" for r in valid]
        x       = np.arange(len(names))

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            "Multi-Model Benchmark on Full Pipeline Dataset",
            fontsize=13, fontweight="bold"
        )

        for ax, values, title, ylabel in [
            (axes[0], aurocs, "Image-level AUROC",    "AUROC"),
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
            ax.set_xticklabels(names, fontsize=9, rotation=12, ha="right")
            y_min = max(0.70, min(values) - 0.05)
            ax.set_ylim([y_min, 1.04])
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, axis="y", alpha=0.3)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#E74C3C", label="PatchCore (Ours)"),
            Patch(facecolor="#5B9BD5", label="Comparison models"),
        ]
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.01))

        plt.tight_layout(rect=[0, 0.07, 1, 1])
        save_path = os.path.join(output_dir, "benchmark_table2.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Figure saved : {save_path}")
    except Exception as e:
        print(f"  [Plot Error] {e}")
        traceback.print_exc()


# ============================================================
# Main
# ============================================================
def main():
    import anomalib

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("  Multi-Model Benchmark  —  Paper Table 2  (Fixed)")
    print(f"  anomalib version : {anomalib.__version__}")
    print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    print(f"  HF mirror        : {os.environ['HF_ENDPOINT']}")
    print("=" * 65)

    final_results = []

    for model_cfg in MODEL_CONFIGS:
        if model_cfg["skip"]:
            print(f"\n  [{model_cfg['id']}/{len(MODEL_CONFIGS)}] "
                  f"{model_cfg['name']} — [Cached] "
                  f"AUROC={model_cfg['cached_auroc']:.4f}  "
                  f"F1={model_cfg['cached_f1']:.4f}")
            final_results.append({
                "id"   : model_cfg["id"],
                "name" : model_cfg["name"],
                "auroc": model_cfg["cached_auroc"],
                "f1"   : model_cfg["cached_f1"],
            })
        else:
            try:
                auroc, f1, elapsed = run_model(model_cfg)
                final_results.append({
                    "id"   : model_cfg["id"],
                    "name" : model_cfg["name"],
                    "auroc": auroc,
                    "f1"   : f1,
                })
            except Exception as e:
                print(f"\n  ERROR in {model_cfg['name']} : {e}")
                traceback.print_exc()
                final_results.append({
                    "id"   : model_cfg["id"],
                    "name" : model_cfg["name"],
                    "auroc": 0.0,
                    "f1"   : 0.0,
                })

    print_table(final_results)
    plot_results(final_results, RESULTS_DIR)

    print("\n" + "=" * 65)
    print("  Done!")
    print(f"  Results : {RESULTS_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
