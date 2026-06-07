"""
PatchCore Ablation Study — Backbone & Layer Comparison
Experiment Objective: Find the optimal feature extraction combination for high-reflective metal textures
"""
# ── Patch timm bypass HuggingFace ──
import timm.models._hub as _timm_hub
import torch as _torch, os as _os

_WEIGHTS_DIR = "home/wwww/models"

_orig_load = _timm_hub.load_state_dict_from_hf

def _patched_load(model_id, *args, **kwargs):
    fname = model_id.replace("/", "--") + ".safetensors"
    local = _os.path.join(_WEIGHTS_DIR, fname)
    local2 = _os.path.join(_WEIGHTS_DIR, "model.safetensors")
    if _os.path.exists(local):
        print(f"  [Patch] Loading from: {local}")
        return _torch.load(local, weights_only=True, map_location="cpu")
    elif "resnet50" in model_id:
        local_r50 = "/home/wwww/models/resnet50/model.safetensors"
        if _os.path.exists(local_r50):
            print(f"  [Patch] Loading resnet50 from: {local_r50}")
            return _torch.load(local_r50, weights_only=True, map_location="cpu")
    if "resnet" in model_id and _os.path.exists(local2):
        print(f"  [Patch] Loading resnet from: {local2}")
        return _torch.load(local2, weights_only=True, map_location="cpu")
    return _orig_load(model_id, *args, **kwargs)

_timm_hub.load_state_dict_from_hf = _patched_load
# ── End patch ──

import os
import gc
import time
import inspect
import warnings
import torch
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

warnings.filterwarnings("ignore")

# ============================================================
# ★ Ablation Experiment Matrix
# ============================================================
ABLATION_EXPERIMENTS = [

    # ── Group A: Backbone comparison, image_size=256, layer2+3 ──
    # ── Existing results, all skip=True using cached values ──
    {
        "id"         : "A1",
        "backbone"   : "wide_resnet50_2",
        "layers"     : ["layer2", "layer3"],
        "desc"       : "Baseline (256)",
        "image_size" : 256,
        "skip"       : True,
        "cached"     : {"auroc": 0.9542, "f1": 0.9822},
    },
    {
        "id"         : "A2",
        "backbone"   : "resnet50",
        "layers"     : ["layer2", "layer3"],
        "desc"       : "Lighter backbone (256)",
        "image_size" : 256,
        "skip"       : True,
        "cached"     : {"auroc": 0.9247, "f1": 0.9809},
    },
    {
        "id"         : "A3",
        "backbone"   : "tf_efficientnet_b4",
        "layers"     : ["blocks.2", "blocks.4"],
        "desc"       : "EfficientNet (256)",
        "image_size" : 256,
        "skip"       : True,
        "cached"     : {"auroc": 0.9252, "f1": 0.9815},
    },

    # ── Group B: Layer comparison, image_size=128, backbone fixed to wide_resnet50_2 ──
    # ── Purpose: Verify the value of layer1 (shallow/detail layer) for high-reflective textures ──
    {
        "id"         : "B1",
        "backbone"   : "wide_resnet50_2",
        "layers"     : ["layer2", "layer3"],       # Excluding layer1, as baseline for Group B
        "desc"       : "layer2+3 only (128)",
        "image_size" : 256,                        
        "skip"       : True,
        "cached"     : {"auroc": 0.9552, "f1": 0.9841},
    },
    {
        "id"         : "B2",
        "backbone"   : "wide_resnet50_2",
        "layers"     : ["layer1", "layer2", "layer3"],
        "desc"       : "layer1+2+3 (256) ← Contribution point",
        "image_size" : 64,
        "skip"       : False,
        "cached"     : None,
        # ★ layer1 produces 64x64 feature maps, memory bank explodes, need to limit separately
        "override"   : {
            "coreset_sampling_ratio": 0.01,  # 0.1→0.01 Significantly reduce memory bank size
            "train_batch_size"      : 8,     # 32→8 Reduce peak VRAM during training
            "eval_batch_size"       : 8,     # 32→8
        },
      },
{
    "id"         : "B3",
    "backbone"   : "wide_resnet50_2",
    "layers"     : ["layer1", "layer2"],
    "desc"       : "layer1+2 (256) ← Contribution point",
    "image_size" : 256,
    "skip"       : True,
    "cached"     : {"auroc": 0.8898, "f1": 0.9787},
    "override"   : {
        "coreset_sampling_ratio": 0.01,
        "train_batch_size"      : 8,
        "eval_batch_size"       : 8,
    },
},
     ]
# ============================================================
# Fixed Configuration (unchanged)
# ============================================================
CONFIG = {
    "dataset_root"           : "/home/wwww/data/my_category",
    "dataset_name"           : "my_category",
    "output_dir"             : "/home/wwww/results/ablation",
    "coreset_sampling_ratio" : 0.1,
    "num_neighbors"          : 9,
    "train_batch_size"       : 32,
    "eval_batch_size"        : 32,
    "num_workers"            : 8,
}

# ============================================================
# DataModule Construction (supports separate image_size for each experiment)
# ============================================================
def build_datamodule(cfg, image_size):
    from anomalib.data import Folder

    sig    = inspect.signature(Folder.__init__)
    params = set(sig.parameters.keys())

    kwargs = {
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
        kwargs["image_size"] = (image_size, image_size)
    elif "train_transform" in params:
        from torchvision.transforms import v2 as T
        _t = T.Compose([
            T.Resize((image_size, image_size),
                     interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])
        kwargs["train_transform"] = _t
        kwargs["eval_transform"]  = _t

    return Folder(**kwargs)


# ============================================================
# Single Experiment Run
# ============================================================
def run_one(exp, cfg):
    from anomalib.models import Patchcore
    from anomalib.engine import Engine

    # ★ Merge experiment-specific override parameters
    merged_cfg = cfg.copy()
    if exp.get("override"):
        merged_cfg.update(exp["override"])
        print(f"  ⚙️  Override: {exp['override']}")

    image_size = exp.get("image_size", 256)
    run_dir    = os.path.join(merged_cfg["output_dir"], f"exp_{exp['id']}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Exp [{exp['id']}] backbone={exp['backbone']}")
    print(f"         layers    ={exp['layers']}")
    print(f"         image_size={image_size}")
    print(f"         coreset   ={merged_cfg['coreset_sampling_ratio']}")
    print(f"         batch     ={merged_cfg['train_batch_size']}")
    print(f"         desc      ={exp['desc']}")
    print(f"{'='*65}")

    # ★ Clear GPU memory before experiment starts
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    free_gb = (torch.cuda.get_device_properties(0).total_memory
               - torch.cuda.memory_allocated()) / 1024**3
    print(f"  🖥️  Free VRAM before run: {free_gb:.1f} GB")

    datamodule = build_datamodule(merged_cfg, image_size)
    datamodule.setup()

    model = engine = None
    try:
        model = Patchcore(
            backbone               = exp["backbone"],
            layers                 = exp["layers"],
            coreset_sampling_ratio = merged_cfg["coreset_sampling_ratio"],
            num_neighbors          = merged_cfg["num_neighbors"],
        )
    except Exception as e:
        print(f"  ❌ Model init failed: {e}")
        return None, None, 0

    engine = Engine(
        default_root_dir = run_dir,
        accelerator      = "gpu",
        devices          = 1,
        max_epochs       = 1,
    )

    t0 = time.time()
    try:
        engine.fit(model=model, datamodule=datamodule)
        results = engine.test(model=model, datamodule=datamodule)
        elapsed = time.time() - t0
        metrics = results[0] if results else {}
        auroc   = float(metrics.get("image_AUROC",   0.0))
        f1      = float(metrics.get("image_F1Score", 0.0))
        print(f"\n  ✅ AUROC={auroc:.4f}  F1={f1:.4f}  Time={elapsed/60:.1f}min")
        return auroc, f1, round(elapsed/60, 1)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ❌ Failed: {e}")
        return None, None, round(elapsed/60, 1)
    finally:
        # ★ Completely release GPU memory
        del model
        del engine
        del datamodule
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated()) / 1024**3
        print(f"  🖥️  Free VRAM after cleanup: {free_gb:.1f} GB")
  

# ============================================================
# Summary Table
# ============================================================
def print_ablation_table(all_results):
    print("\n\n" + "=" * 80)
    print("  ABLATION STUDY — Backbone & Layer Comparison")
    print("=" * 80)
    print(f"  {'ID':<5} {'Backbone':<26} {'Layers':<32} {'Size':^6} {'AUROC':^8} {'F1':^8}")
    print("-" * 80)

    baseline_auroc = next(
        (r["auroc"] for r in all_results if r["id"] == "A1" and r["auroc"]), 0
    )

    valid_aurocs = [x["auroc"] for x in all_results if x["auroc"]]
    best_auroc   = max(valid_aurocs) if valid_aurocs else 0

    for r in all_results:
        auroc_str = f"{r['auroc']:.4f}" if r["auroc"] else "  FAIL"
        f1_str    = f"{r['f1']:.4f}"    if r["f1"]    else "  FAIL"
        delta     = ""
        if r["auroc"] and baseline_auroc:
            d = r["auroc"] - baseline_auroc
            delta = "  (baseline)" if r["id"] == "A1" else f"  ({d:+.4f})"
        star = " ★" if r["auroc"] and r["auroc"] == best_auroc else ""
        layers_str = str(r["layers"])
        size_str   = str(r.get("image_size", 256))
        print(f"  {r['id']:<5} {r['backbone']:<26} {layers_str:<32} "
              f"{size_str:^6} {auroc_str:^8} {f1_str:^8}{delta}{star}")

    print("=" * 80)
    if valid_aurocs:
        best = max((r for r in all_results if r["auroc"]), key=lambda x: x["auroc"])
        print(f"\n  🏆 Best: [{best['id']}] {best['backbone']} + {best['layers']}")
        print(f"     AUROC={best['auroc']:.4f}  F1={best['f1']:.4f}")
        print(f"     ({best['desc']})")


# ============================================================
# Visualization
# ============================================================
def plot_ablation(all_results, output_dir):
    try:
        valid  = [r for r in all_results if r["auroc"]]
        aurocs = [r["auroc"] for r in valid]
        f1s    = [r["f1"]    for r in valid]

        # Labels: ID + last layer + size
        labels = []
        for r in valid:
            last_layer = r["layers"][-1]
            size = r.get("image_size", 256)
            labels.append(f"{r['id']}\n{last_layer}\n({size}px)")

        best_auroc = max(aurocs)
        colors = []
        for r in valid:
            if r["id"] == "A1":
                colors.append("#E74C3C")
            elif r["auroc"] == best_auroc:
                colors.append("#2ECC71")
            else:
                colors.append("#5B9BD5")

        x   = np.arange(len(valid))
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("PatchCore Ablation Study — Backbone & Layer Comparison",
                     fontsize=13, fontweight="bold")

        for ax, vals, title in [
            (axes[0], aurocs, "Image AUROC"),
            (axes[1], f1s,    "Image F1 Score"),
        ]:
            bars = ax.bar(x, vals, color=colors, width=0.55,
                          edgecolor="white", linewidth=1.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.003,
                        f"{val:.4f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylim([max(0.5, min(vals) - 0.05), 1.05])
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylabel("Score", fontsize=11)
            ax.grid(True, axis="y", alpha=0.3)

        from matplotlib.patches import Patch
        legend = [
            Patch(facecolor="#E74C3C", label="Baseline (A1)"),
            Patch(facecolor="#2ECC71", label="Best"),
            Patch(facecolor="#5B9BD5", label="Others"),
        ]
        fig.legend(handles=legend, loc="lower center", ncol=3,
                   fontsize=10, bbox_to_anchor=(0.5, 0.01))
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        save_path = os.path.join(output_dir, "ablation_results.png")
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
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("=" * 65)
    print("  PatchCore Ablation Study")
    print(f"  anomalib : {anomalib.__version__}")
    print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    all_results = []

    for exp in ABLATION_EXPERIMENTS:
        if exp["skip"]:
            c = exp["cached"]
            print(f"\n  [{exp['id']}] {exp['backbone']} — [Cached] "
                  f"AUROC={c['auroc']:.4f}  F1={c['f1']:.4f}")
            all_results.append({
                "id": exp["id"], "backbone": exp["backbone"],
                "layers": exp["layers"], "desc": exp["desc"],
                "image_size": exp.get("image_size", 256),
                "auroc": c["auroc"], "f1": c["f1"],
            })
        else:
            auroc, f1, _ = run_one(exp, CONFIG)
            all_results.append({
                "id": exp["id"], "backbone": exp["backbone"],
                "layers": exp["layers"], "desc": exp["desc"],
                "image_size": exp.get("image_size", 256),
                "auroc": auroc, "f1": f1,
            })

    print_ablation_table(all_results)
    plot_ablation(all_results, CONFIG["output_dir"])

    print("\n" + "=" * 65)
    print(f"  Done! Results: {CONFIG['output_dir']}")
    print("=" * 65)


if __name__ == "__main__":
    main()