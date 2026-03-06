
import torch
import gc
import os
import pandas as pd
from anomalib.data import MVTecAD
from anomalib.models import EfficientAd
from anomalib.engine import Engine

# 设置显存分配策略
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# 清理显存
try:
    torch.cuda.empty_cache()
    gc.collect()
    print(f"清理后显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
except Exception:
    print("显存清理跳过（无残留）")

# 加载数据集
datamodule = MVTecAD(
    root="./datasets/MVTec",
    category="metal_nut",
    train_batch_size=1,
    eval_batch_size=8,
    num_workers=0,
)
datamodule.setup()
print("数据集加载成功！")

# EfficientAD 训练
model_efficientad = EfficientAd()
engine_efficientad = Engine(max_epochs=1)
engine_efficientad.fit(model=model_efficientad, datamodule=datamodule)
print("训练完成！")

# 测试
res_efficientad = engine_efficientad.test(model=model_efficientad, datamodule=datamodule)
print("EfficientAD 结果:", res_efficientad)

# ─────────────────────────────────────────────────────
# 汇总三模型对比（前两个为之前实验跑出的结果）
# ─────────────────────────────────────────────────────
df = pd.DataFrame([
    {
        "模型":          "PADIM (ResNet18)",
        "image_AUROC":   0.9604,
        "image_F1Score": 0.9634,
        "pixel_AUROC":   0.9573,
        "pixel_F1Score": 0.7056,
    },
    {
        "模型":          "PADIM (Wide-ResNet50)",
        "image_AUROC":   0.9389,
        "image_F1Score": 0.9474,
        "pixel_AUROC":   0.9751,
        "pixel_F1Score": 0.7753,
    },
    {
        "模型":          "EfficientAD",
        "image_AUROC":   res_efficientad[0].get("image_AUROC", None),
        "image_F1Score": res_efficientad[0].get("image_F1Score", None),
        "pixel_AUROC":   res_efficientad[0].get("pixel_AUROC", None),
        "pixel_F1Score": res_efficientad[0].get("pixel_F1Score", None),
    },
])

print("\n📊 三方模型对比结果")
print(df.to_string(index=False))
