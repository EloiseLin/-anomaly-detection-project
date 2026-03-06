
import torch
import gc
import os
from anomalib.data import MVTecAD
from anomalib.models import Padim
from anomalib.engine import Engine

# 设置显存分配策略
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# 清理显存（防止上一个脚本残留）
try:
    torch.cuda.empty_cache()
    gc.collect()
    print(f"清理后显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
except Exception:
    print("显存清理跳过（无残留）")

# 加载数据集（batch_size 调小，节省显存）
datamodule = MVTecAD(
    root="./datasets/MVTec",
    category="metal_nut",
    train_batch_size=4,
    eval_batch_size=4,
    num_workers=0,
)
datamodule.setup()
print("数据集加载成功！")

# PADIM Wide-ResNet50，限制 n_features 节省显存
model_padim_w50 = Padim(
    backbone="wide_resnet50_2",
    n_features=200,  # 默认550，调小节省显存
)

# 训练
engine_padim_w50 = Engine(max_epochs=1)
engine_padim_w50.fit(model=model_padim_w50, datamodule=datamodule)
print("训练完成！")

# 测试并输出结果
results_padim_w50 = engine_padim_w50.test(model=model_padim_w50, datamodule=datamodule)
print("PADIM Wide-ResNet50 结果:", results_padim_w50)
