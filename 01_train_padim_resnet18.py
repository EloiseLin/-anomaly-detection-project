
import sys
sys.setrecursionlimit(10000)

from anomalib.data import MVTecAD
from anomalib.models import Padim
from anomalib.engine import Engine

# 加载数据集
datamodule = MVTecAD(
    root="./datasets/MVTec",
    category="metal_nut",
    train_batch_size=16,
    eval_batch_size=16,
    num_workers=0,
)
datamodule.setup()
print("数据集加载成功！")

# PADIM 模型（默认 ResNet18 backbone）
model_padim_r18 = Padim()

# 训练
engine_padim_r18 = Engine(max_epochs=1)
engine_padim_r18.fit(model=model_padim_r18, datamodule=datamodule)
print("训练完成！")

# 测试并输出结果
results_padim_r18 = engine_padim_r18.test(model=model_padim_r18, datamodule=datamodule)
print("PADIM ResNet18 结果:", results_padim_r18)
