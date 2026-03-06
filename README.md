

\# Anomaly Detection Project



基于 \[Anomalib](https://github.com/openvinotoolkit/anomalib) 框架，对 MVTec AD 数据集中的 `metal\_nut` 类别进行异常检测，对比三种模型的性能表现。



\## 模型对比



| 模型 | image\_AUROC | image\_F1Score | pixel\_AUROC | pixel\_F1Score |

|------|-------------|---------------|-------------|---------------|

| PADIM (ResNet18) | 0.9604 | 0.9634 | 0.9573 | 0.7056 |

| PADIM (Wide-ResNet50) | 0.9389 | 0.9474 | 0.9751 | 0.7753 |

| EfficientAD | - | - | - | - |



\## 文件说明



| 文件 | 说明 |

|------|------|

| `01\_train\_padim\_resnet18.py` | PADIM ResNet18 训练 + 测试 |

| `02\_train\_padim\_wide\_resnet50.py` | PADIM Wide-ResNet50 训练 + 测试 |

| `03\_train\_efficientad\_compare.py` | EfficientAD 训练 + 三模型汇总对比 |

| `04\_visualize\_results.py` | 结果可视化 |



\## 环境安装



```bash

pip install -r requirements.txt

```



\## 数据集



使用 \[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) 数据集，下载后放至：



```

./datasets/MVTec/

```



