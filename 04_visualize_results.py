
from PIL import Image
import matplotlib.pyplot as plt
import os

result_path = "./results/Padim/MVTecAD/metal_nut/latest/images"

# 收集所有子文件夹里的图片
all_images = []
for folder in os.listdir(result_path):
    folder_path = os.path.join(result_path, folder)
    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            if img_file.endswith(".png") or img_file.endswith(".jpg"):
                all_images.append(os.path.join(folder_path, img_file))

print(f"共找到 {len(all_images)} 张图片")

# 展示前6张
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, img_path in enumerate(all_images[:6]):
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(
        img_path.split("/")[-2] + "/" + img_path.split("/")[-1],
        fontsize=9
    )
    axes[i].axis("off")

# 隐藏多余的子图
for j in range(len(all_images[:6]), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("results_preview.png", dpi=150)
plt.show()
print("图片已保存为 results_preview.png")
