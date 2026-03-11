
import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. Dataset Loader
# ==========================================
class MVTecDataset(Dataset):
    def __init__(self, root_dir, category="screw", is_train=True, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = [] # 0 正常, 1 异常
        
        phase_dir = os.path.join(root_dir, category, "train" if is_train else "test")
        
        if not os.path.exists(phase_dir):
            raise FileNotFoundError(f"找不到数据集路径，请检查: {phase_dir}")

        for defect_type in os.listdir(phase_dir):
            defect_dir = os.path.join(phase_dir, defect_type)
            if not os.path.isdir(defect_dir): continue
                
            label = 0 if defect_type == "good" else 1
            for img_name in os.listdir(defect_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(defect_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label, img_path

# ==========================================
# 2. Enhanced Feature Extractor
# ==========================================
class AdvancedFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 用 WideResNet50_2，官方 PatchCore 标配
        self.backbone = models.wide_resnet50_2(pretrained=True)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.features = []
        # Hook 提取多尺度特征
        self.backbone.layer1.register_forward_hook(self.hook) # Dim: 256 (专门针对极小划痕)
        self.backbone.layer2.register_forward_hook(self.hook) # Dim: 512
        self.backbone.layer3.register_forward_hook(self.hook) # Dim: 1024
        
        # 加个平滑，防止 layer1 的高频噪点太多
        self.patch_maker = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def hook(self, module, input, output):
        self.features.append(output)

    def forward(self, x):
        self.features = []
        _ = self.backbone(x)
        
        # 把 layer1 和 layer3 对齐到 layer2 的分辨率
        ref_size = self.features[1].shape[-2:] 
        
        resized_features = []
        for feat in self.features:
            smoothed_feat = self.patch_maker(feat) 
            resized = F.interpolate(smoothed_feat, size=ref_size, mode='bilinear', align_corners=False)
            resized_features.append(resized)
            
        # 拼接: 256 + 512 + 1024 = 1792 维
        fused_features = torch.cat(resized_features, dim=1) 
        
        B, C, H, W = fused_features.shape
        fused_features = fused_features.view(B, C, -1).permute(0, 2, 1).contiguous()
        
        return fused_features

# ==========================================
# 3. K-Center Greedy Coreset 
# ==========================================
def k_center_greedy(features, sampling_ratio=0.1):
    device = features.device
    num_samples = features.shape[0]
    coreset_size = int(num_samples * sampling_ratio)
    
    print(f"[*] 开始贪心采样 (Coreset)... 从 {num_samples} 降维到 {coreset_size}")
    
    coreset_idx = [np.random.randint(0, num_samples)]
    min_distances = torch.cdist(features[coreset_idx], features).squeeze()
    
    for _ in tqdm(range(1, coreset_size), desc="构建 Memory Bank"):
        new_idx = torch.argmax(min_distances).item()
        coreset_idx.append(new_idx)
        
        new_dist = torch.cdist(features[new_idx:new_idx+1], features).squeeze()
        min_distances = torch.minimum(min_distances, new_dist)
        
    return features[coreset_idx]

# ==========================================
# 4. Main Model
# ==========================================
class AdvancedPatchCore:
    def __init__(self, device, category, sampling_ratio=0.1, n_neighbors=9):
        self.device = device
        self.category = category
        self.extractor = AdvancedFeatureExtractor().to(device)
        self.sampling_ratio = sampling_ratio
        # K值设为9，之前测过5容易被噪点影响
        self.n_neighbors = n_neighbors 
        self.memory_bank = None

    def fit(self, train_loader):
        # 实用功能：如果本地有存好的 Memory Bank，直接加载，省得每次跑好几分钟
        cache_path = f"memory_bank_{self.category}_ratio{self.sampling_ratio}.pt"
        if os.path.exists(cache_path):
            print(f"[*] 发现本地缓存 {cache_path}，直接加载！")
            self.memory_bank = torch.load(cache_path).to(self.device)
            return

        self.extractor.eval()
        features_list = []
        
        print("[*] 正在提取正常样本特征...")
        t0 = time.time()
        with torch.no_grad():
            for imgs, _, _ in tqdm(train_loader):
                imgs = imgs.to(self.device)
                features = self.extractor(imgs) # [B, N, C]
                features = features.view(-1, features.shape[-1])
                features_list.append(features.cpu())
                
        all_features = torch.cat(features_list, dim=0).to(self.device)
        print(f"[*] 特征提取完毕，耗时 {time.time()-t0:.2f}s")
        
        # 降维采样
        self.memory_bank = k_center_greedy(all_features, self.sampling_ratio)
        
        # 存到本地，下次直接用
        torch.save(self.memory_bank.cpu(), cache_path)
        print(f"[*] Memory Bank 构建完毕并保存到本地！尺寸: {self.memory_bank.shape}")
        self.memory_bank = self.memory_bank.to(self.device)

    def predict(self, test_loader):
        self.extractor.eval()
        image_scores = []
        image_labels = []
        
        print("[*] 开始推理测试集...")
        with torch.no_grad():
            for imgs, labels, _ in tqdm(test_loader):
                imgs = imgs.to(self.device)
                features = self.extractor(imgs)
                
                B, N, C = features.shape
                features_flat = features.view(B * N, C)
                
                # KNN 计算与 Memory Bank 的距离
                distances = torch.cdist(features_flat, self.memory_bank)
                topk_values, _ = distances.topk(self.n_neighbors, largest=False, dim=1)
                
                nearest_dist = topk_values[:, 0]
                neighborhood_dist = topk_values.mean(dim=1)
                patch_scores = nearest_dist + neighborhood_dist # 重加权距离
                
                patch_scores = patch_scores.view(B, N)
                
                # 取图片里最异常的 patch 作为整图的异常分
                img_scores, _ = patch_scores.max(dim=1)
                
                image_scores.extend(img_scores.cpu().numpy())
                image_labels.extend(labels.numpy())
                
                # TODO: 后面可以加个把 heatmaps 保存成图片的逻辑（暂时先不画图了）
                # output_dir = r"D:\results\heatmaps"
                
        return np.array(image_scores), np.array(image_labels)

# ==========================================
# 5. 启动入口
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Advanced PatchCore 本地跑分脚本")
    
   
    parser.add_argument("--data_path", type=str, default=r"D:\data\MVTecAD", help="本地MVTec数据集路径")
    parser.add_argument("--category", type=str, default="screw", help="MVTec类别 (screw, metal_nut 等)")
    parser.add_argument("--img_size", type=int, default=224, help="图像分辨率")
    parser.add_argument("--coreset_ratio", type=float, default=0.1, help="降维采样率")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 任务开始！")
    if torch.cuda.is_available():
        print(f"[*] 使用显卡: {torch.cuda.get_device_name(0)}")
    else:
        print("[!] 警告: 没检测到GPU，CPU跑 K-Center Greedy 可能会卡很久...")

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MVTecDataset(args.data_path, category=args.category, is_train=True, transform=transform)
    test_dataset = MVTecDataset(args.data_path, category=args.category, is_train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = AdvancedPatchCore(device, category=args.category, sampling_ratio=args.coreset_ratio)
    model.fit(train_loader)

    scores, labels = model.predict(test_loader)
    
    auroc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
    best_f1 = np.max(f1_scores)
    
    print("\n" + "="*45)
    print(f" 🚀 MVTec Category: [{args.category.upper()}] 评估结果")
    print("="*45)
    print(f"  > AUROC         : {auroc:.4f}")
    print(f"  > Best F1 Score : {best_f1:.4f}")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()
