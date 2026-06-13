import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import numpy as np
from PIL import Image, ImageFile
from utils import get_class_names, count_samples
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import torchvision.models as models
from config import (
    INPUT_SIZE, NORM_MEAN, NORM_STD, BATCH_SIZE, NUM_WORKERS, SEED,
    MIXUP_ALPHA, CUTMIX_ALPHA, AUGMENTATION_PROBABILITY,
    USE_OPTIMIZED_SIMILARITY, ENABLE_SIMILARITY_ANALYSIS, USE_FOCAL_LOSS
)

# 处理图像文件损坏问题
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 配置基础数据增强策略
train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),  # 随机旋转增强
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),  # 仿射变换增强
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动增强
    transforms.RandomAutocontrast(p=0.2),  # 自动对比度调整
    transforms.RandomEqualize(p=0.1),  # 直方图均衡化
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 高斯模糊增强
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), value=0),  # 随机擦除增强
])

# 为特定数据集定制的加强型数据增强
colon_specific_transform = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),  # 大角度旋转增强
    transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.7, 1.3), shear=20),  # 强化仿射变换
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  # 强化颜色变化
    transforms.RandomPerspective(distortion_scale=0.6, p=0.5),  # 透视变换增强
    transforms.RandomAutocontrast(p=0.3),
    transforms.RandomEqualize(p=0.2),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),  # 加强高斯模糊
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), value=0),  # 加强随机擦除
])

val_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

def safe_normalize_weights(weights):
    """安全的权重归一化处理，防止数值异常"""
    # 数据类型转换
    if isinstance(weights, torch.Tensor):
        weights = weights.numpy()
    
    weights = np.array(weights, dtype=np.float64)
    
    # 处理特殊数值（NaN、无穷大等）
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    
    # 保证权重为正数
    weights = np.maximum(weights, 1e-10)
    
    # 执行权重归一化
    weights_sum = np.sum(weights)
    if weights_sum > 0:
        weights = weights / weights_sum
    else:
        # 全零权重的特殊处理
        weights = np.ones_like(weights) / len(weights)
    
    return weights

class SimilarityAnalyzer:
    """图像相似性分析工具，检测和处理难区分样本"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # 初始化特征提取网络
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # 去除分类层，保留特征提取部分
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def extract_features(self, img_path):
        """从单张图像中提取深度特征"""
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feature = self.model(img).squeeze().cpu().numpy()
        
        return feature
    
    def extract_features_batch(self, batch_paths, batch_size=32):
        """批量提取特征，提高效率"""
        features_list = []
        
        for batch_start in range(0, len(batch_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_paths))
            current_batch = batch_paths[batch_start:batch_end]
            
            batch_images = []
            valid_indices = []
            
            for i, (img_path, _) in enumerate(current_batch):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = self.transform(img)
                    batch_images.append(img)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"加载图像 {img_path} 失败: {e}")
                    continue
            
            if batch_images:
                try:
                    batch_tensor = torch.stack(batch_images).to(self.device)
                    
                    with torch.no_grad():
                        batch_features = self.model(batch_tensor).squeeze().cpu().numpy()
                    
                    # 处理单个样本的情况
                    if len(batch_features.shape) == 1:
                        batch_features = batch_features.reshape(1, -1)
                    
                    # 将特征对应到原始位置
                    batch_results = [None] * len(current_batch)
                    for feature_idx, original_idx in enumerate(valid_indices):
                        batch_results[original_idx] = batch_features[feature_idx]
                    
                    features_list.extend(batch_results)
                    
                except Exception as e:
                    print(f"批处理特征提取失败: {e}")
                    features_list.extend([None] * len(current_batch))
            else:
                features_list.extend([None] * len(current_batch))
        
        return features_list
    
    def compute_similarity_gpu(self, features_dict, similarity_threshold):
        """使用GPU加速计算相似度，追求最佳效果"""
        similar_pairs = []
        paths = list(features_dict.keys())
        n_samples = len(paths)
        
        print(f"使用GPU加速计算 {n_samples} 个样本的相似度...")
        
        # 将特征转换为Tensor并移到GPU
        features_array = np.array([features_dict[path]['feature'] for path in paths])
        features_tensor = torch.from_numpy(features_array).float().to(self.device)
        
        # 计算L2归一化
        features_norm = torch.nn.functional.normalize(features_tensor, p=2, dim=1)
        
        # 使用GPU计算相似度矩阵
        similarity_matrix = torch.matmul(features_norm, features_norm.T)
        
        # 找出高相似度的对
        mask = (similarity_matrix > similarity_threshold) & (torch.triu(torch.ones_like(similarity_matrix, dtype=torch.bool), diagonal=1))
        high_sim_indices = torch.where(mask)
        
        # 将结果转换回 CPU
        similarities = similarity_matrix[high_sim_indices].cpu().numpy()
        indices_i = high_sim_indices[0].cpu().numpy()
        indices_j = high_sim_indices[1].cpu().numpy()
        
        # 构建相似对列表
        for idx, (i, j) in enumerate(zip(indices_i, indices_j)):
            path_i = paths[i]
            path_j = paths[j]
            class_i = features_dict[path_i]['class']
            class_j = features_dict[path_j]['class']
            
            similar_pairs.append({
                'image1': path_i,
                'image2': path_j,
                'similarity': float(similarities[idx]),
                'same_class': class_i == class_j
            })
        
        # 清理GPU内存
        del features_tensor, features_norm, similarity_matrix
        torch.cuda.empty_cache()
        
        return similar_pairs
    
    def find_similar_images(self, dataset_path, similarity_threshold=0.85):
        """找出数据集中高度相似的图像对"""
        import time
        
        print("分析相似图像...")
        features_dict = {}
        processed_count = 0
        start_time = time.time()
        
        # 遍历数据集，使用分批处理避免内存问题
        all_paths = []
        for class_dir in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            print(f"处理类别: {class_dir}")
            for img_file in tqdm(os.listdir(class_path)):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                    
                img_path = os.path.join(class_path, img_file)
                all_paths.append((img_path, class_dir))
        
        print(f"发现 {len(all_paths)} 张图片，开始特征提取...")
        
        # 分批处理特征提取，优化批次大小以提高效率
        batch_size = min(2000, len(all_paths))  # 动态调整批次大小，最大可处理2000张图片
        for batch_start in range(0, len(all_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(all_paths))
            batch_paths = all_paths[batch_start:batch_end]
            
            print(f"处理批次 {batch_start//batch_size + 1}/{(len(all_paths)-1)//batch_size + 1} ({len(batch_paths)} 张图片)")
            
            # 优化特征提取，使用批处理提高效率
            features_batch = self.extract_features_batch(batch_paths)
            for (img_path, class_dir), feature in zip(batch_paths, features_batch):
                if feature is not None:
                    features_dict[img_path] = {
                        'feature': feature,
                        'class': class_dir
                    }
                    processed_count += 1
                else:
                    print(f"处理图像 {img_path} 失败")
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if len(features_dict) == 0:
            print("没有成功提取任何特征，跳过相似度分析")
            return []
            
        print(f"特征提取完成，成功处理 {processed_count} 张图片")
        
        # 优化选择：小数据集使用GPU加速，大数据集使用分块处理
        if processed_count <= 10000 and torch.cuda.is_available():
            return self.compute_similarity_gpu(features_dict, similarity_threshold)
        else:
            return self._compute_similarity_efficiently(features_dict, similarity_threshold, start_time)
    
    def _compute_similarity_efficiently(self, features_dict, similarity_threshold, start_time):
        """高效计算相似度，使用优化的矩阵运算和并行处理"""
        similar_pairs = []
        paths = list(features_dict.keys())
        n_samples = len(paths)
        
        print(f"开始计算 {n_samples} 个样本的相似度...")
        
        # 优化分块处理，根据GPU内存动态调整块大小
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            chunk_size = min(int(gpu_memory * 1000), n_samples)  # 根据GPU内存动态调整
        else:
            chunk_size = min(10000, n_samples)  # CPU模式使用更大的块大小
        
        print(f"使用动态块大小: {chunk_size}")
        
        for i in range(0, n_samples, chunk_size):
            chunk_end_i = min(i + chunk_size, n_samples)
            chunk_paths_i = paths[i:chunk_end_i]
            chunk_features_i = np.array([features_dict[p]['feature'] for p in chunk_paths_i])
            
            for j in range(i, n_samples, chunk_size):
                chunk_end_j = min(j + chunk_size, n_samples)
                chunk_paths_j = paths[j:chunk_end_j]
                chunk_features_j = np.array([features_dict[p]['feature'] for p in chunk_paths_j])
                
                # 优化相似度计算，使用并行化矩阵运算
                # 使用更高效的归一化方法
                norm_i = np.linalg.norm(chunk_features_i, axis=1, keepdims=True)
                norm_j = np.linalg.norm(chunk_features_j, axis=1, keepdims=True)
                
                # 避免除以零的情况
                norm_i = np.maximum(norm_i, 1e-8)
                norm_j = np.maximum(norm_j, 1e-8)
                
                features_norm_i = chunk_features_i / norm_i
                features_norm_j = chunk_features_j / norm_j
                
                # 使用NumPy的优化矩阵乘法
                similarity_matrix = np.matmul(features_norm_i, features_norm_j.T)
                
                # 优化相似对查找，使用向量化操作
                high_sim_indices = np.where(similarity_matrix > similarity_threshold)
                
                for local_i, local_j in zip(high_sim_indices[0], high_sim_indices[1]):
                    global_i = i + local_i
                    global_j = j + local_j
                    
                    # 避免重复比较和自比较
                    if global_i >= global_j:
                        continue
                        
                    similarity = similarity_matrix[local_i, local_j]
                    path_i = chunk_paths_i[local_i]
                    path_j = chunk_paths_j[local_j]
                    class_i = features_dict[path_i]['class']
                    class_j = features_dict[path_j]['class']
                    
                    similar_pairs.append({
                        'image1': path_i,
                        'image2': path_j,
                        'similarity': float(similarity),
                        'same_class': class_i == class_j
                    })
                
                # 清理内存
                del similarity_matrix
            
            # 每处理完一个主块，显示进度
            progress = (chunk_end_i / n_samples) * 100
            elapsed_time = time.time() - start_time
            print(f"相似度计算进度: {progress:.1f}% (耗时 {elapsed_time:.1f}s)")
        
        print(f"相似度分析完成，找到 {len(similar_pairs)} 对相似图像")
        return similar_pairs
    
    def analyze_cross_class_similarities(self, similar_pairs):
        """分析跨类别的相似图像"""
        cross_class_pairs = [p for p in similar_pairs if not p['same_class']]
        
        if not cross_class_pairs:
            print("未发现跨类别相似图像")
            return None
            
        print(f"发现 {len(cross_class_pairs)} 对跨类别相似图像")
        
        # 按相似度排序
        cross_class_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        return cross_class_pairs
    
    def generate_hard_sample_weights(self, dataset, similar_pairs, weight_factor=2.0):
        """为困难样本生成权重"""
        # 初始化所有样本权重为1.0
        weights = torch.ones(len(dataset), dtype=torch.float64)
        
        # 为相似图像对增加权重
        cross_class_pairs = [p for p in similar_pairs if not p['same_class']]
        
        for pair in cross_class_pairs:
            # 找到这些图像在数据集中的索引
            try:
                img1_idx = dataset.get_index_by_path(pair['image1'])
                img2_idx = dataset.get_index_by_path(pair['image2'])
                
                if img1_idx is not None:
                    weights[img1_idx] *= weight_factor
                if img2_idx is not None:
                    weights[img2_idx] *= weight_factor
            except:
                continue
        
        # 安全处理权重
        weights = torch.clamp(weights, min=1e-10, max=1e6)  # 限制权重范围
        
        return weights
    
    def create_augmentation_strategy(self, similar_pairs):
        """根据相似性分析创建增强策略"""
        if not similar_pairs:
            return None
            
        # 根据相似对的特点设计增强策略
        cross_class_pairs = [p for p in similar_pairs if not p['same_class']]
        
        if len(cross_class_pairs) > 0:
            print(f"针对 {len(cross_class_pairs)} 对跨类别相似图像设计增强策略")
            # 返回更强的数据增强
            return colon_specific_transform
        
        return None

class GIDataset(Dataset):
    """胃肠病数据集 - 优化版本"""
    
    def __init__(self, root_dir, split='train', transform=None, task_name=None):
        """
        参数:
            root_dir (string): 数据集根目录
            split (string): 'train' 或 'test'
            transform: 图像转换
            task_name (string): 任务名称,用于应用特定的数据增强
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.task_name = task_name
        
        # 获取数据集路径
        self.data_dir = os.path.join(root_dir, split.capitalize())
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"找不到数据路径: {self.data_dir}")
        
        # 获取类别
        self.classes = [d for d in os.listdir(self.data_dir) 
                       if os.path.isdir(os.path.join(self.data_dir, d))]
        self.classes.sort()  # 确保类别顺序一致
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 收集图像路径和标签
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        # 随机打乱样本顺序
        random.seed(SEED)
        random.shuffle(self.samples)
        
        # 预加载图像信息
        self.image_sizes = {}
        self.image_modes = {}
        
        # 使用更有效的异常处理
        self.valid_samples = []
        invalid_count = 0
        
        print(f"正在验证 {len(self.samples)} 个样本...")
        for i, (img_path, label) in enumerate(tqdm(self.samples, desc=f"验证{task_name or 'unknown'}数据集图像")):
            try:
                # 只检查图像是否可以打开,不完全加载
                with Image.open(img_path) as img:
                    self.image_modes[img_path] = img.mode
                    self.image_sizes[img_path] = img.size
                    self.valid_samples.append((img_path, label))
            except Exception as e:
                print(f"忽略无效图像 {img_path}: {str(e)}")
                invalid_count += 1
                
        if invalid_count > 0:
            print(f"警告: 发现 {invalid_count} 个无效图像,已从数据集中移除")
            self.samples = self.valid_samples
        
        # 创建路径到索引的映射缓存，用于快速查找
        self._path_to_index = {path: i for i, (path, _) in enumerate(self.samples)}
            
        # 为Colon数据集使用专门的增强
        if task_name == 'colon' and split == 'train':
            self.transform = colon_specific_transform
            print(f"为Colon任务使用特定的数据增强")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # 使用已知的图像模式信息提前准备好转换
            image = Image.open(img_path)
            
            # 处理透明调色板图像,避免 PIL warning
            if image.mode == 'P':
                image = image.convert('RGBA')
            
            # 最终统一为 RGB 格式（适配模型）
            image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            print(f"加载图像错误 {img_path}: {str(e)}")
            # 返回数据集中的第一个有效图像作为替代
            return self.__getitem__(0) if idx != 0 else None
    
    def get_index_by_path(self, img_path):
        """通过图像路径获取索引 - 优化版本，使用哈希表查找"""
        return self._path_to_index.get(img_path)

# 优化的mixup函数
def mixup(x, y, alpha=MIXUP_ALPHA):
    """执行MixUp数据增强 - 优化版本"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    
    # 使用torch.randperm代替numpy,避免CPU-GPU传输
    index = torch.randperm(batch_size, device=x.device)

    # 直接在GPU上进行混合
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# 优化的cutmix函数
def cutmix(x, y, alpha=CUTMIX_ALPHA):
    """执行CutMix数据增强 - 优化版本"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    
    # 使用torch.randperm代替numpy,避免CPU-GPU传输
    index = torch.randperm(batch_size, device=x.device)

    # 获取图像尺寸
    h, w = x.size(2), x.size(3)
    
    # 使用GPU计算裁剪区域
    cut_ratio = torch.sqrt(torch.tensor(1. - lam))
    cut_w = (w * cut_ratio).int()
    cut_h = (h * cut_ratio).int()
    
    # 生成随机中心点
    cx = torch.randint(0, w, (1,), device=x.device)[0]
    cy = torch.randint(0, h, (1,), device=x.device)[0]
    
    # 计算裁剪区域
    bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
    bby1 = torch.clamp(cy - cut_h // 2, 0, h)
    bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
    bby2 = torch.clamp(cy + cut_h // 2, 0, h)
    
    # 应用裁剪
    x_cutmix = x.clone()
    x_cutmix[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # 调整混合比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    
    y_a, y_b = y, y[index]
    return x_cutmix, y_a, y_b, lam.item()

# 创建类别平衡采样器 - 修复版本
class ClassBalancedSampler(torch.utils.data.sampler.Sampler):
    """平衡类别的样本采样器"""
    def __init__(self, dataset, weights=None):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.labels = np.array([label for _, label in dataset.samples])
        self.classes = np.unique(self.labels)
        self.num_classes = len(self.classes)
        
        # 根据类别组织索引
        self.class_indices = {}
        for class_id in self.classes:
            self.class_indices[class_id] = np.where(self.labels == class_id)[0]
        
        # 计算最小类别数量，确保不为0
        class_sizes = [len(indices) for indices in self.class_indices.values()]
        self.min_class_size = max(1, min(class_sizes))  # 确保至少为1
        
        # 保存样本权重
        self.weights = weights
        
        # 验证权重
        if self.weights is not None:
            # 确保权重是正数且有限
            if isinstance(self.weights, torch.Tensor):
                self.weights = torch.clamp(self.weights, min=1e-10, max=1e6)
            else:
                self.weights = np.array(self.weights)
                self.weights = np.clip(self.weights, 1e-10, 1e6)
                self.weights = np.nan_to_num(self.weights, nan=1.0, posinf=1.0, neginf=1.0)
    
    def __iter__(self):
        # 从每个类别中随机采样相等数量的样本
        indices = []
        for class_id in self.classes:
            class_indices = self.class_indices[class_id]
            
            # 确保我们有足够的样本
            if len(class_indices) == 0:
                continue
                
            # 如果类别样本数少于最小类别大小，使用有放回采样
            if len(class_indices) < self.min_class_size:
                sample_size = self.min_class_size
                replace = True
            else:
                sample_size = self.min_class_size
                replace = False
            
            # 如果提供了权重,使用加权采样
            if self.weights is not None:
                try:
                    class_weights = self.weights[class_indices]
                    # 安全归一化权重
                    class_weights = safe_normalize_weights(class_weights)
                    
                    selected_indices = np.random.choice(
                        class_indices, 
                        sample_size, 
                        replace=replace, 
                        p=class_weights
                    )
                except (ValueError, RuntimeError) as e:
                    print(f"权重采样失败，使用均匀采样: {e}")
                    selected_indices = np.random.choice(class_indices, sample_size, replace=replace)
            else:
                selected_indices = np.random.choice(class_indices, sample_size, replace=replace)
                
            indices.extend(selected_indices)
        
        # 打乱索引顺序
        np.random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return self.min_class_size * self.num_classes

def get_data_loaders(data_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, 
                    use_balanced_sampler=True, task_name=None, analyze_similarity=False):
    """
    创建并返回训练和测试数据加载器
    
    参数:
        data_path (string): 数据集路径
        batch_size (int): 批次大小
        num_workers (int): 数据加载工作进程数量
        use_balanced_sampler (bool): 是否使用平衡采样器
        task_name (string): 任务名称
        analyze_similarity (bool): 是否分析相似图像
    
    返回:
        train_loader, test_loader: 训练和测试数据加载器
    """
    
    # 创建数据集
    train_dataset = GIDataset(
        root_dir=data_path, 
        split='train',
        transform=train_transform if task_name != 'colon' else colon_specific_transform,
        task_name=task_name
    )
    
    test_dataset = GIDataset(
        root_dir=data_path, 
        split='test',
        transform=val_transform,
        task_name=task_name
    )
    
    # 检查类别分布
    class_counts = count_samples(data_path)
    train_counts = class_counts['train']
    
    # 计算类别权重
    # 根据各类别样本量的倒数计算权重,以平衡不均衡类别
    num_samples = sum(train_counts.values())
    weights = [1.0] * len(train_dataset)
    
    if len(train_counts) > 0:
        class_weights = {}
        for class_name, count in train_counts.items():
            if count > 0:  # 避免除以0
                class_weights[class_name] = num_samples / (len(train_counts) * count)
            else:
                class_weights[class_name] = 1.0  # 为空类别设置默认权重
        
        # 应用样本权重
        for idx, (_, label) in enumerate(train_dataset.samples):
            class_name = train_dataset.classes[label]
            weights[idx] = class_weights.get(class_name, 1.0)
    
    # 分析相似图像并为困难样本设置权重
    if analyze_similarity and task_name == 'colon':
        print("分析Colon数据集中的相似图像...")
        try:
            analyzer = SimilarityAnalyzer()
            similar_pairs = analyzer.find_similar_images(os.path.join(data_path, 'Train'), 0.85)
            
            if similar_pairs:
                # 分析跨类别相似图像
                cross_class_pairs = analyzer.analyze_cross_class_similarities(similar_pairs)
                
                if cross_class_pairs:
                    # 为困难样本增加权重
                    hard_weights = analyzer.generate_hard_sample_weights(train_dataset, cross_class_pairs, 3.0)
                    
                    # 结合原有权重和困难样本权重
                    for i in range(len(weights)):
                        weights[i] *= hard_weights[i].item()
        except Exception as e:
            print(f"相似性分析失败，使用默认权重: {e}")
    
    # 安全处理权重
    weights = np.array(weights, dtype=np.float64)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    weights = np.maximum(weights, 1e-10)  # 确保所有权重都是正数
    weights = torch.DoubleTensor(weights)
    
    # 创建数据加载器
    if use_balanced_sampler and len(train_counts) > 1:
        # 使用类别平衡采样器,包含困难样本信息
        train_sampler = ClassBalancedSampler(train_dataset, weights)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=min(num_workers, 16),  # RTX 5090增加worker数量
            pin_memory=True,  # 启用pin_memory加速GPU传输
            prefetch_factor=6 if num_workers > 0 else None,  # 根据num_workers动态设置
            persistent_workers=True if num_workers > 0 else False,  # 持久worker只在多进程时启用
            drop_last=True
        )
    else:
        # 使用加权随机采样
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(weights), replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=3 if num_workers > 0 else None,  # 根据num_workers动态设置
            persistent_workers=True if num_workers > 0 else False,  # 持久worker只在多进程时启用
            drop_last=True
        )
    
    # test_loader应该在if-else块外定义
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size*2,  # 测试时可用更大批次
        shuffle=False,
        num_workers=min(num_workers, 8),  # RTX 5090优化worker数量
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,  # 增加预取因子
        persistent_workers=True if num_workers > 0 else False  # 持久worker只在多进程时启用
    )
    
    return train_loader, test_loader, train_dataset.classes

def create_multi_task_loaders(colon_path, ugi_path, colon_disease_path, ugi_disease_path, 
                             batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, analyze_similarity=None):
    """创建多任务数据加载器"""
    
    print("开始加载多任务数据集...")
    
    # 检查相似度分析配置
    if analyze_similarity is None:
        analyze_similarity = ENABLE_SIMILARITY_ANALYSIS
    
    # 为colon数据集启用相似性分析
    print("正在加载Colon数据集...")
    colon_train_loader, colon_test_loader, colon_classes = get_data_loaders(
        colon_path, batch_size, num_workers, task_name='colon', analyze_similarity=False  # 禁用相似性分析以加速
    )
    print("✓ Colon数据集加载完成")
    
    print("正在加载UGI数据集...")
    ugi_train_loader, ugi_test_loader, ugi_classes = get_data_loaders(
        ugi_path, batch_size, num_workers, task_name='ugi'
    )
    print("✓ UGI数据集加载完成")
    
    print("正在加载Colon Disease数据集...")
    colon_disease_train_loader, colon_disease_test_loader, colon_disease_classes = get_data_loaders(
        colon_disease_path, batch_size, num_workers, task_name='colon_disease'
    )
    print("✓ Colon Disease数据集加载完成")

    print("正在加载UGI Disease数据集...")
    ugi_disease_train_loader, ugi_disease_test_loader, ugi_disease_classes = get_data_loaders(
        ugi_disease_path, batch_size, num_workers, task_name='ugi_disease'
    )
    print("✓ UGI Disease数据集加载完成")
    print("🎉 所有数据集加载完成！")
    
    return {
        'colon': {
            'train_loader': colon_train_loader,
            'test_loader': colon_test_loader,
            'classes': colon_classes
        },
        'ugi': {
            'train_loader': ugi_train_loader,
            'test_loader': ugi_test_loader,
            'classes': ugi_classes
        },
        'ugi_disease': {
            'train_loader': ugi_disease_train_loader,
            'test_loader': ugi_disease_test_loader,
            'classes': ugi_disease_classes
        },
        'colon_disease': {
            'train_loader': colon_disease_train_loader,
            'test_loader': colon_disease_test_loader,
            'classes': colon_disease_classes
        }
    }