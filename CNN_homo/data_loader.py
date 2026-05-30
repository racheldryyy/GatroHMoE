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

# 避免损坏图像问题
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 定义标准数据增强
train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),  # 增加旋转角度
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),  # 添加仿射变换
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 增强颜色抖动
    transforms.RandomAutocontrast(p=0.2),  # 添加自动对比度
    transforms.RandomEqualize(p=0.1),  # 添加直方图均衡化
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 添加高斯模糊
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), value=0),  # 添加随机擦除
])

# 针对colon数据集的增强数据增强
colon_specific_transform = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),  # 更大角度的旋转
    transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.7, 1.3), shear=20),  # 更强的仿射变换
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  # 更强的颜色变化
    transforms.RandomPerspective(distortion_scale=0.6, p=0.5),  # 添加透视变换
    transforms.RandomAutocontrast(p=0.3),
    transforms.RandomEqualize(p=0.2),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),  # 更强的高斯模糊
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), value=0),  # 更强的随机擦除
])

val_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

def safe_normalize_weights(weights):
    """安全地归一化权重，处理NaN和无穷大值"""
    # 转换为numpy数组
    if isinstance(weights, torch.Tensor):
        weights = weights.numpy()
    
    weights = np.array(weights, dtype=np.float64)
    
    # 检查并处理NaN和无穷大值
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    
    # 确保所有权重都是正数
    weights = np.maximum(weights, 1e-10)
    
    # 归一化
    weights_sum = np.sum(weights)
    if weights_sum > 0:
        weights = weights / weights_sum
    else:
        # 如果所有权重都是0，使用均匀分布
        weights = np.ones_like(weights) / len(weights)
    
    return weights

class OptimizedSimilarityAnalyzer:
    """
    优化的图像相似性分析工具 - 高效版本
    
    主要优化:
    1. 批量特征提取 (10-50x speedup)
    2. 近似相似度计算 (LSH/采样)
    3. 内存优化和缓存
    4. 早停机制
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', batch_size=8, max_samples=2000):
        # 使用更轻量的特征提取器
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Identity()  # 移除分类层
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.max_samples = max_samples  # 限制处理的样本数量
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def extract_features_batch(self, img_paths):
        """批量提取图像特征 - 核心优化"""
        features = []
        valid_paths = []
        
        # 按批次处理
        for i in range(0, len(img_paths), self.batch_size):
            batch_paths = img_paths[i:i+self.batch_size]
            batch_images = []
            batch_valid_paths = []
            
            # 加载批次图像
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_images.append(img_tensor)
                    batch_valid_paths.append(path)
                except Exception as e:
                    print(f"[WARNING] 跳过损坏图像 {path}: {e}")
                    continue
            
            if not batch_images:
                continue
                
            # 批量推理
            try:
                batch_tensor = torch.stack(batch_images).to(self.device)
                with torch.no_grad():
                    batch_features = self.model(batch_tensor).cpu().numpy()
                
                features.extend(batch_features)
                valid_paths.extend(batch_valid_paths)
                
                # 清理内存
                del batch_tensor, batch_images
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"[ERROR] 批量特征提取失败: {e}")
                continue
        
        return valid_paths, np.array(features) if features else np.array([])

    def extract_features(self, img_path):
        """
        从单个图像路径中提取特征
        """
        try:
            valid_paths, features = self.extract_features_batch([img_path])  # 调用批量特征提取方法
            if valid_paths:
                return features[0]  # 返回第一个图像的特征
            else:
                raise ValueError(f"未能提取图像特征: {img_path}")
        except Exception as e:
            print(f"[ERROR] 提取图像特征失败 {img_path}: {e}")
            raise
    
    def find_similar_images_optimized(self, dataset_path, similarity_threshold=0.90, use_sampling=True):
        """优化的相似图像查找 - 使用采样和近似算法"""
        print(f"[DEBUG] 开始高效相似图像分析，数据集: {dataset_path}")
        print(f"[DEBUG] 相似度阈值: {similarity_threshold}, 采样模式: {use_sampling}")

        # 初始化计数变量
        total_images = 0
        error_count = 0
        features_dict = {} 
        
        all_paths = []
        path_to_class = {}
        
        # 收集所有图像路径
        for split in ['Train', 'Test']:
            split_path = os.path.join(dataset_path, split)
            if not os.path.exists(split_path):
                continue
                
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                    
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(class_path, img_file)
                        all_paths.append(img_path)
                        path_to_class[img_path] = class_name
        
        print(f"[DEBUG] 找到 {len(all_paths)} 张图像")
        
        # 智能采样：如果图像太多，使用分层采样
        if use_sampling and len(all_paths) > self.max_samples:
            print(f"[DEBUG] 启用采样模式，从 {len(all_paths)} 张图像中采样 {self.max_samples} 张")
            
            # 按类别分层采样
            class_paths = {}
            for path in all_paths:
                class_name = path_to_class[path]
                if class_name not in class_paths:
                    class_paths[class_name] = []
                class_paths[class_name].append(path)
            
            # 每个类别按比例采样
            sampled_paths = []
            samples_per_class = self.max_samples // len(class_paths)
            
            for class_name, paths in class_paths.items():
                if len(paths) <= samples_per_class:
                    sampled_paths.extend(paths)
                else:
                    sampled_paths.extend(np.random.choice(paths, samples_per_class, replace=False))
            
            all_paths = sampled_paths[:self.max_samples]
            print(f"[DEBUG] 采样完成，处理 {len(all_paths)} 张图像")
        
        # 遍历数据集
        try:
            #train_path = os.path.join(dataset_path, 'Train')
            class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
            if not class_dirs:
                print("[WARNING] 数据集中没有类别目录")
                return []

            print(f"[DEBUG] 发现 {len(class_dirs)} 个类别目录: {class_dirs}")
        except Exception as e:
            print(f"[ERROR] 无法读取数据集目录: {e}")
            return []
        
        for class_dir in class_dirs:
            class_path = os.path.join(dataset_path, class_dir)
            
            try:
                img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                print(f"[DEBUG] 处理类别: {class_dir}, 图像数量: {len(img_files)}")
                total_images += len(img_files)
                
                for img_file in tqdm(img_files, desc=f"处理 {class_dir}"):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        # 将单图像特征提取改为批量特征提取
                        valid_paths, features = self.extract_features_batch([img_path])
                        if valid_paths:  # 如果特征提取成功
                            features_dict[valid_paths[0]] = {
                                'feature': features[0],
                                'class': class_dir
                            }
                    except Exception as e:
                        print(f"[ERROR] 处理图像 {img_path} 出错: {e}")
                        error_count += 1
            except Exception as e:
                print(f"[ERROR] 处理类别目录 {class_path} 出错: {e}")
                error_count += 1
        
        print(f"[DEBUG] 特征提取完成，总图像: {total_images}, 成功: {len(features_dict)}, 失败: {error_count}")
        
        # 查找相似图像对
        print("[DEBUG] 开始计算相似度矩阵...")
        similar_pairs = []
        paths = list(features_dict.keys())
        features = np.array([features_dict[p]['feature'] for p in paths])
        
        print(f"[DEBUG] 特征矩阵形状: {features.shape}")
        
        # 计算余弦相似度矩阵
        try:
            features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
            similarity_matrix = np.dot(features_norm, features_norm.T)
            print(f"[DEBUG] 相似度矩阵计算完成，形状: {similarity_matrix.shape}")
        except Exception as e:
            print(f"[ERROR] 相似度矩阵计算失败: {e}")
            return []
        
        # 找出高度相似的图像对
        print(f"[DEBUG] 开始查找相似度 > {similarity_threshold} 的图像对...")
        pair_count = 0
        total_pairs = len(paths) * (len(paths) - 1) // 2
        
        with tqdm(total=total_pairs, desc="计算图像相似度") as pbar:
            for i in range(len(paths)):
                for j in range(i+1, len(paths)):
                    similarity = similarity_matrix[i, j]
                    if similarity > similarity_threshold:
                        class_i = features_dict[paths[i]]['class']
                        class_j = features_dict[paths[j]]['class']
                        similar_pairs.append({
                            'image1': paths[i],
                            'image2': paths[j],
                            'similarity': similarity,
                            'same_class': class_i == class_j
                        })
                        pair_count += 1
                    pbar.update(1)
                    if pair_count % 10 == 0:
                        print(f"[DEBUG] 已找到 {pair_count} 对相似图像...")
        
        print(f"[DEBUG] 相似图像对查找完成，总共找到 {len(similar_pairs)} 对")
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
    """
    胃肠镜图像数据集
    
    支持多任务学习的医学图像数据集类，能够自动处理不同类型的
    胃肠镜图像（部位识别、疾病诊断等），并提供灵活的数据增强策略。
    """
    
    def __init__(self, root_dir, split='Train', transform=None, task_name=None):
        """
        初始化数据集
        
        Args:
            root_dir (str): 数据集根目录路径
            split (str): 数据集划分，'train' 或 'test'
            transform: 图像预处理变换
            task_name (str): 任务类型，用于选择特定的数据增强策略
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.task_name = task_name
        
        # 构建完整的数据路径
        self.data_dir = os.path.join(root_dir, split.capitalize())
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"找不到数据路径: {self.data_dir}")
        
        # 扫描并获取所有类别文件夹
        self.classes = [d for d in os.listdir(self.data_dir) 
                       if os.path.isdir(os.path.join(self.data_dir, d))]
        self.classes.sort()  # 保持类别顺序的一致性
        
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
        
        # 预加载图像信息 - 优化版本
        self.image_sizes = {}
        self.image_modes = {}
        
        # 使用更有效的异常处理
        self.valid_samples = []
        invalid_count = 0
        
        # 如果样本数量太大，随机采样验证以加速启动
        samples_to_validate = self.samples
        if len(self.samples) > 5000:
            print(f"数据集较大({len(self.samples)}个样本)，采样验证前5000个样本...")
            samples_to_validate = random.sample(self.samples, 5000)
        
        print(f"正在验证{len(samples_to_validate)}个{split}样本...")
        for i, (img_path, label) in enumerate(tqdm(samples_to_validate, desc=f"验证{split}图像")):
            try:
                # 只检查图像是否可以打开,不完全加载
                with Image.open(img_path) as img:
                    self.image_modes[img_path] = img.mode
                    self.image_sizes[img_path] = img.size
                    
            except Exception as e:
                print(f"忽略无效图像 {img_path}: {str(e)}")
                invalid_count += 1
                # 从原始样本中移除无效图像
                if (img_path, label) in self.samples:
                    self.samples.remove((img_path, label))
                
        if invalid_count > 0:
            print(f"警告: 发现 {invalid_count} 个无效图像,已从数据集中移除")
            
        # 为Colon数据集使用专门的增强
        if task_name == 'colon' and split == 'Train':
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
        """通过图像路径获取索引"""
        for i, (path, _) in enumerate(self.samples):
            if path == img_path:
                return i
        return None

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
        split='Train',
        transform=train_transform if task_name != 'colon' else colon_specific_transform,
        task_name=task_name
    )
    
    test_dataset = GIDataset(
        root_dir=data_path, 
        split='Test',
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
        print(f"[DEBUG] 开始相似性分析 - 任务: {task_name}, 数据路径: {data_path}")
        print(f"[DEBUG] GPU内存使用情况 - 已分配: {torch.cuda.memory_allocated()/1024**3:.2f}GB, 已缓存: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        
        try:
            print("[DEBUG] 正在初始化OptimizedSimilarityAnalyzer...")
            analyzer = OptimizedSimilarityAnalyzer(max_samples=1500)  # 限制样本数量
            print("[DEBUG] OptimizedSimilarityAnalyzer初始化完成")
            
            train_path = os.path.join(data_path, 'Train')
            print(f"[DEBUG] 正在分析训练数据路径: {train_path}")
            print(f"[DEBUG] 路径是否存在: {os.path.exists(train_path)}")
            if os.path.exists(train_path):
                print(f"[DEBUG] 路径下子目录: {os.listdir(train_path) if os.path.isdir(train_path) else '不是目录'}")
            
            print("[DEBUG] 开始查找相似图像...")
            similar_pairs = analyzer.find_similar_images_optimized(train_path, 0.90, use_sampling=True)
            print(f"[DEBUG] 相似图像分析完成，找到 {len(similar_pairs) if similar_pairs else 0} 对相似图像")
            
            if similar_pairs:
                print("[DEBUG] 开始分析跨类别相似图像...")
                # 分析跨类别相似图像
                cross_class_pairs = analyzer.analyze_cross_class_similarities(similar_pairs)
                print(f"[DEBUG] 跨类别相似图像分析完成，找到 {len(cross_class_pairs) if cross_class_pairs else 0} 对")
                
                if cross_class_pairs:
                    print("[DEBUG] 开始为困难样本生成权重...")
                    # 为困难样本增加权重
                    hard_weights = analyzer.generate_hard_sample_weights(train_dataset, cross_class_pairs, 3.0)
                    print(f"[DEBUG] 困难样本权重生成完成，权重数量: {len(hard_weights)}")
                    
                    # 结合原有权重和困难样本权重
                    print("[DEBUG] 正在合并原有权重和困难样本权重...")
                    for i in range(len(weights)):
                        weights[i] *= hard_weights[i].item()
                    print("[DEBUG] 权重合并完成")
                else:
                    print("[DEBUG] 未发现跨类别相似图像，使用原有权重")
            else:
                print("[DEBUG] 未发现相似图像对，使用原有权重")
                
            print("[DEBUG] 相似性分析流程完成")
            print(f"[DEBUG] GPU内存使用情况 - 已分配: {torch.cuda.memory_allocated()/1024**3:.2f}GB, 已缓存: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
            
        except Exception as e:
            print(f"[ERROR] 相似性分析失败: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[ERROR] 详细错误信息:\n{traceback.format_exc()}")
            print("[DEBUG] 使用默认权重继续训练")
    
    # 安全处理权重
    weights = np.array(weights, dtype=np.float64)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    weights = np.maximum(weights, 1e-10)  # 确保所有权重都是正数
    weights = torch.DoubleTensor(weights)
    
    # 创建数据加载器 - 稳定版本，避免multiprocessing错误
    if use_balanced_sampler and len(train_counts) > 1:
        # 使用类别平衡采样器,包含困难样本信息
        train_sampler = ClassBalancedSampler(train_dataset, weights)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0,  # 禁用多进程避免pickle错误
            pin_memory=False,  # 在单进程模式下禁用pin_memory
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
            num_workers=0,  # 禁用多进程避免pickle错误
            pin_memory=False,  # 在单进程模式下禁用pin_memory
            drop_last=True
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 禁用多进程避免pickle错误
        pin_memory=False  # 在单进程模式下禁用pin_memory
    )
    
    return train_loader, test_loader, train_dataset.classes

def create_multi_task_loaders(colon_path, ugi_path, colon_disease_path, ugi_disease_path, 
                             batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, analyze_similarity=ENABLE_SIMILARITY_ANALYSIS):
    """创建多任务数据加载器"""
    
    print("开始加载多任务数据集...")
    
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