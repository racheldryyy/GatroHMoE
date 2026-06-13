import os
import torch

# 数据集存储路径配置 - 使用绝对路径
DATA_ROOT = "/home/xxx/xxx/Dataset"
COLON_DATA_PATH = os.path.join(DATA_ROOT, "colon_site")
UGI_DATA_PATH = os.path.join(DATA_ROOT, "UGI_10_site")
COLON_DISEASE_PATH = os.path.join(DATA_ROOT, "colon_disease")
UGI_DISEASE_PATH = os.path.join(DATA_ROOT, "upper_disease")

# 实验结果保存路径 - 确保保存到models目录
RESULT_PATH = os.path.join(os.path.dirname(__file__), "results")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "models")  # 直接保存到models目录

# 确保所有必需的目录存在
os.makedirs(RESULT_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(os.path.join(RESULT_PATH, "logs"), exist_ok=True)
os.makedirs(os.path.join(RESULT_PATH, "visualizations"), exist_ok=True)

# 图像预处理相关参数
INPUT_SIZE = 224
NORM_MEAN = [0.46265157, 0.27411144, 0.2351086]
NORM_STD = [0.29663012, 0.20209971, 0.17599086]

# 模型训练基础参数
# GPU配置 - RTX 5090优化版本
def get_gpu_config():
    """RTX 5090优化配置，充分利用32GB显存"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        
        print(f"Found {num_gpus} GPU(s):")
        for i, name in enumerate(gpu_names):
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {name} ({memory:.1f}GB)")
        
        # RTX 5090优化配置
        import multiprocessing
        max_workers = multiprocessing.cpu_count()
        gpu_ids = [0]
        
        # 根据GPU型号自动优化batch size
        primary_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if primary_gpu_memory >= 30:  # RTX 5090 32GB
            batch_size = 8  # 大幅提升batch size充分利用5090
            num_workers = min(16, max_workers // 2)  # 优化worker数量
            print(f"RTX 5090 detected! Using optimized batch size: {batch_size}")
        elif primary_gpu_memory >= 20:  # RTX 4090 24GB
            batch_size = 8
            num_workers = min(12, max_workers // 2)
        elif primary_gpu_memory >= 10:  # RTX 3090/4080 12-16GB
            batch_size = 8
            num_workers = min(8, max_workers // 2)
        else:
            batch_size = 8
            num_workers = 4
        
        print(f"Using single GPU mode: GPU 0 only")
        print(f"Optimized workers: {num_workers} (max available: {max_workers})")
        print(f"Optimized batch size: {batch_size} for GPU memory: {primary_gpu_memory:.1f}GB")
            
        return gpu_ids, batch_size, num_workers, torch.device(f"cuda:0")
    else:
        print("No CUDA GPU detected, using CPU")
        return [], 16, 4, torch.device("cpu")

# Initialize GPU configuration - 简化版本避免阻塞
try:
    GPU_IDS, BATCH_SIZE, NUM_WORKERS, DEVICE = get_gpu_config()
    
    print(f"GPU config: GPU_IDS={GPU_IDS}, BATCH_SIZE={BATCH_SIZE}, NUM_WORKERS={NUM_WORKERS}")
    print(f"Primary device: {DEVICE}")
    print(f"Estimated memory requirement: {BATCH_SIZE * len(GPU_IDS) * 0.2:.1f}GB per GPU")
    
    # 简化GPU验证 - 避免潜在的死循环
    if torch.cuda.is_available() and len(GPU_IDS) > 0:
        print("GPU test passed")
    elif torch.cuda.is_available():
        print("GPU available but not configured")
    else:
        print("No GPU available, using CPU")
        
except Exception as e:
    print(f"GPU config failed: {e}, using fallback settings")
    GPU_IDS = []
    BATCH_SIZE = 16
    NUM_WORKERS = 3
    DEVICE = torch.device("cpu")

# 优化训练参数以提高稳定性，避免NaN
LEARNING_RATE = 5e-5        # 大幅降低学习率避免NaN
NUM_EPOCHS = 50         # 统一训练轮数
WEIGHT_DECAY = 1e-5     # 进一步降低L2正则化强度

# 混合精度训练开关 - 16GB内存必须开启
USE_AMP = True

# 梯度检查点开关 - 16GB内存建议开启
USE_GRADIENT_CHECKPOINTING = True

# 内存优化配置
MEMORY_OPTIMIZATION = True
MAX_MEMORY_USAGE_PERCENT = 85  # 最大内存使用率阈值

# 早停策略配置
PATIENCE = 10           # 等待改进的轮数
MIN_DELTA = 0.001       # 认定为改进的最小阈值

# 标签平滑化程度
LABEL_SMOOTHING = 0.1

# 学习率调度相关
LR_WARMUP_EPOCHS = 5    # 学习率预热的轮数
LR_MIN_FACTOR = 0.01    # 最小学习率相对于初始学习率的比例

# 混合专家模型（MoE）配置 - RTX 5090优化
NUM_EXPERTS = 6           # 专家网络的数量（6个Transformer专家）
TOP_K_EXPERTS = 3         # 增加激活专家数量充分利用5090算力
HIDDEN_DIM = 512          # 增加隐藏层维度充分利用GPU
EXPERT_DROPOUT = 0.05     # 降低dropout提升速度

# 动态任务权重策略 - 改为固定以提高稳定性
TASK_WEIGHT_STRATEGY = "fixed"  # 可选: 'fixed', 'dynamic', 'uncertainty'
TASK_WEIGHT_ALPHA = 0.3         # 权重更新的平滑系数

# 专家负载均衡相关参数
LOAD_BALANCE_WEIGHT = 0.1     # 负载均衡损失的权重
LOAD_BALANCE_DECAY = 0.95     # 负载均衡权重的衰减率

# 性能优化配置
USE_OPTIMIZED_SIMILARITY = False   # 完全禁用相似度分析
ENABLE_SIMILARITY_ANALYSIS = False # 完全禁用相似度分析
USE_FOCAL_LOSS = False             # 暂时禁用Focal Loss以简化训练
FOCAL_LOSS_TYPE = 'balanced'       
FOCAL_GAMMA = 2.5                 
FOCAL_ALPHA = 1.0

# Transformer专家模型配置 - 使用较小版本以适应内存限制
TRANSFORMER_EXPERT_MODELS = [
    {'name': 'ViT-Small', 'feature_dim': 384, 'class': 'vit_small_patch16_224', 'params': '22.1M'},
    {'name': 'Swin-Tiny', 'feature_dim': 768, 'class': 'swin_tiny_patch4_window7_224', 'params': '28.3M'},
    {'name': 'ConvNeXt-Tiny', 'feature_dim': 768, 'class': 'convnext_tiny', 'params': '28.6M'},
    {'name': 'MaxViT-Tiny', 'feature_dim': 512, 'class': 'maxvit_tiny_tf_224', 'params': '31.0M'},
    {'name': 'BEiT-Base', 'feature_dim': 768, 'class': 'beit_base_patch16_224', 'params': '86.7M'},
    {'name': 'DeiT-Small', 'feature_dim': 384, 'class': 'deit_small_patch16_224', 'params': '22.1M'}
]
# 总参数量: 218.8M (修复BEiT模型名称后)

# 异构专家网络架构选择 - 对应上述较小模型
HETERO_ARCHITECTURES = [
    'vit_small_patch16_224',          # 专家1: ViT-Small
    'swin_tiny_patch4_window7_224',   # 专家2: Swin-Tiny
    'convnext_tiny',                  # 专家3: ConvNeXt-Tiny
    'maxvit_tiny_tf_224',             # 专家4: MaxViT-Tiny
    'beit_base_patch16_224',          # 专家5: BEiT-Base (使用Base版本，Small不存在)
    'deit_small_patch16_224'          # 专家6: DeiT-Small
]

# 数据增强策略配置
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
AUGMENTATION_PROBABILITY = 0.5  # 数据增强应用概率

# Teacher-student model parameters
DISTILLATION_TEMP = 4.0  # 知识蒸馏温度参数
ALPHA = 0.5  # 教师网络与真实标签的损失平衡系数

# 随机种子设置
SEED = 42

# 可视化配置
FIG_SIZE = (10, 6)  # 图表大小
DPI = 300  # 图表DPI

# 科研论文配色方案
LANCET_COLORS = [
    "#00468BFF",  # Deep blue
    "#ED0000FF",  # Red
    "#42B540FF",  # Green
    "#0099B4FF",  # Cyan
    "#925E9FFF",  # Purple
    "#FDAF91FF",  # Light orange
    "#AD002AFF",  # Dark red
    "#ADB6B6FF"   # Gray
]

# 柔和配色方案，用于美观的可视化效果
LANCET_PASTEL_COLORS = [
    "#6AADE4",  # Pastel blue
    "#FF9E9E",  # Pastel red
    "#94D6A8",  # Pastel green
    "#87CEEB",  # Pastel cyan
    "#D8BFD8",  # Pastel purple
    "#FFD8C2",  # Pastel orange
    "#E6A9B1",  # Pastel dark red
    "#D3D3D3"   # Pastel gray
]

# 轻量化模型选项
LIGHTWEIGHT_MODELS = ["mobilenetv3_small", "efficientnet_b0", "shufflenet_v2_x0_5"]

# 卷积-Transformer混合模型
HYBRID_MODELS = ["efficientnet_b0", "vit_b_16"]

# 图表可视化配置
FIG_SIZE = (12, 10)
DPI = 300

# 图表字体样式设置
FONT_FAMILY = "Times New Roman"
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10

# 模型评估指标配置
EVAL_METRICS = ["accuracy", "precision", "recall", "f1-score", "auc"]

# ========== 性能优化配置 ==========

# 相似度分析配置 - 已优化以追求最佳效果
ENABLE_SIMILARITY_ANALYSIS = True  # 启用完整相似度分析
USE_OPTIMIZED_SIMILARITY = True     # 使用优化版本
SIMILARITY_MAX_SAMPLES = None       # 移除样本数限制，处理全部样本
SIMILARITY_BATCH_SIZE = 512         # 增大批次大小以提高效率
SIMILARITY_THRESHOLD = 0.8          # 相似度阈值

# Focal Loss配置
USE_FOCAL_LOSS = True               # 启用Focal Loss
FOCAL_LOSS_TYPE = 'balanced'        # 损失函数类型: 'focal', 'balanced', 'adaptive', 'smooth'
FOCAL_ALPHA = 1.0                   # Alpha参数
FOCAL_GAMMA = 2.5                  # Gamma参数，更强的困难样本关注
FOCAL_GAMMA_RANGE = (1.0, 3.0)     # 自适应Focal Loss的gamma范围
FOCAL_SMOOTHING = 0.1               # 标签平滑参数

# 数据加载优化配置 - RTX 5090优化版本
PREFETCH_FACTOR = 16                # 大幅增加预取因子充分利用5090带宽
PERSISTENT_WORKERS = True           # 持久化worker
DATALOADER_TIMEOUT = 300            # 优化超时时间
MULTIPROCESSING_CONTEXT = 'spawn'   # CUDA兼容性

# 内存管理配置 - RTX 5090 32GB优化
MEMORY_THRESHOLD = 90               # 充分利用32GB显存
GRADIENT_ACCUMULATION_STEPS = 2     # 减少梯度累积因为batch size已经足够大
USE_GRADIENT_CHECKPOINTING = False  # 32GB显存无需梯度检查点
MEMORY_CLEANUP_INTERVAL = 1         # 减少清理频率提升性能

# GPU配置
AUTO_SELECT_GPU = True              # 自动选择最佳GPU
MIN_GPU_MEMORY_GB = 2.0             # 最小GPU内存要求
SINGLE_GPU_MODE = True              # 强制单GPU模式
AUTO_BATCH_SIZE = False             # 自动调整批次大小

# 训练优化配置
WARMUP_STEPS = 1000                 # 学习率预热步数
COSINE_SCHEDULE = True              # 使用余弦学习率调度
LABEL_SMOOTHING_EPSILON = 0.1       # 标签平滑参数
DROPOUT_RATE = 0.1                  # Dropout率
WEIGHT_DECAY_EXCLUDE = ['bias', 'LayerNorm.weight']  # 权重衰减排除项

# 模型优化配置 - 优化以追求最佳效果
USE_EFFICIENT_ATTENTION = True      # 使用高效注意力机制
EXPERT_CAPACITY_FACTOR = 2.0        # 增加专家容量因子以处理更多数据
EXPERT_SPARSITY_LOSS_WEIGHT = 0.005 # 减少稀疏性损失权重以优先效果
ROUTING_JITTER_NOISE = 0.05         # 减少路由噪声以提高稳定性

# 实验配置
EXPERIMENT_NAME = "optimized_hetero_moe"  # 实验名称
SAVE_BEST_ONLY = True               # 只保存最佳模型
CHECKPOINT_INTERVAL = 5             # 检查点保存间隔
TENSORBOARD_LOG_DIR = os.path.join(RESULT_PATH, "tensorboard")  # TensorBoard日志目录

# 调试配置
DEBUG_MODE = False                  # 调试模式
PROFILE_MEMORY = False              # 内存分析
PROFILE_TIME = False                # 时间分析
VERBOSE_LOGGING = False             # 详细日志