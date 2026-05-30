import os
import torch


# 数据集根目录配置 - 使用相对路径
# 假设数据集位于项目根目录下的MOE_dataset文件夹
#project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # 向上三级到达根目录
#DATA_ROOT = os.path.join(project_root, "Dataset")
DATA_ROOT = "/home/rachel/Rachel/Dataset"
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
os.makedirs(os.path.join(RESULT_PATH, "models"), exist_ok=True)  # 添加结果模型保存目录

# 图像预处理相关参数
INPUT_SIZE = 224
NORM_MEAN = [0.46265157, 0.27411144, 0.2351086]  # 基于训练数据集计算的均值
NORM_STD = [0.29663012, 0.20209971, 0.17599086]   # 基于训练数据集计算的标准差

# GPU配置 - 单GPU模式
def get_gpu_config():
    """GPU配置，只使用单个GPU避免DataParallel问题"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        
        print(f"Found {num_gpus} GPU(s):")
        for i, name in enumerate(gpu_names):
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {name} ({memory:.1f}GB)")
        
        # 根据CPU核心数优化worker数量
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
# 使用全局标志避免多进程重复初始化
if not hasattr(os.environ, '_CNN_GPU_CONFIG_INITIALIZED'):
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
        
        # 设置标志避免重复初始化
        os.environ['_CNN_GPU_CONFIG_INITIALIZED'] = '1'
            
    except Exception as e:
        print(f"GPU config failed: {e}, using fallback settings")
        GPU_IDS = []
        BATCH_SIZE = 16
        NUM_WORKERS = 4
        DEVICE = torch.device("cpu")
        os.environ['_CNN_GPU_CONFIG_INITIALIZED'] = '1'
else:
    # 使用默认配置避免重复初始化
    GPU_IDS = [0] if torch.cuda.is_available() else []
    BATCH_SIZE = 32 if torch.cuda.is_available() else 16
    NUM_WORKERS = 12 if torch.cuda.is_available() else 4
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# 性能优化配置 - 启用相似度分析
USE_OPTIMIZED_SIMILARITY = True    # 启用优化的相似度分析
ENABLE_SIMILARITY_ANALYSIS = True  # 启用相似度分析
SIMILARITY_THRESHOLD = 0.8         # 相似度阈值
USE_FOCAL_LOSS = True             # 暂时禁用Focal Loss以简化训练
FOCAL_LOSS_TYPE = 'balanced'       
FOCAL_GAMMA = 2.5                 
FOCAL_ALPHA = 1.0
# 优化训练参数以提高稳定性，与Transformer保持一致
LEARNING_RATE = 5e-5        # 大幅降低学习率避免NaN
NUM_EPOCHS = 50         # 统一训练轮数
WEIGHT_DECAY = 1e-5     # 进一步降低L2正则化强度

# 强化学习训练配置 - 与Transformer保持一致
RL_LEARNING_RATE = 1e-4  # RL训练的学习率
RL_GAMMA = 0.99          # 奖励折扣因子
RL_BATCH_SIZE = 128      # RL批次大小
RL_EPOCHS = 15           # RL训练轮数
PRETRAIN_EPOCHS = 20     # 预训练轮数
FINETUNE_EPOCHS = 15     # 微调轮数
RL_ENTROPY_COEF = 0.01   # 熵正则化系数

# 任务权重策略 - 改为固定以提高稳定性
TASK_WEIGHT_STRATEGY = "fixed"  # 可选: 'fixed', 'dynamic', 'uncertainty'
TASK_WEIGHT_ALPHA = 0.3         # 权重更新的平滑系数

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

# 混合专家模型（MoE）配置 - 单专家模式
NUM_EXPERTS = 1           # 专家网络的数量（1个ResNet50专家）
TOP_K_EXPERTS = 1         # 单专家模式
HIDDEN_DIM = 256          # 减少隐藏层维度以节省计算
EXPERT_DROPOUT = 0.1      # 减少dropout以加速训练

# 动态任务权重策略
TASK_WEIGHT_STRATEGY = "dynamic"  # 可选: 'fixed', 'dynamic', 'uncertainty'
TASK_WEIGHT_ALPHA = 0.3           # 权重更新的平滑系数

# 专家负载均衡相关参数
LOAD_BALANCE_WEIGHT = 0.1     # 负载均衡损失的权重
LOAD_BALANCE_DECAY = 0.95     # 负载均衡权重的衰减率

# CNN专家模型配置 - 单ResNet50专家
CNN_EXPERT_MODELS = [
    #{'name': 'ResNet50', 'feature_dim': 2048, 'class': 'ResidualCNN', 'params': '25.6M'}
    {'name': 'ResNeXt50-32x4d', 'feature_dim': 2048, 'class': 'ResNeXtCNN', 'params': '25.0M'}
]
# 总参数量: 8M

# 异构专家网络架构选择 - 单ResNet50专家
HETERO_ARCHITECTURES = [
    'ResNeXtCNN'          # 专家1: ResNet50 (唯一专家)
]

# 数据增强参数
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
AUGMENTATION_PROBABILITY = 0.5  # 高级数据增强概率

# Teacher-student model parameters
DISTILLATION_TEMP = 4.0  # 提高温度使软标签更加平滑
ALPHA = 0.5  # Balance between teacher and hard label loss

# Random seed
SEED = 42

# Lancet journal color scheme
# 可视化配置
FIG_SIZE = (10, 6)  # 图表大小
DPI = 300  # 图表DPI
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

# Lancet pastel color palette for more aesthetically pleasing visualizations
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

# Lightweight model options
LIGHTWEIGHT_MODELS = ["mobilenetv3_small", "efficientnet_b0", "shufflenet_v2_x0_5"]

# CNN-ViT hybrid models
HYBRID_MODELS = ["efficientnet_b0", "vit_b_16"]

# Visualization settings
FIG_SIZE = (12, 10)
DPI = 300

# Font settings for visualizations
FONT_FAMILY = "Times New Roman"
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10

# Evaluation settings
EVAL_METRICS = ["accuracy", "precision", "recall", "f1-score", "auc"]