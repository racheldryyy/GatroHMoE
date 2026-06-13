import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from config import LANCET_COLORS, FIG_SIZE, DPI, RESULT_PATH
from collections import defaultdict
from PIL import Image

def set_seed(seed):
    """设置全局随机种子，保证实验结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class ProgressBar:
    """训练进度显示工具"""
    def __init__(self, total, desc="Progress"):
        self.pbar = tqdm(total=total, desc=desc, ncols=100)
        
    def update(self, n=1):
        self.pbar.update(n)
        
    def set_description(self, desc):
        self.pbar.set_description(desc)
        
    def close(self):
        self.pbar.close()

def get_class_names(data_path):
    """从数据集目录中提取类别名称列表"""
    train_path = os.path.join(data_path, "Train")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练数据目录不存在: {train_path}")
    
    class_names = [d for d in os.listdir(train_path) 
                  if os.path.isdir(os.path.join(train_path, d))]
    return sorted(class_names)

def count_samples(data_path):
    """统计各类别数据集中的样本数量分布"""
    train_path = os.path.join(data_path, "Train")
    test_path = os.path.join(data_path, "Test")
    
    class_counts = {"train": {}, "test": {}}
    
    for split, path in [("train", train_path), ("test", test_path)]:
        if not os.path.exists(path):
            print(f"警告: 数据路径不存在 {path}")
            continue
            
        for class_name in os.listdir(path):
            class_dir = os.path.join(path, class_name)
            if os.path.isdir(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
                class_counts[split][class_name] = count
    
    return class_counts

def save_training_curve(train_losses, val_losses, train_accs, val_accs, model_name, result_path=RESULT_PATH):
    """生成并保存训练过程的损失和准确率曲线图"""
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color=LANCET_COLORS[0])
    plt.plot(val_losses, label='Validation Loss', color=LANCET_COLORS[1])
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy', color=LANCET_COLORS[2])
    plt.plot(val_accs, label='Validation Accuracy', color=LANCET_COLORS[3])
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    save_path = os.path.join(result_path, "visualizations", f"{model_name}_training_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"训练曲线图保存位置: {save_path}")

def save_model(model, optimizer, epoch, best_acc, model_name, result_path=RESULT_PATH):
    """保存模型检查点文件，包含模型权重和训练状态"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc
    }
    
    # 保存详细检查点到results目录
    checkpoint_save_path = os.path.join(result_path, "models", f"{model_name}_checkpoint.pth")
    torch.save(checkpoint, checkpoint_save_path)
    
    # 额外保存统一格式的模型到Transformer/models目录，供对比项目使用
    from config import MODEL_SAVE_PATH
    unified_save_path = os.path.join(MODEL_SAVE_PATH, "transformer.pth")
    torch.save(checkpoint, unified_save_path)
    
    print(f"模型检查点已保存至: {checkpoint_save_path}")
    print(f"统一格式模型已保存至: {unified_save_path}")
    
    return unified_save_path

def load_model(model, optimizer, model_name, result_path=RESULT_PATH):
    """从检查点文件恢复模型和训练状态"""
    checkpoint_path = os.path.join(result_path, "models", f"{model_name}_checkpoint.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return 0, 0
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    
    print(f"恢复训练：第{epoch}轮，历史最佳准确率: {best_acc:.4f}")
    return epoch, best_acc

class EarlyStopping:
    """早停机制实现，防止模型过度拟合"""
    def __init__(self, patience=10, min_delta=0, verbose=False):
        """
        初始化早停参数:
            patience (int): 容忍没有改进的轮数
            min_delta (float): 认为有效改进的最小变化
            verbose (bool): 是否输出详细信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def check(self, val_loss):
        """
        判断是否触发早停条件
        
        参数:
            val_loss (float): 当前验证集损失值
            
        返回:
            bool: 需要早停返回True，否则返回False
        """
        score = val_loss
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        # 检查是否有有效改进
        if score > self.best_score - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            if self.verbose and score < self.best_score:
                improvement = self.best_score - score
                print(f'验证指标改进: {self.best_score:.6f} -> {score:.6f} (提升{improvement:.6f})')
            self.best_score = score
            self.counter = 0
            
        return False

class LearningRateScheduler:
    """学习率动态调整工具"""
    def __init__(self, optimizer, init_lr, warmup_epochs=5, max_epochs=50, min_lr_factor=0.01):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = init_lr * min_lr_factor
        
    def step(self, epoch):
        """根据当前轮数调整学习率"""
        if epoch < self.warmup_epochs:
            # 学习率线性预热阶段
            lr = self.init_lr * (epoch + 1) / self.warmup_epochs
        else:
            # 余弦退火调度阶段
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(1.0, progress)
            lr = self.min_lr + 0.5 * (self.init_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        # 应用新的学习率到优化器
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr
    
def calculate_task_weights(val_losses, prev_weights, alpha=0.3):
    """多任务学习中的动态权重计算
    
    参数:
        val_losses (dict): 各任务的验证损失值
        prev_weights (dict): 上一轮的任务权重
        alpha (float): 权重更新的平滑系数
        
    返回:
        dict: 调整后的任务权重分配
    """
    # 计算各任务的相对损失比例
    total_loss = sum(val_losses.values())
    if total_loss == 0:
        # 特殊情况处理：防止除零错误
        relative_losses = {task: 1.0 / len(val_losses) for task in val_losses.keys()}
    else:
        relative_losses = {task: loss / total_loss for task, loss in val_losses.items()}
    
    # 按平滑系数更新权重
    new_weights = {}
    for task in prev_weights.keys():
        if task in relative_losses:
            new_weights[task] = prev_weights[task] * (1 - alpha) + relative_losses[task] * alpha
        else:
            new_weights[task] = prev_weights[task]
    
    # 最终的权重归一化处理
    total_weight = sum(new_weights.values())
    if total_weight > 0:
        new_weights = {task: weight * len(new_weights) / total_weight for task, weight in new_weights.items()}
    
    return new_weights

def save_detailed_curves(train_history, val_history, model_name, epoch, result_path=RESULT_PATH):
    """生成并保存详细的多任务训练曲线和指标报告"""
    plt.figure(figsize=(15, 10), dpi=DPI)
    
    # 提取所有任务名称
    tasks = list(train_history['task_losses'].keys())
    num_tasks = len(tasks)
    
    # 设置子图布局
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=DPI)
    
    # 绘制总体损失变化趋势
    ax = axs[0, 0]
    ax.plot(train_history['total_loss'], label='Train Loss', color=LANCET_COLORS[0])
    ax.plot(val_history['total_loss'], label='Validation Loss', color=LANCET_COLORS[1])
    ax.set_title('Overall Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制各任务的损失曲线对比
    ax = axs[0, 1]
    for i, task in enumerate(tasks):
        ax.plot(train_history['task_losses'][task], 
                label=f'{task.capitalize()} Train',
                color=LANCET_COLORS[i % len(LANCET_COLORS)],
                linestyle='-')
        ax.plot(val_history['task_losses'][task], 
                label=f'{task.capitalize()} Val',
                color=LANCET_COLORS[i % len(LANCET_COLORS)],
                linestyle='--')
    ax.set_title('Task-specific Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制各任务的准确率变化趋势
    ax = axs[1, 0]
    for i, task in enumerate(tasks):
        ax.plot(train_history['acc'][task], 
                label=f'{task.capitalize()} Train',
                color=LANCET_COLORS[i % len(LANCET_COLORS)],
                linestyle='-')
        ax.plot(val_history['acc'][task], 
                label=f'{task.capitalize()} Val',
                color=LANCET_COLORS[i % len(LANCET_COLORS)],
                linestyle='--')
    ax.set_title('Task-specific Accuracy Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制学习率调整趋势
    ax = axs[1, 1]
    if 'lr' in train_history:
        ax.plot(train_history['lr'], label='Learning Rate', color=LANCET_COLORS[4])
        ax.set_title('Learning Rate Schedule')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存可视化图表
    save_path = os.path.join(result_path, "visualizations", f"{model_name}_detailed_curves_{epoch+1}.png")
    plt.savefig(save_path)
    plt.close(fig)
    
    # 生成训练数据报告（CSV格式）
    data = {
        'epoch': list(range(epoch+1)),
        'train_loss': train_history['total_loss'],
        'val_loss': val_history['total_loss']
    }
    
    # 整合各任务的训练指标
    for task in tasks:
        data[f'{task}_train_loss'] = train_history['task_losses'][task]
        data[f'{task}_val_loss'] = val_history['task_losses'][task]
        data[f'{task}_train_acc'] = train_history['acc'][task]
        data[f'{task}_val_acc'] = val_history['acc'][task]

    # 添加学习率
    if 'lr' in train_history:
        data['learning_rate'] = train_history['lr']

    # 填充列到相同长度
    max_length = max(len(value) for value in data.values())
    for key in data:
        if len(data[key]) < max_length:  # 找到不足的列
            data[key] += [None] * (max_length - len(data[key]))  # 用 None 填充
    

    # 生成数据表格并保存
    df = pd.DataFrame(data)
    csv_path = os.path.join(result_path, "logs", f"{model_name}_training_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"详细训练曲线保存位置: {save_path}")
    print(f"训练数据报告保存位置: {csv_path}")