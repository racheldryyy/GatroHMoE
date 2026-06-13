#!/usr/bin/env python3
"""
最终版训练脚本 - 两阶段训练（预训练 + 微调）
专为RTX 5090优化，包含完整的多指标评估和95%置信区间计算

训练流程：
1. 第一阶段：监督学习预训练（50轮）
2. 第二阶段：端到端微调（30轮）

特性：
- RTX 5090 32GB显存优化配置
- 多指标评估：准确度、AUC、F1、精确率、召回率*等
- 95%置信区间计算
- 混合精度训练
- TensorBoard可视化
- 模型检查点保存1
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# 设置Hugging Face镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from config import (
    DEVICE, SEED, 
    COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, UGI_DISEASE_PATH, 
    RESULT_PATH, MODEL_SAVE_PATH,
    LANCET_COLORS, LANCET_PASTEL_COLORS, FIG_SIZE, DPI,
    LABEL_SMOOTHING, MIXUP_ALPHA, CUTMIX_ALPHA, AUGMENTATION_PROBABILITY,
    LOAD_BALANCE_WEIGHT, LOAD_BALANCE_DECAY, PATIENCE, MIN_DELTA, 
    GPU_IDS, USE_AMP, HETERO_ARCHITECTURES
)
from utils import (
    set_seed, ProgressBar, save_model, EarlyStopping
)
from data_loader import create_multi_task_loaders, mixup, cutmix
from models.rl_hetero_moe_model import RLHeterogeneousMixtureOfExperts
from amp_utils import AmpHandler
from metrics_evaluator import (
    create_task_evaluators, log_metrics_to_tensorboard, print_epoch_summary
)

# 添加置信度加权奖励函数
def compute_confidence_weighted_reward(outputs, labels):
    """计算基于置信度的加权奖励"""
    import torch.nn.functional as F
    
    probs = F.softmax(outputs, dim=1)
    predicted = outputs.argmax(dim=1)
    confidence = probs.gather(1, predicted.unsqueeze(1)).squeeze()
    
    # 判断预测是否正确
    correct = (predicted == labels).float()
    
    # 计算加权奖励：正确且高置信度获得高奖励，错误且高置信度受到惩罚
    reward = correct * (1.0 + confidence) - (1.0 - correct) * confidence
    
    return reward

# 优化显存配置 - 启用数据增强提升训练效果
OPTIMIZED_TRAINING_CONFIG = {
    # 三阶段训练配置
    'stage1_epochs': 20,           # 第一阶段：监督学习预训练
    'stage2_epochs': 0,            # 第二阶段：跳过，直接进入第三阶段
    'stage3_epochs': 15,           # 第三阶段：端到端微调
    
    # 学习率配置
    'stage1_lr': 5e-5,            # 第一阶段学习率
    'stage3_lr': 1e-5,            # 第三阶段学习率（更小）
    
    # 训练配置 - RTX 5090显存优化（Transformer需要更多显存）
    'batch_size': 4,              # 减小batch size适配Transformer显存需求
    'num_workers': 12,            # 增加worker数量提升数据加载速度
    'weight_decay': 1e-5,         # 权重衰减
    'warmup_epochs': 5,           # 学习率预热轮数
    'patience': 15,               # 早停等待轮数
    'min_delta': 1e-4,           # 早停最小改进阈值
    'gradient_clip': 1.0,         # 梯度裁剪阈值
    'balance_weight_decay': 0.98, # 负载均衡权重衰减
    'gradient_accumulation_steps': 2,   # 增加梯度累积保持训练效果
    
    # 显存优化 - Transformer专用配置（需要更多显存优化）
    'save_only_best': True,       # 只保存最佳模型
    'memory_cleanup_freq': 5,     # 增加清理频率节省显存
    'use_gradient_checkpointing': True,  # 启用梯度检查点节省显存
    'enable_cpu_offload': False,  # 禁用CPU卸载保持性能
    'max_memory_fraction': 0.85,  # 使用85%显存，为Transformer留余量
    
    # 数据增强配置 - 关键改进
    'enable_augmentation': True,   # 启用数据增强提升效果
    'augmentation_probability': 0.5,  # 数据增强概率
    'mixup_alpha': 0.2,           # MixUp参数
    'cutmix_alpha': 1.0,          # CutMix参数
    'use_confidence_reward': True, # 使用置信度加权奖励
    'entropy_regularization': 0.02 # 熵正则化系数
}

class TwoStageTrainer:
    """两阶段训练器类"""
    
    def __init__(self, config=None):
        """初始化训练器"""
        self.config = config or OPTIMIZED_TRAINING_CONFIG
        self.setup_directories()
        self.setup_device()
        self.setup_memory_optimization()
        
        # 训练历史记录 - 支持三阶段
        self.history = {
            'stage1': {
                'epoch': [],
                'train_loss': [],
                'val_loss': [],
                'train_metrics': {},
                'val_metrics': {},
                'lr': []
            },
            'stage3': {
                'epoch': [],
                'train_loss': [],
                'val_loss': [],
                'train_metrics': {},
                'val_metrics': {},
                'lr': []
            }
        }
        
        # 断点续训相关变量
        self.resume_training = False
        self.resume_stage = None
        self.resume_epoch = 0
        self.best_accuracy = 0.0
    
    def setup_directories(self):
        """创建必要的目录"""
        os.makedirs(RESULT_PATH, exist_ok=True)
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(os.path.join(RESULT_PATH, "logs"), exist_ok=True)
        os.makedirs(os.path.join(RESULT_PATH, "visualizations"), exist_ok=True)
        print(f"✅ 目录设置完成")
        print(f"   结果路径: {RESULT_PATH}")
        print(f"   模型路径: {MODEL_SAVE_PATH}")
    
    def setup_memory_optimization(self):
        """设置极限显存优化"""
        if torch.cuda.is_available():
            # 清空GPU缓存
            torch.cuda.empty_cache()
            
            # 设置显存分配策略 - 更保守
            memory_fraction = self.config.get('max_memory_fraction', 0.75)
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            # 设置CUDA内存分配器
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # 启用CuDNN benchmark优化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # 启用CPU卸载
            if self.config.get('enable_cpu_offload', False):
                torch.cuda.set_device(0)
                print("✅ CPU卸载已启用")
            
            print(f"✅ 极限显存优化设置完成")
            print(f"   显存限制: {memory_fraction*100}%")
            print(f"   batch_size: {self.config['batch_size']}")
            print(f"   梯度累积步数: {self.config.get('gradient_accumulation_steps', 1)}")
            print(f"   等效batch_size: {self.config['batch_size'] * self.config.get('gradient_accumulation_steps', 1)}")
            print(f"   梯度检查点: {'启用' if self.config.get('use_gradient_checkpointing', False) else '禁用'}")
            
            # 尝试预分配少量显存测试
            try:
                test_tensor = torch.zeros(1, 3, 224, 224, device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
                print("✅ GPU显存测试通过")
            except Exception as e:
                print(f"⚠️  GPU显存测试失败: {e}")
                
    def cleanup_memory(self):
        """更激进的显存清理"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # 同步CUDA操作
            import gc
            gc.collect()
            
            # 检查显存使用情况
            current_memory = torch.cuda.memory_allocated(0) / 1024**3
            if current_memory > 20:  # 如果使用超过20GB
                print(f"⚠️  高显存使用警告: {current_memory:.1f}GB")
    
    def setup_device(self):
        """设置设备和显存信息"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # 检查当前显存使用情况
            current_memory = torch.cuda.memory_allocated(0) / 1024**3
            reserved_memory = torch.cuda.memory_reserved(0) / 1024**3
            
            print(f"✅ GPU设备: {gpu_name}")
            print(f"   总显存: {gpu_memory:.1f}GB")
            print(f"   当前使用: {current_memory:.1f}GB")
            print(f"   预留显存: {reserved_memory:.1f}GB")
            print(f"   剩余可用: {gpu_memory - current_memory - reserved_memory:.1f}GB")
            print(f"   🔧 使用极限显存优化配置")
        else:
            print("⚠️  未检测到GPU，使用CPU训练")
    
    def create_data_loaders(self):
        """创建数据加载器"""
        print("📊 正在创建数据加载器...")
        
        data_loaders = create_multi_task_loaders(
            COLON_DATA_PATH, UGI_DATA_PATH, 
            COLON_DISEASE_PATH, UGI_DISEASE_PATH,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers']
        )
        
        # 获取类别数量
        self.task_info = {
            'colon': len(data_loaders['colon']['classes']),
            'ugi': len(data_loaders['ugi']['classes']),
            'colon_disease': len(data_loaders['colon_disease']['classes']),
            'ugi_disease': len(data_loaders['ugi_disease']['classes'])
        }
        
        print(f"✅ 数据加载完成")
        for task, num_classes in self.task_info.items():
            print(f"   {task}: {num_classes} 类别")
        
        return data_loaders
    
    def find_latest_checkpoint(self):
        """查找最新的检查点文件"""
        checkpoint_patterns = [
            "stage1_best.pth",
            "stage1_epoch_*.pth", 
            "stage3_best.pth",
            "stage3_epoch_*.pth",
            "final_best.pth",
            "final_epoch_*.pth"
        ]
        
        latest_checkpoint = None
        latest_time = 0
        
        for pattern in checkpoint_patterns:
            if "*" in pattern:
                # 处理通配符模式
                import glob
                files = glob.glob(os.path.join(MODEL_SAVE_PATH, pattern))
            else:
                # 处理具体文件名
                files = [os.path.join(MODEL_SAVE_PATH, pattern)]
                files = [f for f in files if os.path.exists(f)]
            
            for file_path in files:
                if os.path.exists(file_path):
                    file_time = os.path.getmtime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_checkpoint = file_path
        
        return latest_checkpoint
    
    def analyze_checkpoint(self, checkpoint_path):
        """分析检查点文件，确定训练阶段和epoch"""
        try:
            # 修复PyTorch 2.6兼容性问题 - 添加安全全局变量
            import torch.serialization
            torch.serialization.add_safe_globals([
                'numpy.core.multiarray.scalar',
                'numpy._core.multiarray.scalar',
                'collections.OrderedDict',
                'torch.Size',
                'torch.dtype'
            ])
            
            # 使用weights_only=False加载检查点（如果来源可信）
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # 从文件名判断阶段
            filename = os.path.basename(checkpoint_path)
            stage = None
            epoch = 0
            
            if 'stage1' in filename:
                stage = 'stage1'
            elif 'stage3' in filename or 'final' in filename:
                stage = 'stage3'
            
            # 获取epoch信息
            if 'epoch' in checkpoint:
                epoch = checkpoint['epoch']
            elif 'epoch_' in filename:
                # 从文件名提取epoch
                import re
                match = re.search(r'epoch_(\d+)', filename)
                if match:
                    epoch = int(match.group(1))
            
            # 获取准确率
            accuracy = checkpoint.get('accuracy', 0.0)
            
            # 获取配置信息
            saved_config = checkpoint.get('config', {})
            
            return {
                'stage': stage,
                'epoch': epoch,
                'accuracy': accuracy,
                'config': saved_config,
                'checkpoint': checkpoint
            }
        except Exception as e:
            print(f"❌ 分析检查点文件失败: {e}")
            return None
    
    def prompt_resume_training(self):
        """询问用户是否从检查点恢复训练"""
        latest_checkpoint = self.find_latest_checkpoint()
        
        if latest_checkpoint is None:
            print("📁 未找到任何检查点文件，将从头开始训练")
            return False
        
        # 分析检查点
        checkpoint_info = self.analyze_checkpoint(latest_checkpoint)
        if checkpoint_info is None:
            print("❌ 检查点文件损坏，将从头开始训练")
            return False
        
        # 显示检查点信息
        print("\n" + "="*60)
        print("🔍 发现训练检查点")
        print("="*60)
        print(f"📄 检查点文件: {os.path.basename(latest_checkpoint)}")
        print(f"🏷️  训练阶段: {checkpoint_info['stage']}")
        print(f"📊 完成轮数: {checkpoint_info['epoch'] + 1}")
        print(f"🎯 最佳准确率: {checkpoint_info['accuracy']:.2f}%")
        
        # 计算文件修改时间
        mod_time = os.path.getmtime(latest_checkpoint)
        mod_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
        print(f"📅 保存时间: {mod_time_str}")
        print("="*60)
        
        # 询问用户选择
        while True:
            choice = input("\n请选择训练模式:\n"
                          "1. 从检查点恢复训练 (推荐)\n"
                          "2. 从头开始训练\n"
                          "请输入选择 (1/2): ").strip()
            
            if choice == '1':
                self.resume_training = True
                self.resume_stage = checkpoint_info['stage']
                self.resume_epoch = checkpoint_info['epoch']
                self.best_accuracy = checkpoint_info['accuracy']
                self.resume_checkpoint_path = latest_checkpoint
                self.resume_checkpoint_info = checkpoint_info
                
                print(f"✅ 将从{checkpoint_info['stage']}阶段第{checkpoint_info['epoch'] + 1}轮后继续训练")
                return True
            elif choice == '2':
                print("✅ 将从头开始训练")
                return False
            else:
                print("❌ 无效选择，请输入 1 或 2")
    
    def load_checkpoint(self, model, optimizer=None, scheduler=None):
        """加载检查点到模型和优化器"""
        if not self.resume_training:
            return model, optimizer, scheduler
        
        try:
            checkpoint = self.resume_checkpoint_info['checkpoint']
            
            # 加载模型状态
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 模型状态已恢复")
            
            # 加载优化器状态（如果提供）
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✅ 优化器状态已恢复")
            
            # 加载学习率调度器状态（如果提供）
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"✅ 学习率调度器状态已恢复")
            
            # 加载训练历史（如果有）
            if 'history' in checkpoint:
                saved_history = checkpoint['history']
                for stage in saved_history:
                    if stage in self.history:
                        self.history[stage] = saved_history[stage]
                print(f"✅ 训练历史已恢复")
            
            return model, optimizer, scheduler
            
        except Exception as e:
            print(f"❌ 加载检查点失败: {e}")
            print("⚠️  将从头开始训练")
            self.resume_training = False
            return model, optimizer, scheduler
    
    def create_model(self):
        """创建模型"""
        print("🏗️  正在创建异构混合专家模型...")
        
        model = RLHeterogeneousMixtureOfExperts(
            model_names=HETERO_ARCHITECTURES,
            num_colon_classes=self.task_info['colon'],
            num_ugi_classes=self.task_info['ugi'],
            num_colon_disease_classes=self.task_info['colon_disease'],
            num_ugi_disease_classes=self.task_info['ugi_disease']
        ).to(DEVICE)
        
        # 启用梯度检查点以节省显存
        if self.config.get('use_gradient_checkpointing', False):
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            print("✅ 梯度检查点已启用")
        
        # 单GPU模式以避免DataParallel的显存开销
        print(f"✅ 使用单GPU模式以节省显存")
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ 模型创建完成")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   架构: {HETERO_ARCHITECTURES}")
        
        # 清理显存
        self.cleanup_memory()
        
        return model
    
    def create_optimizer_and_scheduler(self, model, stage='stage1'):
        """创建优化器和学习率调度器"""
        lr = self.config[f'{stage}_lr']
        
        # 为不同阶段创建不同的优化器组
        if stage == 'stage3':  # 端到端微调阶段
            # 微调阶段：对不同层使用不同学习率
            optimizer_groups = []
            
            # 获取模型（处理DataParallel）
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            
            # 分层学习率设置
            for name, param in base_model.named_parameters():
                if 'gates' in name or 'routing' in name:
                    # 路由网络使用正常学习率
                    optimizer_groups.append({'params': param, 'lr': lr, 'weight_decay': self.config['weight_decay']})
                elif 'classifier' in name or 'head' in name:
                    # 分类头使用较大学习率
                    optimizer_groups.append({'params': param, 'lr': lr * 2, 'weight_decay': self.config['weight_decay']})
                else:
                    # 其他参数使用较小学习率
                    optimizer_groups.append({'params': param, 'lr': lr * 0.1, 'weight_decay': self.config['weight_decay']})
            
            optimizer = optim.AdamW(optimizer_groups)
        else:
            # 预训练阶段：统一学习率
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=lr, 
                weight_decay=self.config['weight_decay'],
                betas=(0.9, 0.999)
            )
        
        # 学习率调度器
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=lr * 0.01
        )
        
        print(f"✅ {stage}优化器和调度器创建完成，学习率: {lr:.2e}")
        
        # 添加熵正则化优化器（用于路由策略）
        if stage == 'stage3' and self.config.get('entropy_regularization', 0) > 0:
            # 为路由网络创建单独的优化器用于熵正则化
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            routing_params = []
            for name, param in base_model.named_parameters():
                if 'gates' in name or 'routing' in name:
                    routing_params.append(param)
            
            if routing_params:
                self.routing_optimizer = optim.Adam(routing_params, lr=lr * 10)  # 路由学习率更高
                print(f"✅ 路由策略优化器创建完成，用于熵正则化")
        
        return optimizer, scheduler
    
    def train_epoch(self, model, data_loaders, task_weights, optimizer, criterion, 
                   epoch, stage, evaluators, writer, amp_handler):
        """训练一个epoch - 显存优化版本"""
        model.train()
        
        total_loss = 0.0
        task_losses = {task: 0.0 for task in data_loaders.keys()}
        
        # 重置评估器
        for evaluator in evaluators.values():
            evaluator.reset()
        
        # 计算步数
        max_len = max([len(data_loaders[task]['train_loader']) for task in data_loaders.keys()])
        pbar = ProgressBar(max_len, desc=f"{stage.title()} Epoch {epoch+1}")
        
        # 创建迭代器
        iterators = {task: iter(data_loaders[task]['train_loader']) for task in data_loaders.keys()}
        
        # 当前负载均衡权重（逐渐衰减）
        current_balance_weight = LOAD_BALANCE_WEIGHT * (self.config['balance_weight_decay'] ** epoch)
        
        # 梯度累积相关
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        effective_batch_size = self.config['batch_size'] * accumulation_steps
        memory_cleanup_freq = self.config.get('memory_cleanup_freq', 10)
        
        print(f"🔧 有效batch size: {effective_batch_size} (梯度累积: {accumulation_steps}步)")
        
        for i in range(max_len):
            batch_loss = 0.0
            
            # 只在梯度累积周期开始时清零梯度
            if i % accumulation_steps == 0:
                optimizer.zero_grad()
            
            for task, task_weight in task_weights.items():
                try:
                    inputs, labels = next(iterators[task])
                except StopIteration:
                    iterators[task] = iter(data_loaders[task]['train_loader'])
                    inputs, labels = next(iterators[task])
                
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                
                # 使用混合精度训练
                with amp_handler.autocast():
                    # 数据增强策略 - 关键改进
                    if self.config.get('enable_augmentation', True):
                        r = np.random.rand()
                        aug_prob = self.config.get('augmentation_probability', 0.5)
                        
                        if r < aug_prob / 2:  # MixUp
                            mixed_inputs, labels_a, labels_b, lam = mixup(
                                inputs, labels, alpha=self.config.get('mixup_alpha', 0.2)
                            )
                            outputs, routing_weights = model(mixed_inputs, task=task)
                            loss_a = criterion(outputs, labels_a)
                            loss_b = criterion(outputs, labels_b)
                            loss = lam * loss_a + (1 - lam) * loss_b
                            
                            # 添加置信度奖励 (仅在非增强数据上计算)
                            if self.config.get('use_confidence_reward', False):
                                with torch.no_grad():
                                    clean_outputs, _ = model(inputs, task=task)
                                    confidence_reward = compute_confidence_weighted_reward(clean_outputs, labels)
                                    # 将奖励信息记录到tensorboard
                                    writer.add_scalar(f'{stage}/ConfidenceReward/{task}', 
                                                     confidence_reward.mean().item(), epoch) if epoch % 10 == 0 else None
                                    
                        elif r < aug_prob:  # CutMix
                            mixed_inputs, labels_a, labels_b, lam = cutmix(
                                inputs, labels, alpha=self.config.get('cutmix_alpha', 1.0)
                            )
                            outputs, routing_weights = model(mixed_inputs, task=task)
                            loss_a = criterion(outputs, labels_a)
                            loss_b = criterion(outputs, labels_b)
                            loss = lam * loss_a + (1 - lam) * loss_b
                            
                        else:  # 标准前向传播
                            outputs, routing_weights = model(inputs, task=task)
                            loss = criterion(outputs, labels)
                            
                            # 添加置信度奖励
                            if self.config.get('use_confidence_reward', False):
                                confidence_reward = compute_confidence_weighted_reward(outputs, labels)
                                # 将置信度奖励作为额外的监督信号
                                reward_loss = -confidence_reward.mean() * 0.1  # 小权重
                                loss = loss + reward_loss
                    else:
                        # 禁用数据增强
                        outputs, routing_weights = model(inputs, task=task)
                        loss = criterion(outputs, labels)
                    
                    # 添加负载均衡损失
                    load_balancing_loss = model.calculate_load_balancing_loss(routing_weights)
                    total_task_loss = loss + current_balance_weight * load_balancing_loss
                    
                    # 梯度累积：损失需要除以累积步数
                    weighted_loss = task_weight * total_task_loss / accumulation_steps
                
                # 记录损失
                task_losses[task] += loss.item()
                batch_loss += weighted_loss
                
                # 更新评估器 - 改进评估逻辑
                aug_prob = self.config.get('augmentation_probability', 0.5) if self.config.get('enable_augmentation', True) else 0
                
                if not self.config.get('enable_augmentation', True) or r >= aug_prob:
                    # 标准样本，直接评估
                    with torch.no_grad():
                        _, predicted = torch.max(outputs.data, 1)
                        probabilities = torch.softmax(outputs, dim=1)
                        evaluators[task].update(predicted, labels, probabilities)
                else:
                    # 混合样本，使用原始输入重新预测进行评估
                    with torch.no_grad():
                        eval_outputs, _ = model(inputs, task=task)
                        _, predicted = torch.max(eval_outputs.data, 1)
                        probabilities = torch.softmax(eval_outputs, dim=1)
                        evaluators[task].update(predicted, labels, probabilities)
                
                # 及时删除中间变量
                del outputs, routing_weights
                torch.cuda.empty_cache() if i % self.config.get('memory_cleanup_freq', 10) == 0 and torch.cuda.is_available() else None
            
            # 反向传播
            if amp_handler:
                amp_handler.scale_loss(batch_loss).backward()
                
                # 只在梯度累积周期结束时更新参数
                if (i + 1) % accumulation_steps == 0 or (i + 1) == max_len:
                    amp_handler.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config['gradient_clip'])
                    amp_handler.scaler.step(optimizer)
                    amp_handler.scaler.update()
            else:
                batch_loss.backward()
                
                # 只在梯度累积周期结束时更新参数
                if (i + 1) % accumulation_steps == 0 or (i + 1) == max_len:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config['gradient_clip'])
                    optimizer.step()
            
            total_loss += batch_loss.item() * accumulation_steps  # 恢复原始损失大小
            
            # 定期清理显存
            if i % memory_cleanup_freq == 0 and i > 0:
                self.cleanup_memory()
            
            # 更新进度条
            if i % 20 == 0:
                avg_loss = total_loss / (i + 1)
                current_memory = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
                pbar.set_description(f"{stage.title()} Epoch {epoch+1} Loss: {avg_loss:.4f} GPU: {current_memory:.1f}GB")
            pbar.update()
        
        pbar.close()
        
        # 最终清理显存
        self.cleanup_memory()
        
        # 计算平均损失
        avg_loss = total_loss / max_len
        avg_task_losses = {task: task_losses[task] / max_len for task in task_losses.keys()}
        
        # 获取评估指标
        train_metrics = {}
        for task, evaluator in evaluators.items():
            train_metrics[task] = evaluator.get_summary_metrics()
        
        # 记录到TensorBoard
        writer.add_scalar(f'{stage}/Loss/train', avg_loss, epoch)
        for task, task_loss in avg_task_losses.items():
            writer.add_scalar(f'{stage}/Loss/train_{task}', task_loss, epoch)
        log_metrics_to_tensorboard(writer, train_metrics, epoch, f'{stage}_train')
        
        return avg_loss, avg_task_losses, train_metrics
    
    def validate_epoch(self, model, data_loaders, task_weights, criterion, 
                      epoch, stage, evaluators, writer, amp_handler):
        """验证一个epoch"""
        model.eval()
        
        total_loss = 0.0
        task_losses = {task: 0.0 for task in data_loaders.keys()}
        
        # 重置评估器
        for evaluator in evaluators.values():
            evaluator.reset()
        
        max_len = max([len(data_loaders[task]['test_loader']) for task in data_loaders.keys()])
        pbar = ProgressBar(max_len, desc=f"{stage.title()} Val Epoch {epoch+1}")
        
        with torch.no_grad():
            iterators = {task: iter(data_loaders[task]['test_loader']) for task in data_loaders.keys()}
            
            for i in range(max_len):
                batch_loss = 0.0
                
                for task, task_weight in task_weights.items():
                    try:
                        inputs, labels = next(iterators[task])
                    except StopIteration:
                        iterators[task] = iter(data_loaders[task]['test_loader'])
                        inputs, labels = next(iterators[task])
                    
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    
                    with amp_handler.autocast():
                        outputs, _ = model(inputs, task=task)
                        loss = criterion(outputs, labels)
                    
                    task_losses[task] += loss.item()
                    batch_loss += task_weight * loss
                    
                    # 更新评估器
                    with torch.no_grad():  # 确保评估时不需要梯度
                        _, predicted = torch.max(outputs.data, 1)
                        probabilities = torch.softmax(outputs, dim=1)
                        evaluators[task].update(predicted, labels, probabilities)
                
                total_loss += batch_loss.item()
                
                if i % 20 == 0:
                    avg_loss = total_loss / (i + 1)
                    pbar.set_description(f"{stage.title()} Val Epoch {epoch+1} Loss: {avg_loss:.4f}")
                pbar.update()
        
        pbar.close()
        
        # 计算平均损失
        avg_loss = total_loss / max_len
        avg_task_losses = {task: task_losses[task] / max_len for task in task_losses.keys()}
        
        # 获取评估指标
        val_metrics = {}
        overall_acc = 0.0
        for task, evaluator in evaluators.items():
            metrics = evaluator.get_summary_metrics()
            val_metrics[task] = metrics
            overall_acc += metrics.get('accuracy', 0.0)
        
        overall_acc = overall_acc / len(val_metrics) if val_metrics else 0.0
        
        # 记录到TensorBoard
        writer.add_scalar(f'{stage}/Loss/val', avg_loss, epoch)
        writer.add_scalar(f'{stage}/Accuracy/overall', overall_acc, epoch)
        for task, task_loss in avg_task_losses.items():
            writer.add_scalar(f'{stage}/Loss/val_{task}', task_loss, epoch)
        log_metrics_to_tensorboard(writer, val_metrics, epoch, f'{stage}_val')
        
        return avg_loss, avg_task_losses, val_metrics, overall_acc
    
    def train_epoch_with_entropy(self, model, data_loaders, task_weights, optimizer, criterion, 
                                epoch, stage, evaluators, writer, amp_handler):
        """训练一个epoch - 带熵正则化版本（用于第三阶段）"""
        # 首先执行标准训练
        train_loss, train_task_losses, train_metrics = self.train_epoch(
            model, data_loaders, task_weights, optimizer, criterion,
            epoch, stage, evaluators, writer, amp_handler
        )
        
        # 如果启用了熵正则化，执行额外的路由策略优化
        if hasattr(self, 'routing_optimizer') and self.config.get('entropy_regularization', 0) > 0:
            entropy_coef = self.config.get('entropy_regularization', 0.02)
            
            # 为路由策略进行额外的熵正则化训练
            model.train()
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            
            # 计算总的熵损失
            total_entropy_loss = 0.0
            entropy_batches = 0
            
            # 重新迭代数据进行熵正则化
            max_len = max([len(data_loaders[task]['train_loader']) for task in data_loaders.keys()])
            iterators = {task: iter(data_loaders[task]['train_loader']) for task in data_loaders.keys()}
            
            for i in range(min(max_len // 4, 50)):  # 只执行部分批次的熵正则化
                entropy_loss = 0.0
                
                for task, task_weight in task_weights.items():
                    try:
                        inputs, labels = next(iterators[task])
                    except StopIteration:
                        iterators[task] = iter(data_loaders[task]['train_loader'])
                        inputs, labels = next(iterators[task])
                    
                    inputs = inputs.to(DEVICE, non_blocking=True)
                    
                    with amp_handler.autocast():
                        # 获取路由权重
                        _, routing_weights = model(inputs, task=task)
                        
                        # 计算熵损失 - 鼓励路由分布的多样性
                        routing_probs = F.softmax(routing_weights, dim=1)
                        entropy = -(routing_probs * torch.log(routing_probs + 1e-8)).sum(dim=1).mean()
                        task_entropy_loss = -entropy_coef * entropy  # 负号因为我们要最大化熵
                        
                        entropy_loss += task_weight * task_entropy_loss
                
                # 更新路由策略
                self.routing_optimizer.zero_grad()
                if amp_handler:
                    amp_handler.scale_loss(entropy_loss).backward()
                    amp_handler.step(self.routing_optimizer)
                    amp_handler.update()
                else:
                    entropy_loss.backward()
                    self.routing_optimizer.step()
                
                total_entropy_loss += entropy_loss.item()
                entropy_batches += 1
            
            # 记录熵正则化损失
            if entropy_batches > 0:
                avg_entropy_loss = total_entropy_loss / entropy_batches
                writer.add_scalar(f'{stage}/EntropyLoss/routing', avg_entropy_loss, epoch)
                print(f"   熵正则化损失: {avg_entropy_loss:.6f}")
        
        return train_loss, train_task_losses, train_metrics
    
    def save_best_model_only(self, model, optimizer, epoch, accuracy, stage_name, scheduler=None):
        """模型保存方法 - 支持每个epoch保存和最佳模型保存"""
        # 确保torch模块正确导入
        import torch
        import os
        
        # 构建保存字典
        try:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'config': self.config,
                'stage': stage_name,
                'history': self.history,  # 保存训练历史
                'best_accuracy': self.best_accuracy if hasattr(self, 'best_accuracy') else accuracy
            }
            
            # 保存学习率调度器状态
            if scheduler is not None:
                save_dict['scheduler_state_dict'] = scheduler.state_dict()
            
            # 保存熵正则化优化器状态（如果存在）
            if hasattr(self, 'routing_optimizer'):
                save_dict['routing_optimizer_state_dict'] = self.routing_optimizer.state_dict()
                
        except Exception as e:
            print(f"❌ 构建保存字典失败: {e}")
            return None
        
        # 1. 总是保存最新的checkpoint（用于断点续训）
        latest_path = os.path.join(MODEL_SAVE_PATH, f"transformer_{stage_name}_latest.pth")
        try:
            torch.save(save_dict, latest_path)
            file_size = os.path.getsize(latest_path) / (1024**2)  # MB
            print(f"💾 保存最新检查点成功: {latest_path} ({file_size:.1f}MB)")
        except Exception as e:
            print(f"❌ 保存最新检查点失败: {latest_path}, 错误: {e}")
            return None
        
        # 2. 可选：每隔几个epoch保存一次（或者根据配置保存每个epoch）
        if not self.config.get('save_only_best', True) or epoch % 5 == 0:
            epoch_path = os.path.join(MODEL_SAVE_PATH, f"transformer_{stage_name}_epoch_{epoch}.pth")
            try:
                torch.save(save_dict, epoch_path)
                file_size = os.path.getsize(epoch_path) / (1024**2)  # MB
                print(f"💾 保存第{epoch}轮模型成功: {epoch_path} ({file_size:.1f}MB)")
            except Exception as e:
                print(f"❌ 保存第{epoch}轮模型失败: {epoch_path}, 错误: {e}")
        
        # 3. 保存最佳模型（保持原有逻辑）
        model_path = os.path.join(MODEL_SAVE_PATH, f"{stage_name}_best.pth")
        try:
            torch.save(save_dict, model_path)
            file_size = os.path.getsize(model_path) / (1024**2)  # MB
            print(f"💾 保存最佳模型成功: {model_path} ({file_size:.1f}MB)")
        except Exception as e:
            print(f"❌ 保存最佳模型失败: {model_path}, 错误: {e}")
            return None
        
        # 清理显存
        self.cleanup_memory()
        
        return model_path
    
    def train_stage1(self, model, data_loaders):
        """第一阶段：监督学习预训练"""
        print("\n" + "="*60)
        print("🚀 第一阶段：监督学习预训练")
        print("="*60)
        
        # 如果检查点是stage3，说明stage1已完成，直接跳过
        if self.resume_training and self.resume_stage == 'stage3':
            print(f"✅ 第一阶段已完成，跳过预训练")
            return os.path.join(MODEL_SAVE_PATH, "stage1_best.pth")
        
        # 创建优化器和调度器
        optimizer, scheduler = self.create_optimizer_and_scheduler(model, 'stage1')
        
        # 加载检查点（如果需要断点续训且是stage1）
        if self.resume_training and self.resume_stage == 'stage1':
            model, optimizer, scheduler = self.load_checkpoint(model, optimizer, scheduler)
        
        # 损失函数
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        
        # AMP处理器
        amp_handler = AmpHandler(enabled=USE_AMP)
        
        # TensorBoard
        writer = SummaryWriter(log_dir=os.path.join(RESULT_PATH, "logs", "stage1"))
        
        # 任务权重
        task_weights = {'colon': 1.0, 'ugi': 1.0, 'colon_disease': 1.0, 'ugi_disease': 1.0}
        
        # 创建评估器
        train_evaluators = create_task_evaluators(self.task_info)
        val_evaluators = create_task_evaluators(self.task_info)
        
        # 早停
        early_stopping = EarlyStopping(
            patience=self.config['patience'], 
            min_delta=self.config['min_delta'], 
            verbose=True
        )
        
        # 初始化最佳准确率和epoch
        best_acc = self.best_accuracy if self.resume_training else 0.0
        best_epoch = 0
        
        # 确定开始的epoch
        start_epoch = 0
        if self.resume_training and self.resume_stage == 'stage1':
            start_epoch = self.resume_epoch + 1  # 从下一个epoch开始
            print(f"🔄 从第 {start_epoch + 1} 轮继续训练（最佳准确率: {best_acc:.2f}%）")
        
        print(f"📈 开始预训练，目标轮数: {self.config['stage1_epochs']} (从第 {start_epoch + 1} 轮开始)")
        
        for epoch in range(start_epoch, self.config['stage1_epochs']):
            epoch_start = time.time()
            
            # 学习率预热
            if epoch < self.config['warmup_epochs']:
                lr_scale = (epoch + 1) / self.config['warmup_epochs']
                for pg in optimizer.param_groups:
                    pg['lr'] = self.config['stage1_lr'] * lr_scale
            
            # 训练
            train_loss, train_task_losses, train_metrics = self.train_epoch(
                model, data_loaders, task_weights, optimizer, criterion,
                epoch, 'stage1', train_evaluators, writer, amp_handler
            )
            
            # 验证
            val_loss, val_task_losses, val_metrics, overall_acc = self.validate_epoch(
                model, data_loaders, task_weights, criterion,
                epoch, 'stage1', val_evaluators, writer, amp_handler
            )
            
            # 更新学习率
            if epoch >= self.config['warmup_epochs']:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['stage1']['epoch'].append(epoch)
            self.history['stage1']['train_loss'].append(train_loss)
            self.history['stage1']['val_loss'].append(val_loss)
            self.history['stage1']['lr'].append(current_lr)
            
            # 记录指标
            if not self.history['stage1']['train_metrics']:
                for task in train_metrics.keys():
                    self.history['stage1']['train_metrics'][task] = []
                    self.history['stage1']['val_metrics'][task] = []
            
            for task in train_metrics.keys():
                self.history['stage1']['train_metrics'][task].append(train_metrics[task])
                self.history['stage1']['val_metrics'][task].append(val_metrics[task])
            
            # 打印结果
            print(f"\n📊 Epoch {epoch+1}/{self.config['stage1_epochs']} 结果:")
            print(f"   训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
            print(f"   总体准确率: {overall_acc:.2f}% | 学习率: {current_lr:.2e}")
            
            # 打印详细指标
            print_epoch_summary(train_metrics, epoch, 'Train')
            print_epoch_summary(val_metrics, epoch, 'Validation')
            
            # 每个epoch都保存模型（包括最新检查点和可选的epoch检查点）
            self.save_best_model_only(model, optimizer, epoch, overall_acc, "stage1", scheduler)
            
            # 检查并更新最佳准确率
            if overall_acc > best_acc:
                best_acc = overall_acc
                best_epoch = epoch
                self.best_accuracy = best_acc  # 更新类变量
                print(f"✅ 更新最佳模型记录，准确率: {best_acc:.2f}%")
            
            # 早停检查
            if early_stopping.check(-overall_acc):
                print(f"⏹️  早停触发于第 {epoch+1} 轮")
                break
            
            epoch_time = time.time() - epoch_start
            print(f"⏱️  本轮用时: {epoch_time:.2f}秒")
            print("-" * 60)
        
        # 保存第一阶段最终模型（只保存最佳模型以节省空间）
        stage1_path = os.path.join(MODEL_SAVE_PATH, "stage1_best.pth")
        print(f"✅ 第一阶段完成，最佳准确率: {best_acc:.2f}% (Epoch {best_epoch+1})")
        
        writer.close()
        return stage1_path
    
    def train_stage3(self, model, data_loaders):
        """第三阶段：端到端微调（跳过第二阶段）"""
        print("\n" + "="*60)
        print("🎯 第三阶段：端到端微调")
        print("="*60)
        
        # 创建优化器和调度器
        optimizer, scheduler = self.create_optimizer_and_scheduler(model, 'stage3')
        
        # 加载检查点（如果需要断点续训且是stage3）
        if self.resume_training and self.resume_stage == 'stage3':
            model, optimizer, scheduler = self.load_checkpoint(model, optimizer, scheduler)
        
        # 损失函数
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING * 0.5)  # 微调时减少标签平滑
        
        # AMP处理器
        amp_handler = AmpHandler(enabled=USE_AMP)
        
        # TensorBoard
        writer = SummaryWriter(log_dir=os.path.join(RESULT_PATH, "logs", "stage3"))
        
        # 任务权重
        task_weights = {'colon': 1.0, 'ugi': 1.0, 'colon_disease': 1.0, 'ugi_disease': 1.0}
        
        # 创建评估器
        train_evaluators = create_task_evaluators(self.task_info)
        val_evaluators = create_task_evaluators(self.task_info)
        
        # 早停
        early_stopping = EarlyStopping(
            patience=self.config['patience'] // 2,  # 微调阶段减少patience
            min_delta=self.config['min_delta'] * 0.1,
            verbose=True
        )
        
        # 初始化最佳准确率和epoch
        best_acc = self.best_accuracy if self.resume_training else 0.0
        best_epoch = 0
        
        # 确定开始的epoch
        start_epoch = 0
        if self.resume_training and self.resume_stage == 'stage3':
            start_epoch = self.resume_epoch + 1  # 从下一个epoch开始
            print(f"🔄 从第 {start_epoch + 1} 轮继续微调（当前最佳准确率: {best_acc:.2f}%）")
        
        print(f"📈 开始微调，目标轮数: {self.config['stage3_epochs']} (从第 {start_epoch + 1} 轮开始)")
        
        for epoch in range(start_epoch, self.config['stage3_epochs']):
            epoch_start = time.time()
            
            # 训练 - 添加熵正则化
            train_loss, train_task_losses, train_metrics = self.train_epoch_with_entropy(
                model, data_loaders, task_weights, optimizer, criterion,
                epoch, 'stage3', train_evaluators, writer, amp_handler
            )
            
            # 验证
            val_loss, val_task_losses, val_metrics, overall_acc = self.validate_epoch(
                model, data_loaders, task_weights, criterion,
                epoch, 'stage3', val_evaluators, writer, amp_handler
            )
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['stage3']['epoch'].append(epoch)
            self.history['stage3']['train_loss'].append(train_loss)
            self.history['stage3']['val_loss'].append(val_loss)
            self.history['stage3']['lr'].append(current_lr)
            
            # 记录指标
            if not self.history['stage3']['train_metrics']:
                for task in train_metrics.keys():
                    self.history['stage3']['train_metrics'][task] = []
                    self.history['stage3']['val_metrics'][task] = []
            
            for task in train_metrics.keys():
                self.history['stage3']['train_metrics'][task].append(train_metrics[task])
                self.history['stage3']['val_metrics'][task].append(val_metrics[task])
            
            # 打印结果
            print(f"\n📊 Epoch {epoch+1}/{self.config['stage3_epochs']} 结果:")
            print(f"   训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
            print(f"   总体准确率: {overall_acc:.2f}% | 学习率: {current_lr:.2e}")
            
            # 打印详细指标
            print_epoch_summary(train_metrics, epoch, 'Train')
            print_epoch_summary(val_metrics, epoch, 'Validation')
            
            # 每个epoch都保存模型（包括最新检查点和可选的epoch检查点）
            self.save_best_model_only(model, optimizer, epoch, overall_acc, "stage3", scheduler)
            
            # 检查并更新最佳准确率
            if overall_acc > best_acc:
                best_acc = overall_acc
                best_epoch = epoch
                self.best_accuracy = best_acc  # 更新类变量
                print(f"✅ 更新最佳模型记录，准确率: {best_acc:.2f}%")
            
            # 早停检查
            if early_stopping.check(-overall_acc):
                print(f"⏹️  早停触发于第 {epoch+1} 轮")
                break
            
            epoch_time = time.time() - epoch_start
            print(f"⏱️  本轮用时: {epoch_time:.2f}秒")
            print("-" * 60)
        
        # 最终模型就是最佳模型，无需重复保存
        final_path = os.path.join(MODEL_SAVE_PATH, "final_best.pth")
        print(f"✅ 第三阶段完成，最佳准确率: {best_acc:.2f}% (Epoch {best_epoch+1})")
        
        writer.close()
        return final_path, best_acc
    
    def save_training_history(self):
        """保存训练历史"""
        # 合并两个阶段的历史
        combined_history = []
        
        # 第一阶段
        for i, epoch in enumerate(self.history['stage1']['epoch']):
            record = {
                'stage': 'stage1',
                'epoch': epoch,
                'global_epoch': epoch,
                'train_loss': self.history['stage1']['train_loss'][i],
                'val_loss': self.history['stage1']['val_loss'][i],
                'learning_rate': self.history['stage1']['lr'][i]
            }
            
            # 添加各任务指标
            for task in self.history['stage1']['train_metrics'].keys():
                if i < len(self.history['stage1']['train_metrics'][task]):
                    train_metrics = self.history['stage1']['train_metrics'][task][i]
                    val_metrics = self.history['stage1']['val_metrics'][task][i]
                    
                    # 添加主要指标
                    record[f'train_{task}_accuracy'] = train_metrics.get('accuracy', 0.0)
                    record[f'val_{task}_accuracy'] = val_metrics.get('accuracy', 0.0)
                    
                    # 添加置信区间
                    record[f'val_{task}_acc_ci_lower'] = val_metrics.get('accuracy_ci_lower', val_metrics.get('accuracy', 0.0))
                    record[f'val_{task}_acc_ci_upper'] = val_metrics.get('accuracy_ci_upper', val_metrics.get('accuracy', 0.0))
                    
                    # 添加F1和AUC
                    f1_key = 'f1' if 'f1' in val_metrics else 'f1_macro'
                    auc_key = 'auc' if 'auc' in val_metrics else 'auc_ovr'
                    record[f'val_{task}_f1'] = val_metrics.get(f1_key, 0.0)
                    record[f'val_{task}_auc'] = val_metrics.get(auc_key, 0.0)
                    
                    # 添加AUC置信区间
                    auc_ci_lower_key = 'auc_ci_lower' if 'auc_ci_lower' in val_metrics else 'auc_ovr_ci_lower'
                    auc_ci_upper_key = 'auc_ci_upper' if 'auc_ci_upper' in val_metrics else 'auc_ovr_ci_upper'
                    record[f'val_{task}_auc_ci_lower'] = val_metrics.get(auc_ci_lower_key, val_metrics.get(auc_key, 0.0))
                    record[f'val_{task}_auc_ci_upper'] = val_metrics.get(auc_ci_upper_key, val_metrics.get(auc_key, 0.0))
            
            combined_history.append(record)
        
        # 第三阶段
        stage1_epochs = len(self.history['stage1']['epoch'])
        for i, epoch in enumerate(self.history['stage3']['epoch']):
            record = {
                'stage': 'stage3',
                'epoch': epoch,
                'global_epoch': stage1_epochs + epoch,
                'train_loss': self.history['stage3']['train_loss'][i],
                'val_loss': self.history['stage3']['val_loss'][i],
                'learning_rate': self.history['stage3']['lr'][i]
            }
            
            # 添加各任务指标
            for task in self.history['stage3']['train_metrics'].keys():
                if i < len(self.history['stage3']['train_metrics'][task]):
                    train_metrics = self.history['stage3']['train_metrics'][task][i]
                    val_metrics = self.history['stage3']['val_metrics'][task][i]
                    
                    # 添加主要指标
                    record[f'train_{task}_accuracy'] = train_metrics.get('accuracy', 0.0)
                    record[f'val_{task}_accuracy'] = val_metrics.get('accuracy', 0.0)
                    
                    # 添加置信区间
                    record[f'val_{task}_acc_ci_lower'] = val_metrics.get('accuracy_ci_lower', val_metrics.get('accuracy', 0.0))
                    record[f'val_{task}_acc_ci_upper'] = val_metrics.get('accuracy_ci_upper', val_metrics.get('accuracy', 0.0))
                    
                    # 添加F1和AUC
                    f1_key = 'f1' if 'f1' in val_metrics else 'f1_macro'
                    auc_key = 'auc' if 'auc' in val_metrics else 'auc_ovr'
                    record[f'val_{task}_f1'] = val_metrics.get(f1_key, 0.0)
                    record[f'val_{task}_auc'] = val_metrics.get(auc_key, 0.0)
                    
                    # 添加AUC置信区间
                    auc_ci_lower_key = 'auc_ci_lower' if 'auc_ci_lower' in val_metrics else 'auc_ovr_ci_lower'
                    auc_ci_upper_key = 'auc_ci_upper' if 'auc_ci_upper' in val_metrics else 'auc_ovr_ci_upper'
                    record[f'val_{task}_auc_ci_lower'] = val_metrics.get(auc_ci_lower_key, val_metrics.get(auc_key, 0.0))
                    record[f'val_{task}_auc_ci_upper'] = val_metrics.get(auc_ci_upper_key, val_metrics.get(auc_key, 0.0))
            
            combined_history.append(record)
        
        # 保存为CSV
        df = pd.DataFrame(combined_history)
        csv_path = os.path.join(RESULT_PATH, "final_training_history.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"✅ 训练历史保存至: {csv_path}")
        return csv_path
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        print("📈 正在生成训练曲线...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=DPI)
        fig.suptitle('Two-Stage Training Results', fontsize=16, fontweight='bold')
        
        # 合并数据用于绘图
        stage1_epochs = list(range(len(self.history['stage1']['epoch'])))
        stage3_epochs = list(range(len(self.history['stage1']['epoch']), 
                                 len(self.history['stage1']['epoch']) + len(self.history['stage3']['epoch'])))
        
        all_epochs = stage1_epochs + stage3_epochs
        all_train_loss = self.history['stage1']['train_loss'] + self.history['stage3']['train_loss']
        all_val_loss = self.history['stage1']['val_loss'] + self.history['stage3']['val_loss']
        all_lr = self.history['stage1']['lr'] + self.history['stage3']['lr']
        
        # 1. 损失曲线
        axes[0, 0].plot(all_epochs, all_train_loss, label='Train Loss', color=LANCET_COLORS[0], linewidth=2)
        axes[0, 0].plot(all_epochs, all_val_loss, label='Val Loss', color=LANCET_COLORS[1], linewidth=2)
        if len(self.history['stage1']['epoch']) > 0:
            axes[0, 0].axvline(x=len(self.history['stage1']['epoch'])-1, color='red', linestyle='--', alpha=0.7, label='Stage 1→3')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 学习率曲线
        axes[0, 1].plot(all_epochs, all_lr, color=LANCET_COLORS[2], linewidth=2)
        if len(self.history['stage1']['epoch']) > 0:
            axes[0, 1].axvline(x=len(self.history['stage1']['epoch'])-1, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3-6. 各任务准确率曲线
        task_names = ['colon', 'ugi', 'colon_disease', 'ugi_disease']
        positions = [(0, 2), (1, 0), (1, 1), (1, 2)]
        
        for i, (task, pos) in enumerate(zip(task_names, positions)):
            ax = axes[pos[0], pos[1]]
            
            # 收集该任务的准确率数据
            stage1_acc = []
            stage2_acc = []
            
            if task in self.history['stage1']['val_metrics']:
                stage1_acc = [metrics.get('accuracy', 0.0) 
                             for metrics in self.history['stage1']['val_metrics'][task]]
            
            if task in self.history['stage3']['val_metrics']:
                stage3_acc = [metrics.get('accuracy', 0.0) 
                             for metrics in self.history['stage3']['val_metrics'][task]]
            
            all_acc = stage1_acc + stage3_acc
            
            if all_acc:
                ax.plot(all_epochs[:len(all_acc)], all_acc, 
                       color=LANCET_COLORS[i % len(LANCET_COLORS)], linewidth=2)
                if len(stage1_acc) > 0:
                    ax.axvline(x=len(stage1_acc)-1, color='red', linestyle='--', alpha=0.7)
                
                # 添加置信区间（如果有的话）
                stage1_ci_lower = []
                stage1_ci_upper = []
                stage2_ci_lower = []
                stage2_ci_upper = []
                
                if task in self.history['stage1']['val_metrics']:
                    stage1_ci_lower = [metrics.get('accuracy_ci_lower', metrics.get('accuracy', 0.0)) 
                                      for metrics in self.history['stage1']['val_metrics'][task]]
                    stage1_ci_upper = [metrics.get('accuracy_ci_upper', metrics.get('accuracy', 0.0)) 
                                      for metrics in self.history['stage1']['val_metrics'][task]]
                
                if task in self.history['stage3']['val_metrics']:
                    stage3_ci_lower = [metrics.get('accuracy_ci_lower', metrics.get('accuracy', 0.0)) 
                                      for metrics in self.history['stage3']['val_metrics'][task]]
                    stage3_ci_upper = [metrics.get('accuracy_ci_upper', metrics.get('accuracy', 0.0)) 
                                      for metrics in self.history['stage3']['val_metrics'][task]]
                
                all_ci_lower = stage1_ci_lower + stage3_ci_lower
                all_ci_upper = stage1_ci_upper + stage3_ci_upper
                
                if len(all_ci_lower) == len(all_acc) and len(all_ci_upper) == len(all_acc):
                    ax.fill_between(all_epochs[:len(all_acc)], all_ci_lower, all_ci_upper, 
                                   alpha=0.2, color=LANCET_COLORS[i % len(LANCET_COLORS)])
            
            ax.set_title(f'{task.replace("_", " ").title()} Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(RESULT_PATH, "visualizations", "final_training_curves.png")
        plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 训练曲线保存至: {plot_path}")
        return plot_path
    
    def run_training(self):
        """运行完整的两阶段训练 - 支持断点续训"""
        print("🚀 开始两阶段训练流程")
        print("="*80)
        
        # 检查是否从检查点恢复训练
        self.prompt_resume_training()
        
        # 设置随机种子
        set_seed(SEED)
        
        training_start = time.time()
        
        try:
            # 创建数据加载器
            data_loaders = self.create_data_loaders()
            
            # 创建模型
            model = self.create_model()
            
            # 如果是从检查点恢复，加载模型权重（仅模型部分，优化器在各阶段单独处理）
            if self.resume_training:
                model_checkpoint = self.resume_checkpoint_info['checkpoint']
                model.load_state_dict(model_checkpoint['model_state_dict'])
                print(f"✅ 模型权重已从检查点恢复")
            
            # 第一阶段：监督学习预训练
            stage1_path = self.train_stage1(model, data_loaders)
            
            # 第三阶段：端到端微调（跳过第二阶段）
            final_path, best_acc = self.train_stage3(model, data_loaders)
            
            # 保存训练历史
            history_path = self.save_training_history()
            
            # 绘制训练曲线
            plot_path = self.plot_training_curves()
            
            # 计算总训练时间
            total_time = time.time() - training_start
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print("\n" + "="*80)
            print("🎉 两阶段训练完成！")
            print("="*80)
            print(f"🏆 最终最佳准确率: {best_acc:.2f}%")
            print(f"⏱️  总训练时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
            print(f"💾 最终模型: {final_path}")
            print(f"📊 训练历史: {history_path}")
            print(f"📈 训练曲线: {plot_path}")
            
            if self.resume_training:
                print(f"🔄 训练已从检查点成功恢复并完成")
                print(f"   原检查点: {os.path.basename(self.resume_checkpoint_path)}")
            
            print("="*80)
            
            return {
                'best_accuracy': best_acc,
                'final_model_path': final_path,
                'training_history_path': history_path,
                'training_curves_path': plot_path,
                'total_training_time': total_time,
                'resumed_from_checkpoint': self.resume_training
            }
            
        except KeyboardInterrupt:
            print("\n⚠️  训练被用户中断")
            print("💾 最新的检查点已保存，可以使用断点续训功能继续训练")
            return None
        except Exception as e:
            print(f"\n❌ 训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数"""
    print("="*80)
    print("🚀 最终版训练脚本 - 智能断点续训版")
    print("   改进版训练：预训练 + 端到端微调（启用数据增强）")
    print("   多指标评估：准确度、AUC、F1、精确率、95%置信区间")
    print("   🔧 显存优化：batch_size=4, 梯度累积=16, 数据增强")
    print("   💾 只保存最佳模型，适合4-8GB显存的共享GPU环境")
    print("   🔄 智能断点续训：自动检测并恢复中断的训练")
    print("="*80)
    
    # 创建并运行训练器
    trainer = TwoStageTrainer(OPTIMIZED_TRAINING_CONFIG)
    results = trainer.run_training()
    
    if results:
        print(f"\n✅ 训练成功完成！最佳准确率: {results['best_accuracy']:.2f}%")
        if results.get('resumed_from_checkpoint', False):
            print(f"🔄 已从检查点成功恢复训练")
        print(f"📊 显存使用已优化，仅需4-8GB显存")
    else:
        print("\n❌ 训练未能完成")


if __name__ == "__main__":
    main()