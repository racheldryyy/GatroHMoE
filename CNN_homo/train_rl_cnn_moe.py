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

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, SEED, 
    COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, UGI_DISEASE_PATH, RESULT_PATH,
    LANCET_COLORS, LANCET_PASTEL_COLORS, FIG_SIZE, DPI, CNN_EXPERT_MODELS,
    LABEL_SMOOTHING, LR_WARMUP_EPOCHS, LR_MIN_FACTOR, MIXUP_ALPHA, CUTMIX_ALPHA,
    AUGMENTATION_PROBABILITY, TASK_WEIGHT_STRATEGY, TASK_WEIGHT_ALPHA,
    LOAD_BALANCE_WEIGHT, LOAD_BALANCE_DECAY, PATIENCE, MIN_DELTA, GPU_IDS, USE_AMP,
    RL_LEARNING_RATE, RL_GAMMA, RL_BATCH_SIZE, RL_EPOCHS, PRETRAIN_EPOCHS, 
    FINETUNE_EPOCHS, RL_ENTROPY_COEF
)
from utils import (
    set_seed, ProgressBar, save_training_curve, save_model, load_model,
    EarlyStopping
)
from data_loader import create_multi_task_loaders, mixup, cutmix
from models.rl_cnn_moe_model import RLCNNMixtureOfExperts
from amp_utils import AmpHandler

class PolicyBuffer:
    """经验回放缓冲区，用于存储强化学习训练数据"""
    
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.states = []
        self.routing_logits = []
        self.actions = []
        self.rewards = []
        self.position = 0
        
    def push(self, state, routing_logits, action, reward):
        """将经验添加到缓冲区"""
        if len(self.states) < self.capacity:
            self.states.append(None)
            self.routing_logits.append(None)
            self.actions.append(None)
            self.rewards.append(None)
        
        self.states[self.position] = state
        self.routing_logits[self.position] = routing_logits
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """随机采样一批经验"""
        if len(self.states) < batch_size:
            batch_size = len(self.states)
            
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        
        return (
            torch.stack([self.states[i] for i in indices]),
            torch.stack([self.routing_logits[i] for i in indices]),
            torch.stack([self.actions[i] for i in indices]),
            torch.tensor([self.rewards[i] for i in indices], device=DEVICE)
        )
    
    def __len__(self):
        return len(self.states)

def compute_confidence_weighted_reward(outputs, labels):
    """计算基于置信度的加权奖励"""
    probs = F.softmax(outputs, dim=1)
    predicted = outputs.argmax(dim=1)
    confidence = probs.gather(1, predicted.unsqueeze(1)).squeeze()
    
    # 判断预测是否正确
    correct = (predicted == labels).float()
    
    # 计算加权奖励
    reward = correct * (1.0 + confidence) - (1.0 - correct) * confidence
    
    return reward

def update_policy(model, policy_optimizer, memory, batch_size, amp_handler=None):
    """使用策略梯度更新路由策略"""
    if len(memory) < batch_size:
        return 0.0
        
    # 从经验回放缓冲区采样
    states, routing_logits, actions, rewards = memory.sample(batch_size)
    
    # 计算对数概率
    with amp_handler.autocast() if amp_handler else torch.amp.autocast('cuda', enabled=False):
        log_probs = F.log_softmax(routing_logits, dim=1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # 计算策略损失
        policy_loss = -(selected_log_probs * rewards).mean()
        
        # 添加熵正则化项，鼓励探索
        entropy = -(F.softmax(routing_logits, dim=1) * log_probs).sum(dim=1).mean()
        policy_loss -= RL_ENTROPY_COEF * entropy
    
    # 更新策略网络
    policy_optimizer.zero_grad()
    
    if amp_handler:
        amp_handler.scale_loss(policy_loss).backward()
        if hasattr(amp_handler.scaler, "_enabled") and amp_handler.scaler._enabled:
            amp_handler.scaler.unscale_(policy_optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 放宽梯度裁剪
        amp_handler.step(policy_optimizer)
        amp_handler.update()
    else:
        policy_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        policy_optimizer.step()
    
    return policy_loss.item()

def freeze_experts(model):
    """冻结专家模型参数，只训练路由策略"""
    if isinstance(model, nn.DataParallel):
        module = model.module
    else:
        module = model
        
    for expert in module.experts:
        for param in expert.parameters():
            param.requires_grad = False
            
    for proj in module.projections:
        for param in proj.parameters():
            param.requires_grad = False
            
    # 确保门控网络参数可训练
    for task, gate in module.gates.items():
        for param in gate.parameters():
            param.requires_grad = True

def unfreeze_experts(model):
    """解冻专家模型参数，进行整体微调"""
    if isinstance(model, nn.DataParallel):
        module = model.module
    else:
        module = model
        
    for expert in module.experts:
        for param in expert.parameters():
            param.requires_grad = True
            
    for proj in module.projections:
        for param in proj.parameters():
            param.requires_grad = True

def train_supervised(model, data_loaders, task_weights, optimizer, criterion, 
                     epoch, writer, balance_loss_weight=0.1, amp_handler=None):
    """监督学习训练一个epoch"""
    model.train()
    total_loss = 0.0
    task_losses = {task: 0.0 for task in data_loaders.keys()}
    total_acc = {task: 0.0 for task in data_loaders.keys()}
    samples_count = {task: 0 for task in data_loaders.keys()}
    
    max_len = max([len(data_loaders[task]['train_loader']) for task in data_loaders.keys()])
    pbar = ProgressBar(max_len, desc=f"Epoch {epoch+1} [Supervised Train]")
    
    iterators = {task: iter(data_loaders[task]['train_loader']) for task in data_loaders.keys()}
    
    for i in range(max_len):
        batch_loss = 0.0
        batch_task_losses = {}
        
        for task, task_weight in task_weights.items():
            try:
                inputs, labels = next(iterators[task])
            except StopIteration:
                iterators[task] = iter(data_loaders[task]['train_loader'])
                inputs, labels = next(iterators[task])
            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # 应用数据增强
            r = np.random.rand()
            
            with amp_handler.autocast() if amp_handler else torch.amp.autocast('cuda', enabled=False):
                if r < AUGMENTATION_PROBABILITY / 2:
                    mixed_inputs, labels_a, labels_b, lam = mixup(inputs, labels, alpha=MIXUP_ALPHA)
                    inputs = mixed_inputs
                    
                    outputs, routing_weights = model(inputs, task=task)
                    
                    loss_a = criterion(outputs, labels_a)
                    loss_b = criterion(outputs, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                    
                elif r < AUGMENTATION_PROBABILITY:
                    mixed_inputs, labels_a, labels_b, lam = cutmix(inputs, labels, alpha=CUTMIX_ALPHA)
                    inputs = mixed_inputs
                    
                    outputs, routing_weights = model(inputs, task=task)
                    
                    loss_a = criterion(outputs, labels_a)
                    loss_b = criterion(outputs, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                    
                else:
                    outputs, routing_weights = model(inputs, task=task)
                    loss = criterion(outputs, labels)
                
                # 计算负载均衡损失
                if isinstance(model, nn.DataParallel):
                    load_balancing_loss = model.module.calculate_load_balancing_loss(routing_weights)
                else:
                    load_balancing_loss = model.calculate_load_balancing_loss(routing_weights)
                    
                task_specific_loss = loss + balance_loss_weight * load_balancing_loss
                weighted_task_loss = task_weight * task_specific_loss
            
            batch_task_losses[task] = task_specific_loss.item()
            task_losses[task] += batch_task_losses[task]
            batch_loss += weighted_task_loss
            
            # 计算准确率
            if r >= AUGMENTATION_PROBABILITY:
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                batch_total = labels.size(0)
                
                total_acc[task] += batch_correct
                samples_count[task] += batch_total
            else:
                _, predicted = torch.max(outputs.data, 1)
                if r < AUGMENTATION_PROBABILITY / 2:
                    batch_correct = (predicted == labels_a).sum().item() * lam + (predicted == labels_b).sum().item() * (1 - lam)
                else:
                    batch_correct = (predicted == labels_a).sum().item() * lam + (predicted == labels_b).sum().item() * (1 - lam)
                batch_total = labels.size(0)
                
                total_acc[task] += batch_correct
                samples_count[task] += batch_total
        
        # 反向传播和优化
        optimizer.zero_grad()
        
        if amp_handler:
            amp_handler.scale_loss(batch_loss).backward()
            if hasattr(amp_handler.scaler, "_enabled") and amp_handler.scaler._enabled:
                amp_handler.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            amp_handler.step(optimizer)
            amp_handler.update()
        else:
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        
        total_loss += batch_loss.item()
        
        if i % 10 == 0:
            task_desc = " | ".join([f"{task}: {batch_task_losses[task]:.3f}" for task in batch_task_losses.keys()])
            pbar.set_description(f"Epoch {epoch+1} [Supervised] Loss: {batch_loss.item():.3f} | {task_desc}")
        
        pbar.update()
    
    pbar.close()
    
    # 计算平均损失和准确率
    avg_loss = total_loss / max_len
    avg_task_losses = {task: task_losses[task] / max_len for task in task_losses.keys()}
    avg_acc = {task: total_acc[task] / samples_count[task] * 100.0 if samples_count[task] > 0 else 0.0 
               for task in data_loaders.keys()}
    
    # 记录训练指标
    writer.add_scalar('Loss/supervised/total', avg_loss, epoch)
    for task in avg_task_losses.keys():
        writer.add_scalar(f'Loss/supervised/{task}', avg_task_losses[task], epoch)
        writer.add_scalar(f'Accuracy/supervised/{task}', avg_acc[task], epoch)
    
    print(f"Epoch {epoch+1} 监督学习训练结果:")
    print(f"总损失: {avg_loss:.4f}")
    for task in avg_task_losses.keys():
        print(f"{task} 损失: {avg_task_losses[task]:.4f} | 准确率: {avg_acc[task]:.2f}%")
    print("-" * 60)
    
    return avg_loss, avg_acc, avg_task_losses

def train_rl_epoch(model, data_loaders, task_weights, rl_optimizer, criterion, 
                  memory, epoch, writer, batch_size=RL_BATCH_SIZE, amp_handler=None):
    """强化学习训练一个epoch"""
    model.train()
    total_loss = 0.0
    task_losses = {task: 0.0 for task in data_loaders.keys()}
    rl_losses = []
    total_acc = {task: 0.0 for task in data_loaders.keys()}
    samples_count = {task: 0 for task in data_loaders.keys()}
    total_rewards = {task: 0.0 for task in data_loaders.keys()}
    
    max_len = max([len(data_loaders[task]['train_loader']) for task in data_loaders.keys()])
    pbar = ProgressBar(max_len, desc=f"Epoch {epoch+1} [RL Train]")
    
    iterators = {task: iter(data_loaders[task]['train_loader']) for task in data_loaders.keys()}
    
    for i in range(max_len):
        batch_rl_loss = 0.0
        batch_task_losses = {}
        batch_rewards = {}
        
        for task, task_weight in task_weights.items():
            try:
                inputs, labels = next(iterators[task])
            except StopIteration:
                iterators[task] = iter(data_loaders[task]['train_loader'])
                inputs, labels = next(iterators[task])
            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            with amp_handler.autocast() if amp_handler else torch.amp.autocast('cuda', enabled=False):
                outputs, routing_weights, routing_features = model(inputs, task=task, return_features=True)
                
                ce_loss = criterion(outputs, labels)
                batch_task_losses[task] = ce_loss.item()
                task_losses[task] += batch_task_losses[task]
                
                rewards = compute_confidence_weighted_reward(outputs, labels)
                batch_rewards[task] = rewards.mean().item()
                total_rewards[task] += batch_rewards[task]
            
            # 确定选择的专家
            _, top_k_indices = torch.topk(routing_weights, model.module.top_k if isinstance(model, nn.DataParallel) else model.top_k, dim=1)
            
            # 存储经验到缓冲区
            for b in range(inputs.size(0)):
                action = top_k_indices[b, 0]
                memory.push(
                    routing_features[b].detach(),  
                    routing_weights[b].detach(),   
                    action,                        
                    rewards[b].item()              
                )
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)
            
            total_acc[task] += batch_correct
            samples_count[task] += batch_total
        
        # 更新策略网络
        if i % 5 == 0 and len(memory) >= batch_size:
            rl_loss = update_policy(model, rl_optimizer, memory, batch_size, amp_handler=amp_handler)
            rl_losses.append(rl_loss)
            batch_rl_loss = rl_loss
        
        if i % 10 == 0:
            reward_desc = " | ".join([f"{task} R: {batch_rewards[task]:.2f}" for task in batch_rewards.keys()])
            pbar.set_description(f"Epoch {epoch+1} [RL] Loss: {batch_rl_loss:.3f} | {reward_desc}")
        
        pbar.update()
    
    pbar.close()
    
    avg_task_losses = {task: task_losses[task] / max_len for task in task_losses.keys()}
    avg_rewards = {task: total_rewards[task] / max_len for task in task_weights.keys()}
    avg_acc = {task: total_acc[task] / samples_count[task] * 100.0 if samples_count[task] > 0 else 0.0 
               for task in data_loaders.keys()}
    avg_rl_loss = np.mean(rl_losses) if rl_losses else 0.0
    
    # 记录训练指标
    writer.add_scalar('Loss/rl/policy_loss', avg_rl_loss, epoch)
    for task in task_weights.keys():
        writer.add_scalar(f'Reward/rl/{task}', avg_rewards[task], epoch)
        writer.add_scalar(f'Accuracy/rl/{task}', avg_acc[task], epoch)
    
    print(f"Epoch {epoch+1} 强化学习训练结果:")
    print(f"策略损失: {avg_rl_loss:.4f}")
    for task in task_weights.keys():
        print(f"{task} 奖励: {avg_rewards[task]:.4f} | 准确率: {avg_acc[task]:.2f}%")
    print("-" * 60)
    
    return avg_rl_loss, avg_acc, avg_rewards

def validate(model, data_loaders, task_weights, criterion, epoch, writer, amp_handler=None):
    """验证模型性能"""
    model.eval()
    total_loss = 0.0
    task_losses = {task: 0.0 for task in data_loaders.keys()}
    total_acc = {task: 0.0 for task in data_loaders.keys()}
    samples_count = {task: 0 for task in data_loaders.keys()}
    total_rewards = {task: 0.0 for task in data_loaders.keys()}
    
    max_len = max([len(data_loaders[task]['test_loader']) for task in data_loaders.keys()])
    pbar = ProgressBar(max_len, desc=f"Epoch {epoch+1} [Val]")
    
    with torch.no_grad():
        iterators = {task: iter(data_loaders[task]['test_loader']) for task in data_loaders.keys()}
        
        for i in range(max_len):
            batch_loss = 0.0
            batch_task_losses = {}
            batch_rewards = {}
            
            for task, task_weight in task_weights.items():
                try:
                    inputs, labels = next(iterators[task])
                except StopIteration:
                    iterators[task] = iter(data_loaders[task]['test_loader'])
                    inputs, labels = next(iterators[task])
                
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                with amp_handler.autocast() if amp_handler else torch.amp.autocast('cuda', enabled=False):
                    outputs, routing_weights = model(inputs, task=task)
                    loss = criterion(outputs, labels)
                    
                    rewards = compute_confidence_weighted_reward(outputs, labels)
                    batch_rewards[task] = rewards.mean().item()
                    total_rewards[task] += batch_rewards[task]
                
                batch_task_losses[task] = loss.item()
                task_losses[task] += batch_task_losses[task]
                batch_loss += task_weight * loss
                
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                batch_total = labels.size(0)
                
                total_acc[task] += batch_correct
                samples_count[task] += batch_total
            
            total_loss += batch_loss.item()
            
            if i % 10 == 0:
                acc_desc = " | ".join([f"{task} Acc: {total_acc[task]/max(1, samples_count[task])*100:.1f}%" 
                                     for task in task_weights.keys()])
                pbar.set_description(f"Epoch {epoch+1} [Val] Loss: {batch_loss.item():.3f} | {acc_desc}")
            
            pbar.update()
    
    pbar.close()
    
    avg_loss = total_loss / max_len
    avg_task_losses = {task: task_losses[task] / max_len for task in task_losses.keys()}
    avg_rewards = {task: total_rewards[task] / max_len for task in task_weights.keys()}
    avg_acc = {task: total_acc[task] / samples_count[task] * 100.0 if samples_count[task] > 0 else 0.0 
               for task in data_loaders.keys()}
    
    # 记录验证指标
    writer.add_scalar('Loss/val/total', avg_loss, epoch)
    for task in avg_task_losses.keys():
        writer.add_scalar(f'Loss/val/{task}', avg_task_losses[task], epoch)
        writer.add_scalar(f'Reward/val/{task}', avg_rewards[task], epoch)
        writer.add_scalar(f'Accuracy/val/{task}', avg_acc[task], epoch)
    
    print(f"Epoch {epoch+1} 验证结果:")
    print(f"总损失: {avg_loss:.4f}")
    for task in avg_task_losses.keys():
        print(f"{task} 损失: {avg_task_losses[task]:.4f} | 奖励: {avg_rewards[task]:.4f} | 准确率: {avg_acc[task]:.2f}%")
    print("-" * 60)
    
    # 计算整体准确率
    overall_acc = sum(avg_acc.values()) / len(avg_acc)
    
    return avg_loss, avg_acc, avg_task_losses, overall_acc, avg_rewards

def train_rl_cnn_moe(model_architectures=None, resume=False, use_amp=USE_AMP):
    """使用混合监督学习和强化学习训练CNN混合专家模型"""
    set_seed(SEED)
    
    amp_handler = AmpHandler(enabled=use_amp)
    
    if model_architectures is None:
        expert_models = [model['class'] for model in CNN_EXPERT_MODELS]
        model_architectures = [model['class'] for model in CNN_EXPERT_MODELS]
    
    log_dir = os.path.join(RESULT_PATH, "logs", "rl_cnn_moe")
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    start_time = time.time()
    
    print("正在加载数据...")
    data_loaders = create_multi_task_loaders(
        COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, UGI_DISEASE_PATH
    )
    
    num_colon_classes = len(data_loaders['colon']['classes'])
    num_ugi_classes = len(data_loaders['ugi']['classes'])
    num_colon_disease_classes = len(data_loaders['colon_disease']['classes'])
    num_ugi_disease_classes = len(data_loaders['ugi_disease']['classes'])
    
    print(f"数据加载完成! 用时: {time.time() - start_time:.2f}秒")
    print(f"肠镜部位类别数: {num_colon_classes}")
    print(f"胃镜部位类别数: {num_ugi_classes}")
    print(f"肠镜疾病类别数: {num_colon_disease_classes}")
    print(f"胃镜疾病类别数: {num_ugi_disease_classes}")
    print(f"使用架构: {model_architectures}")
    
    # 初始化固定任务权重
    task_weights = {
        'colon': 1.0,
        'ugi': 1.0,
        'colon_disease': 1.0,
        'ugi_disease': 1.0
    }
    
    print(f"使用固定任务权重: {task_weights}")
    
    # 创建模型
    print(f"创建带强化学习的CNN混合专家模型 ({len(model_architectures)} 专家)...")
    model = RLCNNMixtureOfExperts(
        model_names=model_architectures,
        num_colon_classes=num_colon_classes,
        num_ugi_classes=num_ugi_classes,
        num_colon_disease_classes=num_colon_disease_classes,
        num_ugi_disease_classes=num_ugi_disease_classes
    ).to(DEVICE)
    
    if torch.cuda.device_count() > 1:
        print(f"✅ 使用 {torch.cuda.device_count()} 张 GPU 进行 DataParallel")
        model = nn.DataParallel(model, device_ids=GPU_IDS)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rl_optimizer = optim.Adam(model.parameters(), lr=RL_LEARNING_RATE)
    
    # 创建学习率调度器
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,
        T_mult=2,
        eta_min=LEARNING_RATE * LR_MIN_FACTOR
    )
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # 早停策略
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA, verbose=True)
    
    # 创建经验回放缓冲区
    memory = PolicyBuffer(capacity=10000)
    
    start_epoch = 0
    best_acc = 0.0
    model_name = "rl_cnn_moe"
    
    if resume:
        start_epoch, best_acc = load_model(model, optimizer, model_name)
        print(f"从第 {start_epoch} 轮恢复训练，最佳精度: {best_acc:.4f}")
    
    # 记录训练历史
    history = {
        'supervised': {
            'epoch': [], 'loss': [],
            'acc': {task: [] for task in task_weights.keys()},
            'task_losses': {task: [] for task in task_weights.keys()},
        },
        'rl': {
            'epoch': [], 'loss': [],
            'acc': {task: [] for task in task_weights.keys()},
            'rewards': {task: [] for task in task_weights.keys()},
        },
        'val': {
            'epoch': [], 'loss': [],
            'acc': {task: [] for task in task_weights.keys()},
            'task_losses': {task: [] for task in task_weights.keys()},
            'rewards': {task: [] for task in task_weights.keys()},
        },
        'lr': [],
        'task_weights': {task: [] for task in task_weights.keys()},
    }
    
    # 训练循环
    print("\n" + "="*20 + " 开始混合强化学习训练 " + "="*20)
    training_start = time.time()
    
    try:
        # 第一阶段：监督学习预训练
        print("\n" + "="*20 + " 阶段1：监督学习预训练 " + "="*20)
        
        pretrain_pbar = tqdm(range(start_epoch, start_epoch + PRETRAIN_EPOCHS), 
                           desc="预训练阶段", 
                           unit="epoch",
                           ncols=120,
                           colour='green')
        
        for epoch in pretrain_pbar:
            epoch_start = time.time()
            
            # 学习率预热
            if epoch < LR_WARMUP_EPOCHS:
                lr_scale = min(1.0, (epoch + 1) / LR_WARMUP_EPOCHS)
                for pg in optimizer.param_groups:
                    pg['lr'] = LEARNING_RATE * lr_scale
            
            # 衰减负载均衡权重
            current_balance_weight = LOAD_BALANCE_WEIGHT * (LOAD_BALANCE_DECAY ** epoch)
            
            # 监督学习训练
            train_loss, train_acc, train_task_losses = train_supervised(
                model, data_loaders, task_weights, optimizer, criterion, 
                epoch, writer, balance_loss_weight=current_balance_weight,
                amp_handler=amp_handler
            )
            
            # 验证
            val_loss, val_acc, val_task_losses, overall_acc, val_rewards = validate(
                model, data_loaders, task_weights, criterion, epoch, writer,
                amp_handler=amp_handler
            )
            
            # 更新学习率
            if epoch >= LR_WARMUP_EPOCHS:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录指标
            history['supervised']['epoch'].append(epoch)
            history['supervised']['loss'].append(train_loss)
            history['val']['epoch'].append(epoch)
            history['val']['loss'].append(val_loss)
            history['lr'].append(current_lr)
            
            for task in task_weights.keys():
                history['supervised']['acc'][task].append(train_acc[task])
                history['supervised']['task_losses'][task].append(train_task_losses[task])
                history['val']['acc'][task].append(val_acc[task])
                history['val']['task_losses'][task].append(val_task_losses[task])
                history['val']['rewards'][task].append(val_rewards[task])
                history['task_weights'][task].append(task_weights[task])
            
            # 保存最佳模型
            if overall_acc > best_acc:
                best_acc = overall_acc
                save_model(model, optimizer, epoch, best_acc, model_name)
                print(f"保存最佳模型，精度: {best_acc:.4f}")
            
            # 计算用时
            epoch_time = time.time() - epoch_start
            remaining_epochs = (PRETRAIN_EPOCHS + RL_EPOCHS + FINETUNE_EPOCHS) - (epoch - start_epoch) - 1
            remaining_time = epoch_time * remaining_epochs
            
            print(f"Epoch {epoch+1} 用时: {epoch_time:.2f}秒 | 估计剩余时间: {remaining_time/60:.2f}分钟")
            print("=" * 80)
        
        # 保存预训练模型
        save_model(model, optimizer, epoch, best_acc, f"{model_name}_pretrained")
        print(f"预训练阶段完成，模型已保存")
        
        # 第二阶段：强化学习路由优化
        print("\n" + "="*20 + " 阶段2：强化学习路由优化 " + "="*20)
        
        # 冻结专家网络参数
        freeze_experts(model)
        print("专家网络已冻结，仅训练路由策略")
        
        # 使用强化学习训练
        for epoch in range(start_epoch + PRETRAIN_EPOCHS, start_epoch + PRETRAIN_EPOCHS + RL_EPOCHS):
            epoch_start = time.time()
            
            # 强化学习训练
            rl_loss, rl_acc, rl_rewards = train_rl_epoch(
                model, data_loaders, task_weights, rl_optimizer, criterion,
                memory, epoch, writer, amp_handler=amp_handler
            )
            
            # 验证
            val_loss, val_acc, val_task_losses, overall_acc, val_rewards = validate(
                model, data_loaders, task_weights, criterion, epoch, writer,
                amp_handler=amp_handler
            )
            
            # 记录指标
            history['rl']['epoch'].append(epoch)
            history['rl']['loss'].append(rl_loss)
            history['val']['epoch'].append(epoch)
            history['val']['loss'].append(val_loss)
            
            for task in task_weights.keys():
                history['rl']['acc'][task].append(rl_acc[task])
                history['rl']['rewards'][task].append(rl_rewards[task])
                history['val']['acc'][task].append(val_acc[task])
                history['val']['task_losses'][task].append(val_task_losses[task])
                history['val']['rewards'][task].append(val_rewards[task])
            
            # 保存最佳模型
            if overall_acc > best_acc:
                best_acc = overall_acc
                save_model(model, optimizer, epoch, best_acc, model_name)
                print(f"保存最佳模型，精度: {best_acc:.4f}")
            
            # 计算用时
            epoch_time = time.time() - epoch_start
            remaining_epochs = (PRETRAIN_EPOCHS + RL_EPOCHS + FINETUNE_EPOCHS) - (epoch - start_epoch) - 1
            remaining_time = epoch_time * remaining_epochs
            
            print(f"Epoch {epoch+1} 用时: {epoch_time:.2f}秒 | 估计剩余时间: {remaining_time/60:.2f}分钟")
            print("=" * 80)
        
        # 保存强化学习模型
        save_model(model, optimizer, epoch, best_acc, f"{model_name}_rl")
        print(f"强化学习阶段完成，模型已保存")
        
        # 第三阶段：端到端微调
        print("\n" + "="*20 + " 阶段3：端到端微调 " + "="*20)
        
        # 解冻专家网络参数
        unfreeze_experts(model)
        print("专家网络已解冻，进行端到端微调")
        
        # 重新设置优化器，使用较小学习率
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=WEIGHT_DECAY)
        
        # 端到端微调
        for epoch in range(start_epoch + PRETRAIN_EPOCHS + RL_EPOCHS, 
                          start_epoch + PRETRAIN_EPOCHS + RL_EPOCHS + FINETUNE_EPOCHS):
            epoch_start = time.time()
            
            # 监督学习 + 强化学习混合训练
            train_loss, train_acc, train_task_losses = train_supervised(
                model, data_loaders, task_weights, optimizer, criterion, 
                epoch, writer, balance_loss_weight=LOAD_BALANCE_WEIGHT * 0.1,
                amp_handler=amp_handler
            )
            
            # 再使用强化学习更新路由策略
            rl_loss, rl_acc, rl_rewards = train_rl_epoch(
                model, data_loaders, task_weights, rl_optimizer, criterion,
                memory, epoch, writer, amp_handler=amp_handler
            )
            
            # 验证
            val_loss, val_acc, val_task_losses, overall_acc, val_rewards = validate(
                model, data_loaders, task_weights, criterion, epoch, writer,
                amp_handler=amp_handler
            )
            
            # 记录指标
            history['supervised']['epoch'].append(epoch)
            history['supervised']['loss'].append(train_loss)
            history['rl']['epoch'].append(epoch)
            history['rl']['loss'].append(rl_loss)
            history['val']['epoch'].append(epoch)
            history['val']['loss'].append(val_loss)
            
            for task in task_weights.keys():
                history['supervised']['acc'][task].append(train_acc[task])
                history['supervised']['task_losses'][task].append(train_task_losses[task])
                history['rl']['acc'][task].append(rl_acc[task])
                history['rl']['rewards'][task].append(rl_rewards[task])
                history['val']['acc'][task].append(val_acc[task])
                history['val']['task_losses'][task].append(val_task_losses[task])
                history['val']['rewards'][task].append(val_rewards[task])
            
            # 保存最佳模型
            if overall_acc > best_acc:
                best_acc = overall_acc
                save_model(model, optimizer, epoch, best_acc, model_name)
                print(f"保存最佳模型，精度: {best_acc:.4f}")
            
            # 检查早停
            if early_stopping.check(-overall_acc):
                print(f"早停触发于第 {epoch+1} 轮")
                break
            
            # 计算用时
            epoch_time = time.time() - epoch_start
            remaining_epochs = (PRETRAIN_EPOCHS + RL_EPOCHS + FINETUNE_EPOCHS) - (epoch - start_epoch) - 1
            remaining_time = epoch_time * remaining_epochs
            
            print(f"Epoch {epoch+1} 用时: {epoch_time:.2f}秒 | 估计剩余时间: {remaining_time/60:.2f}分钟")
            print("=" * 80)
    
    except KeyboardInterrupt:
        print("\n训练被中断!")
    
    # 计算总训练时间
    total_time = time.time() - training_start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"训练完成! 总用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    
    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': list(range(start_epoch, epoch + 1)),
    })
    
    # 保存CSV
    csv_path = os.path.join(RESULT_PATH, "logs", f"{model_name}_history.csv")
    history_df.to_csv(csv_path, index=False)
    print(f"训练历史已保存至: {csv_path}")
    
    return model, data_loaders, history

if __name__ == "__main__":
    train_rl_cnn_moe()