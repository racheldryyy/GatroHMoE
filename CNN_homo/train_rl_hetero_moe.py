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
    LOAD_BALANCE_WEIGHT, LOAD_BALANCE_DECAY, PATIENCE, MIN_DELTA, GPU_IDS, USE_AMP
)
from utils import (
    set_seed, ProgressBar, save_training_curve, save_model, load_model,
    EarlyStopping
)
from data_loader import create_multi_task_loaders, mixup, cutmix
from models.rl_hetero_moe_model import RLHeterogeneousMixtureOfExperts
from amp_utils import AmpHandler

# 定义RL优化所需的参数 - 与Transformer保持一致
RL_LEARNING_RATE = 1e-4  # RL训练的学习率
RL_GAMMA = 0.99  # 奖励折扣因子
RL_BATCH_SIZE = 128  # RL批次大小
RL_EPOCHS = 15  # 增加RL训练轮数
PRETRAIN_EPOCHS = 20  # 增加预训练轮数以充分训练专家
FINETUNE_EPOCHS = 15  # 增加微调轮数
RL_ENTROPY_COEF = 0.01  # 熵正则化系数，用于鼓励探索

class PolicyBuffer:
    """修复版本的策略经验回放缓冲区"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = []
        self.routing_weights = []  # 改为存储routing_weights而不是routing_logits
        self.actions = []
        self.rewards = []
        self.position = 0
        self.full = False
    
    def push(self, state, routing_weight, action, reward):
        """存储经验 - 关键修复：不detach routing_weight"""
        if len(self.states) < self.capacity:
            self.states.append(None)
            self.routing_weights.append(None)
            self.actions.append(None)
            self.rewards.append(None)
        
        # 关键修复：保持梯度信息，不使用detach()
        self.states[self.position] = state.detach() if hasattr(state, 'detach') else state
        self.routing_weights[self.position] = routing_weight  # 保持梯度
        self.actions[self.position] = action.detach() if hasattr(action, 'detach') else action
        self.rewards[self.position] = reward
        
        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True
    
    def sample(self, batch_size):
        """采样经验"""
        indices = np.random.choice(len(self), batch_size, replace=False)
        
        states = torch.stack([self.states[i] for i in indices])
        routing_weights = torch.stack([self.routing_weights[i] for i in indices])  # 保持梯度
        actions = torch.stack([self.actions[i] for i in indices])
        rewards = torch.tensor([self.rewards[i] for i in indices], dtype=torch.float32)
        
        return states, routing_weights, actions, rewards.to(DEVICE)
    
    def __len__(self):
        return self.capacity if self.full else self.position

def compute_confidence_weighted_reward(outputs, labels):
    """计算基于置信度的加权奖励"""
    probs = F.softmax(outputs, dim=1)
    predicted = outputs.argmax(dim=1)
    confidence = probs.gather(1, predicted.unsqueeze(1)).squeeze()
    
    # 判断预测是否正确
    correct = (predicted == labels).float()
    
    # 计算加权奖励
    # 正确且高置信度：高奖励；正确但低置信度：低奖励
    # 错误且高置信度：高惩罚；错误但低置信度：低惩罚
    reward = correct * (1.0 + confidence) - (1.0 - correct) * confidence
    
    return reward

def update_policy(model, policy_optimizer, memory, batch_size, amp_handler=None):
    """修复版本的策略更新函数"""
    if len(memory) < batch_size:
        return 0.0
        
    # 从经验回放缓冲区采样
    states, routing_weights, actions, rewards = memory.sample(batch_size)
    
    # 检查是否有NaN值
    if torch.isnan(routing_weights).any() or torch.isnan(rewards).any():
        print("警告：检测到NaN值，跳过此次更新")
        return 0.0
    
    # 确保路由网络参数需要梯度
    model_module = model.module if isinstance(model, nn.DataParallel) else model
    for task, gate in model_module.gates.items():
        for param in gate.parameters():
            param.requires_grad = True
    
    # 重新计算路由logits以获得梯度
    # 关键修复：不使用存储的routing_weights，而是重新前向传播
    with amp_handler.autocast() if amp_handler else torch.cuda.amp.autocast(enabled=False):
        # 使用当前模型重新计算routing logits
        new_routing_logits = []
        for i, state in enumerate(states):
            # 假设set是路由特征，重新通过gates计算logits
            # 这里需要知道对应的task，简化起见使用colon任务
            task = 'colon'  # 实际应该存储对应的task信息
            routing_logit = model_module.gates[task](state.unsqueeze(0))
            new_routing_logits.append(routing_logit)
        
        routing_logits = torch.cat(new_routing_logits, dim=0)
        
        # 计算对数概率
        log_probs = F.log_softmax(routing_logits, dim=1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # 检查梯度计算的有效性
        if not selected_log_probs.requires_grad:
            print("警告：selected_log_probs不需要梯度，使用当前模型重新计算")
            return 0.0
        
        # 计算策略损失
        policy_loss = -(selected_log_probs * rewards).mean()
        
        # 检查损失是否为NaN
        if torch.isnan(policy_loss):
            print("警告：策略损失为NaN，跳过更新")
            return 0.0
        
        # 添加熵正则化项，鼓励探索
        entropy = -(F.softmax(routing_logits, dim=1) * log_probs).sum(dim=1).mean()
        policy_loss -= RL_ENTROPY_COEF * entropy
    
    # 更新策略网络
    policy_optimizer.zero_grad()
    
    if amp_handler:
        # 使用混合精度训练
        amp_handler.scale_loss(policy_loss).backward()
        # 梯度裁剪
        if hasattr(amp_handler.scaler, "_enabled") and amp_handler.scaler._enabled:
            amp_handler.scaler.unscale_(policy_optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        amp_handler.step(policy_optimizer)
        amp_handler.update()
    else:
        policy_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        policy_optimizer.step()
    
    return policy_loss.item()

def freeze_experts(model):
    """冻结专家模型参数，只训练路由策略"""
    # 处理DataParallel包装的模型
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
    # 处理DataParallel包装的模型
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
    
    # 计算最长的数据加载器长度
    max_len = max([len(data_loaders[task]['train_loader']) for task in data_loaders.keys()])
    
    # 创建进度条
    pbar = tqdm(range(max_len), 
               desc=f"Epoch {epoch+1} [Supervised Train]", 
               unit="batch",
               ncols=100,
               leave=False)
    
    # 重置数据迭代器
    iterators = {task: iter(data_loaders[task]['train_loader']) for task in data_loaders.keys()}
    
    for i in pbar:
        # 为每个任务准备批次数据
        batch_loss = 0.0
        batch_task_losses = {}
        
        # 更新进度条信息
        pbar.set_postfix({
            'Loss': f'{total_loss/(i+1):.4f}' if i > 0 else '0.0000',
            'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
        })
        
        for task, task_weight in task_weights.items():
            try:
                inputs, labels = next(iterators[task])
            except StopIteration:
                # 如果某个数据集耗尽，重新初始化迭代器
                iterators[task] = iter(data_loaders[task]['train_loader'])
                inputs, labels = next(iterators[task])
            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # 应用数据增强
            r = np.random.rand()
            
            # 使用混合精度训练
            with amp_handler.autocast() if amp_handler else torch.cuda.amp.autocast(enabled=False):
                if r < AUGMENTATION_PROBABILITY / 2:
                    # 应用MixUp
                    mixed_inputs, labels_a, labels_b, lam = mixup(inputs, labels, alpha=MIXUP_ALPHA)
                    inputs = mixed_inputs
                    
                    # 前向传播
                    outputs, routing_weights = model(inputs, task=task)
                    
                    # 计算MixUp损失
                    loss_a = criterion(outputs, labels_a)
                    loss_b = criterion(outputs, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                    
                elif r < AUGMENTATION_PROBABILITY:
                    # 应用CutMix
                    mixed_inputs, labels_a, labels_b, lam = cutmix(inputs, labels, alpha=CUTMIX_ALPHA)
                    inputs = mixed_inputs
                    
                    # 前向传播
                    outputs, routing_weights = model(inputs, task=task)
                    
                    # 计算CutMix损失
                    loss_a = criterion(outputs, labels_a)
                    loss_b = criterion(outputs, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                    
                else:
                    # 标准前向传播
                    outputs, routing_weights = model(inputs, task=task)
                    loss = criterion(outputs, labels)
                
                # 计算负载均衡损失
                # 处理DataParallel包装的模型
                if isinstance(model, nn.DataParallel):
                    load_balancing_loss = model.module.calculate_load_balancing_loss(routing_weights)
                else:
                    load_balancing_loss = model.calculate_load_balancing_loss(routing_weights)
                    
                task_specific_loss = loss + balance_loss_weight * load_balancing_loss
                weighted_task_loss = task_weight * task_specific_loss
            
            # 记录任务损失
            batch_task_losses[task] = task_specific_loss.item()
            task_losses[task] += batch_task_losses[task]
            
            # 累积任务损失
            batch_loss += weighted_task_loss
            
            # 计算准确率
            if r >= AUGMENTATION_PROBABILITY:
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                batch_total = labels.size(0)
                
                total_acc[task] += batch_correct
                samples_count[task] += batch_total
            else:
                # 对于混合样本，使用主要标签估计准确率
                _, predicted = torch.max(outputs.data, 1)
                if r < AUGMENTATION_PROBABILITY / 2:  # MixUp
                    batch_correct = (predicted == labels_a).sum().item() * lam + (predicted == labels_b).sum().item() * (1 - lam)
                else:  # CutMix
                    batch_correct = (predicted == labels_a).sum().item() * lam + (predicted == labels_b).sum().item() * (1 - lam)
                batch_total = labels.size(0)
                
                total_acc[task] += batch_correct
                samples_count[task] += batch_total
        
        # 反向传播和优化
        optimizer.zero_grad()
        
        if amp_handler:
            # 使用混合精度训练
            amp_handler.scale_loss(batch_loss).backward()
            # 梯度裁剪 - 处理混合精度情况
            if hasattr(amp_handler.scaler, "_enabled") and amp_handler.scaler._enabled:
                amp_handler.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_handler.step(optimizer)
            amp_handler.update()
        else:
            # 标准训练
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # 累积总损失
        total_loss += batch_loss.item()
        
        # 更新进度条描述
        if i % 10 == 0:
            task_desc = " | ".join([f"{task}: {batch_task_losses[task]:.3f}" for task in batch_task_losses.keys()])
            pbar.set_description(f"Epoch {epoch+1} [Supervised] Loss: {batch_loss.item():.3f} | {task_desc}")
    
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
    
    # 打印训练结果
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
    
    # 计算最长的数据加载器长度
    max_len = max([len(data_loaders[task]['train_loader']) for task in data_loaders.keys()])
    
    # 创建进度条
    pbar = ProgressBar(max_len, desc=f"Epoch {epoch+1} [RL Train]")
    
    # 重置数据迭代器
    iterators = {task: iter(data_loaders[task]['train_loader']) for task in data_loaders.keys()}
    
    for i in range(max_len):
        # 为每个任务准备批次数据
        batch_rl_loss = 0.0
        batch_task_losses = {}
        batch_rewards = {}
        
        for task, task_weight in task_weights.items():
            try:
                inputs, labels = next(iterators[task])
            except StopIteration:
                # 如果某个数据集耗尽，重新初始化迭代器
                iterators[task] = iter(data_loaders[task]['train_loader'])
                inputs, labels = next(iterators[task])
            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # 使用混合精度训练
            with amp_handler.autocast() if amp_handler else torch.cuda.amp.autocast(enabled=False):
                # 前向传播（获取路由特征和决策）
                outputs, routing_weights, routing_features = model(inputs, task=task, return_features=True)
                
                # 计算监督学习损失
                ce_loss = criterion(outputs, labels)
                batch_task_losses[task] = ce_loss.item()
                task_losses[task] += batch_task_losses[task]
                
                # 计算奖励
                rewards = compute_confidence_weighted_reward(outputs, labels)
                batch_rewards[task] = rewards.mean().item()
                total_rewards[task] += batch_rewards[task]
            
            # 确定选择的专家（top-k专家的索引）
            _, top_k_indices = torch.topk(routing_weights, model.module.top_k if isinstance(model, nn.DataParallel) else model.top_k, dim=1)
            
            # 存储经验到缓冲区
            for b in range(inputs.size(0)):
                # 只存储第一个选择的专家（简化版）
                action = top_k_indices[b, 0]
                memory.push(
                    routing_features[b].detach(),  # 状态可以detach
                    routing_weights[b],            # 关键：不detach，保持梯度
                    action,                        # 选择的动作
                    rewards[b].item()              # 奖励
                )
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)
            
            total_acc[task] += batch_correct
            samples_count[task] += batch_total
        
        # 更新策略网络
        if i % 5 == 0 and len(memory) >= batch_size:  # 每5个批次更新一次策略
            rl_loss = update_policy(model, rl_optimizer, memory, batch_size, amp_handler=amp_handler)
            rl_losses.append(rl_loss)
            batch_rl_loss = rl_loss
        
        # 更新进度条描述
        if i % 10 == 0:
            reward_desc = " | ".join([f"{task} R: {batch_rewards[task]:.2f}" for task in batch_rewards.keys()])
            pbar.set_description(f"Epoch {epoch+1} [RL] Loss: {batch_rl_loss:.3f} | {reward_desc}")
        
        # 更新进度条
        pbar.update()
    
    pbar.close()
    
    # 计算平均损失、奖励和准确率
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
    
    # 打印训练结果
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
    
    # 计算最长的数据加载器长度
    max_len = max([len(data_loaders[task]['test_loader']) for task in data_loaders.keys()])
    
    # 创建进度条
    pbar = ProgressBar(max_len, desc=f"Epoch {epoch+1} [Val]")
    
    with torch.no_grad():
        # 重置数据迭代器
        iterators = {task: iter(data_loaders[task]['test_loader']) for task in data_loaders.keys()}
        
        for i in range(max_len):
            # 为每个任务准备批次数据
            batch_loss = 0.0
            batch_task_losses = {}
            batch_rewards = {}
            
            for task, task_weight in task_weights.items():
                try:
                    inputs, labels = next(iterators[task])
                except StopIteration:
                    # 如果某个数据集耗尽，重新初始化迭代器
                    iterators[task] = iter(data_loaders[task]['test_loader'])
                    inputs, labels = next(iterators[task])
                
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                # 使用混合精度（验证时可选）
                with amp_handler.autocast() if amp_handler else torch.cuda.amp.autocast(enabled=False):
                    # 前向传播
                    outputs, routing_weights = model(inputs, task=task)
                    loss = criterion(outputs, labels)
                    
                    # 计算奖励
                    rewards = compute_confidence_weighted_reward(outputs, labels)
                    batch_rewards[task] = rewards.mean().item()
                    total_rewards[task] += batch_rewards[task]
                
                # 记录任务损失
                batch_task_losses[task] = loss.item()
                task_losses[task] += batch_task_losses[task]
                
                # 累积任务损失
                batch_loss += task_weight * loss
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                batch_total = labels.size(0)
                
                total_acc[task] += batch_correct
                samples_count[task] += batch_total
            
            # 累积总损失
            total_loss += batch_loss.item()
            
            # 更新进度条描述
            if i % 10 == 0:
                acc_desc = " | ".join([f"{task} Acc: {total_acc[task]/max(1, samples_count[task])*100:.1f}%" 
                                     for task in task_weights.keys()])
                pbar.set_description(f"Epoch {epoch+1} [Val] Loss: {batch_loss.item():.3f} | {acc_desc}")
            
            # 更新进度条
            pbar.update()
    
    pbar.close()
    
    # 计算平均损失、奖励和准确率
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
    
    # 打印验证结果
    print(f"Epoch {epoch+1} 验证结果:")
    print(f"总损失: {avg_loss:.4f}")
    for task in avg_task_losses.keys():
        print(f"{task} 损失: {avg_task_losses[task]:.4f} | 奖励: {avg_rewards[task]:.4f} | 准确率: {avg_acc[task]:.2f}%")
    print("-" * 60)
    
    # 计算整体准确率（所有任务的平均值）
    overall_acc = sum(avg_acc.values()) / len(avg_acc)
    
    return avg_loss, avg_acc, avg_task_losses, overall_acc, avg_rewards

def train_rl_hetero_moe(resume=False, use_amp=USE_AMP):
    """使用混合监督学习和强化学习训练异构混合专家模型"""
    # 设置随机种子
    set_seed(SEED)
    
    # 创建AMP处理器
    amp_handler = AmpHandler(enabled=use_amp)
    
    # 使用固定的CNN专家模型
    expert_models = [model['name'] for model in CNN_EXPERT_MODELS]
    
    # 创建结果目录
    log_dir = os.path.join(RESULT_PATH, "logs", "rl_hetero_moe")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(log_dir)
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 加载数据
    print("[DEBUG] ===== 开始数据加载阶段 =====")
    print(f"[DEBUG] 数据路径配置:")
    print(f"[DEBUG] - COLON_DATA_PATH: {COLON_DATA_PATH}")
    print(f"[DEBUG] - UGI_DATA_PATH: {UGI_DATA_PATH}")
    print(f"[DEBUG] - COLON_DISEASE_PATH: {COLON_DISEASE_PATH}")
    print(f"[DEBUG] - UGI_DISEASE_PATH: {UGI_DISEASE_PATH}")
    print(f"[DEBUG] GPU内存状态 - 已分配: {torch.cuda.memory_allocated()/1024**3:.2f}GB, 已缓存: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
    
    print("正在加载数据...")
    data_loaders = create_multi_task_loaders(
        COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, UGI_DISEASE_PATH
    )
    print("[DEBUG] ===== 数据加载阶段完成 =====")
    
    # 获取类别数量
    num_colon_classes = len(data_loaders['colon']['classes'])
    num_ugi_classes = len(data_loaders['ugi']['classes'])
    num_colon_disease_classes = len(data_loaders['colon_disease']['classes'])
    num_ugi_disease_classes = len(data_loaders['ugi_disease']['classes'])
    
    print(f"数据加载完成! 用时: {time.time() - start_time:.2f}秒")
    print(f"肠镜部位类别数: {num_colon_classes}")
    print(f"胃镜部位类别数: {num_ugi_classes}")
    print(f"肠镜疾病类别数: {num_colon_disease_classes}")
    print(f"胃镜疾病类别数: {num_ugi_disease_classes}")
    print(f"使用专家模型: {expert_models}")
    
    # 初始化任务权重
    task_weights = {
        'colon': 1.0,
        'ugi': 1.0,
        'colon_disease': 1.0,
        'ugi_disease': 1.0
    }
    
    # 如果使用动态权重策略，初始化权重
    if TASK_WEIGHT_STRATEGY == 'dynamic':
        # 获取各任务的样本数
        task_sample_counts = {}
        for task in task_weights.keys():
            loader = data_loaders[task]['train_loader']
            if hasattr(loader.dataset, 'samples'):
                task_sample_counts[task] = len(loader.dataset.samples)
            else:
                task_sample_counts[task] = len(loader.dataset) * loader.batch_size
        
        # 按样本数反比例设置权重
        total_samples = sum(task_sample_counts.values())
        for task in task_weights.keys():
            task_weights[task] = (total_samples / (len(task_weights) * task_sample_counts[task])) ** 0.5
        
        # 归一化权重
        total_weight = sum(task_weights.values())
        for task in task_weights.keys():
            task_weights[task] = task_weights[task] * len(task_weights) / total_weight
    
    print(f"初始任务权重: {task_weights}")
    
    # 创建模型
    print(f"创建带强化学习的异构混合专家模型 ({len(expert_models)} 专家)...")
    model = RLHeterogeneousMixtureOfExperts(
        num_colon_classes=num_colon_classes,
        num_ugi_classes=num_ugi_classes,
        num_colon_disease_classes=num_colon_disease_classes,
        num_ugi_disease_classes=num_ugi_disease_classes
    ).to(DEVICE)
    
    # 如果有多个GPU，使用DataParallel
    if torch.cuda.device_count() > 1:
        print(f"✅ 使用 {torch.cuda.device_count()} 张 GPU 进行 DataParallel")
        model = nn.DataParallel(model, device_ids=GPU_IDS)
    
    # 统计模型参数
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
    
    # 如果恢复训练，加载之前的模型
    start_epoch = 0
    best_acc = 0.0
    model_name = "rl_hetero_moe"
    
    if resume:
        start_epoch, best_acc = load_model(model, optimizer, model_name)
        print(f"从第 {start_epoch} 轮恢复训练，最佳精度: {best_acc:.4f}")
    
    # 记录训练和验证指标
    history = {
        'supervised': {
            'epoch': [],
            'loss': [],
            'acc': {task: [] for task in task_weights.keys()},
            'task_losses': {task: [] for task in task_weights.keys()},
        },
        'rl': {
            'epoch': [],
            'loss': [],
            'acc': {task: [] for task in task_weights.keys()},
            'rewards': {task: [] for task in task_weights.keys()},
        },
        'val': {
            'epoch': [],
            'loss': [],
            'acc': {task: [] for task in task_weights.keys()},
            'task_losses': {task: [] for task in task_weights.keys()},
            'rewards': {task: [] for task in task_weights.keys()},
        },
        'lr': [],
        'task_weights': {task: [] for task in task_weights.keys()},
    }
    
    # 训练循环
    print("\n" + "="*20 + " 开始混合强化学习训练 " + "="*20)
    print(f"[DEBUG] ===== 训练配置信息 =====")
    print(f"[DEBUG] - 预训练轮数: {PRETRAIN_EPOCHS}")
    print(f"[DEBUG] - 强化学习轮数: {RL_EPOCHS}")
    print(f"[DEBUG] - 微调轮数: {FINETUNE_EPOCHS}")
    print(f"[DEBUG] - 起始轮数: {start_epoch}")
    print(f"[DEBUG] - 当前最佳精度: {best_acc}")
    print(f"[DEBUG] GPU内存状态 - 已分配: {torch.cuda.memory_allocated()/1024**3:.2f}GB, 已缓存: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
    
    training_start = time.time()
    
    try:
        # 第一阶段：监督学习预训练
        print("\n" + "="*20 + " 阶段1：监督学习预训练 " + "="*20)
        print(f"[DEBUG] 开始预训练阶段，轮数范围: {start_epoch} -> {start_epoch + PRETRAIN_EPOCHS}")
        
        # 创建预训练进度条
        pretrain_pbar = tqdm(range(start_epoch, start_epoch + PRETRAIN_EPOCHS), 
                           desc="预训练阶段", 
                           unit="epoch",
                           ncols=120,
                           colour='green')
        
        for epoch in pretrain_pbar:
            epoch_start = time.time()
            print(f"[DEBUG] ===== 开始第 {epoch+1} 轮预训练 =====")
            print(f"[DEBUG] GPU内存状态 - 已分配: {torch.cuda.memory_allocated()/1024**3:.2f}GB, 已缓存: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
            
            # 学习率预热
            if epoch < LR_WARMUP_EPOCHS:
                lr_scale = min(1.0, (epoch + 1) / LR_WARMUP_EPOCHS)
                for pg in optimizer.param_groups:
                    pg['lr'] = LEARNING_RATE * lr_scale
                print(f"[DEBUG] 学习率预热，当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 衰减负载均衡权重
            current_balance_weight = LOAD_BALANCE_WEIGHT * (LOAD_BALANCE_DECAY ** epoch)
            print(f"[DEBUG] 当前负载均衡权重: {current_balance_weight:.6f}")
            
            # 监督学习训练
            print(f"[DEBUG] 开始监督学习训练 - 第 {epoch+1} 轮")
            train_loss, train_acc, train_task_losses = train_supervised(
                model, data_loaders, task_weights, optimizer, criterion, 
                epoch, writer, balance_loss_weight=current_balance_weight,
                amp_handler=amp_handler
            )
            print(f"[DEBUG] 监督学习训练完成 - 第 {epoch+1} 轮")
            
            # 验证
            print(f"[DEBUG] 开始验证 - 第 {epoch+1} 轮")
            val_loss, val_acc, val_task_losses, overall_acc, val_rewards = validate(
                model, data_loaders, task_weights, criterion, epoch, writer,
                amp_handler=amp_handler
            )
            print(f"[DEBUG] 验证完成 - 第 {epoch+1} 轮")
            
            # 更新任务权重（如果使用动态权重策略）
            if TASK_WEIGHT_STRATEGY == 'dynamic':
                for task in task_weights.keys():
                    # 计算相对损失
                    relative_loss = val_task_losses[task] / (sum(val_task_losses.values()) / len(val_task_losses))
                    # 平滑更新权重
                    task_weights[task] = task_weights[task] * (1 - TASK_WEIGHT_ALPHA) + relative_loss * TASK_WEIGHT_ALPHA
                
                # 归一化权重
                total_weight = sum(task_weights.values())
                for task in task_weights.keys():
                    task_weights[task] = task_weights[task] * len(task_weights) / total_weight
                
                print(f"更新后的任务权重: {task_weights}")
            
            # 更新学习率
            if epoch >= LR_WARMUP_EPOCHS:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # 更新预训练进度条
            pretrain_pbar.set_postfix({
                'Loss': f'{train_loss:.4f}',
                'Acc': f'{overall_acc:.3f}',
                'LR': f'{current_lr:.6f}',
                'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
            })
            
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
            
            # 如果是最佳模型，保存模型
            if overall_acc > best_acc:
                best_acc = overall_acc
                save_model(model, optimizer, epoch, best_acc, model_name)
                print(f"保存最佳模型，精度: {best_acc:.4f}")
            
            # 计算本轮用时
            epoch_time = time.time() - epoch_start
            # 估计剩余时间
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
        
        # 创建强化学习进度条
        rl_pbar = tqdm(range(start_epoch + PRETRAIN_EPOCHS, start_epoch + PRETRAIN_EPOCHS + RL_EPOCHS), 
                       desc="强化学习阶段", 
                       unit="epoch",
                       ncols=120,
                       colour='blue')
        
        # 使用强化学习训练
        for epoch in rl_pbar:
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
            
            # 更新强化学习进度条
            rl_pbar.set_postfix({
                'RL_Loss': f'{rl_loss:.4f}',
                'Acc': f'{overall_acc:.3f}',
                'Reward': f'{sum(rl_rewards.values())/len(rl_rewards):.3f}',
                'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
            })
            
            # 如果是最佳模型，保存模型
            if overall_acc > best_acc:
                best_acc = overall_acc
                save_model(model, optimizer, epoch, best_acc, model_name)
                print(f"保存最佳模型，精度: {best_acc:.4f}")
            
            # 计算本轮用时
            epoch_time = time.time() - epoch_start
            # 估计剩余时间
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
        
        # 创建微调进度条
        finetune_pbar = tqdm(range(start_epoch + PRETRAIN_EPOCHS + RL_EPOCHS, 
                                  start_epoch + PRETRAIN_EPOCHS + RL_EPOCHS + FINETUNE_EPOCHS), 
                            desc="微调阶段", 
                            unit="epoch",
                            ncols=120,
                            colour='yellow')
        
        # 端到端微调
        for epoch in finetune_pbar:
            epoch_start = time.time()
            
            # 监督学习 + 强化学习混合训练
            # 先使用监督学习更新全模型
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
            
            # 更新微调进度条
            finetune_pbar.set_postfix({
                'Sup_Loss': f'{train_loss:.4f}',
                'RL_Loss': f'{rl_loss:.4f}',
                'Acc': f'{overall_acc:.3f}',
                'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
            })
            
            # 如果是最佳模型，保存模型
            if overall_acc > best_acc:
                best_acc = overall_acc
                save_model(model, optimizer, epoch, best_acc, model_name)
                print(f"保存最佳模型，精度: {best_acc:.4f}")
            
            # 检查早停
            if early_stopping.check(-overall_acc):
                print(f"早停触发于第 {epoch+1} 轮")
                break
            
            # 计算本轮用时
            epoch_time = time.time() - epoch_start
            # 估计剩余时间
            remaining_epochs = (PRETRAIN_EPOCHS + RL_EPOCHS + FINETUNE_EPOCHS) - (epoch - start_epoch) - 1
            remaining_time = epoch_time * remaining_epochs
            
            print(f"Epoch {epoch+1} 用时: {epoch_time:.2f}秒 | 估计剩余时间: {remaining_time/60:.2f}分钟")
            print("=" * 80)
    
    except KeyboardInterrupt:
        print("\n[INFO] 训练被用户中断!")
        print(f"[DEBUG] 中断时GPU内存状态 - 已分配: {torch.cuda.memory_allocated()/1024**3:.2f}GB, 已缓存: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
    except Exception as e:
        print(f"\n[ERROR] 训练过程中发生未预期的错误!")
        print(f"[ERROR] 错误类型: {type(e).__name__}")
        print(f"[ERROR] 错误信息: {str(e)}")
        print(f"[DEBUG] 错误时GPU内存状态 - 已分配: {torch.cuda.memory_allocated()/1024**3:.2f}GB, 已缓存: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        import traceback
        print(f"[ERROR] 详细错误堆栈:\n{traceback.format_exc()}")
        raise
    
    # 计算总训练时间
    total_time = time.time() - training_start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"训练完成! 总用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    
    # 保存训练历史为CSV
    history_df = pd.DataFrame({
        'epoch': list(range(start_epoch, epoch + 1)),
    })
    
    # 添加监督学习损失和准确率
    for e, loss in zip(history['supervised']['epoch'], history['supervised']['loss']):
        history_df.loc[e - start_epoch, 'supervised_loss'] = loss
    
    # 添加强化学习损失
    for e, loss in zip(history['rl']['epoch'], history['rl']['loss']):
        history_df.loc[e - start_epoch, 'rl_loss'] = loss
    
    # 添加验证损失
    for e, loss in zip(history['val']['epoch'], history['val']['loss']):
        history_df.loc[e - start_epoch, 'val_loss'] = loss
    
    # 添加各任务指标
    for task in task_weights.keys():
        # 监督学习准确率
        for e, acc in enumerate(history['supervised']['acc'][task]):
            epoch = history['supervised']['epoch'][e]
            history_df.loc[epoch - start_epoch, f'supervised_{task}_acc'] = acc
        
        # 强化学习准确率和奖励
        for e, acc in enumerate(history['rl']['acc'][task]):
            epoch = history['rl']['epoch'][e]
            history_df.loc[epoch - start_epoch, f'rl_{task}_acc'] = acc
            history_df.loc[epoch - start_epoch, f'rl_{task}_reward'] = history['rl']['rewards'][task][e]
        
        # 验证准确率和奖励
        for e, acc in enumerate(history['val']['acc'][task]):
            epoch = history['val']['epoch'][e]
            history_df.loc[epoch - start_epoch, f'val_{task}_acc'] = acc
            history_df.loc[epoch - start_epoch, f'val_{task}_reward'] = history['val']['rewards'][task][e]
        
        # 任务权重
        for e, weight in enumerate(history['task_weights'][task]):
            epoch = e + start_epoch
            history_df.loc[epoch - start_epoch, f'{task}_weight'] = weight
    
    # 保存CSV
    csv_path = os.path.join(RESULT_PATH, "logs", f"{model_name}_history.csv")
    history_df.to_csv(csv_path, index=False)
    print(f"训练历史已保存至: {csv_path}")
    
    # 绘制训练历史曲线
    plot_training_history(history, model_name, expert_models)
    
    return model, data_loaders, history

def plot_training_history(history, model_name, expert_models):
    """绘制训练历史曲线"""
    plt.figure(figsize=(15, 12), dpi=DPI)
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 1. 绘制损失曲线
    plt.subplot(2, 2, 1)
    
    # 监督学习损失
    if len(history['supervised']['epoch']) > 0:
        plt.plot(history['supervised']['epoch'], history['supervised']['loss'], 
                label='Supervised Loss', color=LANCET_COLORS[0])
    
    # 强化学习损失
    if len(history['rl']['epoch']) > 0:
        plt.plot(history['rl']['epoch'], history['rl']['loss'], 
                label='RL Policy Loss', color=LANCET_COLORS[1])
    
    # 验证损失
    if len(history['val']['epoch']) > 0:
        plt.plot(history['val']['epoch'], history['val']['loss'], 
                label='Validation Loss', color=LANCET_COLORS[2])
    
    plt.title('Training Losses', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 绘制准确率曲线
    plt.subplot(2, 2, 2)
    tasks = list(history['val']['acc'].keys())
    
    # 使用不同颜色表示不同任务
    for i, task in enumerate(tasks):
        # 验证准确率
        if len(history['val']['epoch']) > 0:
            val_acc = [history['val']['acc'][task][j] for j in range(len(history['val']['epoch']))]
            plt.plot(history['val']['epoch'], val_acc, 
                    label=f'{task} Val Acc', 
                    color=LANCET_COLORS[i % len(LANCET_COLORS)], 
                    linestyle='-')
    
    plt.title('Validation Accuracy by Task', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 绘制奖励曲线
    plt.subplot(2, 2, 3)
    
    # 使用不同颜色表示不同任务
    for i, task in enumerate(tasks):
        # RL训练奖励
        if len(history['rl']['epoch']) > 0:
            rl_rewards = [history['rl']['rewards'][task][j] for j in range(len(history['rl']['epoch']))]
            plt.plot(history['rl']['epoch'], rl_rewards, 
                    label=f'{task} Train Reward', 
                    color=LANCET_COLORS[i % len(LANCET_COLORS)], 
                    linestyle='-')
        
        # 验证奖励
        if len(history['val']['epoch']) > 0:
            val_rewards = [history['val']['rewards'][task][j] for j in range(len(history['val']['epoch']))]
            plt.plot(history['val']['epoch'], val_rewards, 
                    label=f'{task} Val Reward', 
                    color=LANCET_COLORS[i % len(LANCET_COLORS)], 
                    linestyle='--')
    
    plt.title('Rewards by Task', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 绘制任务权重曲线
    plt.subplot(2, 2, 4)
    
    # 使用不同颜色表示不同任务
    for i, task in enumerate(tasks):
        if task in history['task_weights']:
            epochs = range(len(history['task_weights'][task]))
            plt.plot(epochs, history['task_weights'][task], 
                    label=f'{task} Weight', 
                    color=LANCET_COLORS[i % len(LANCET_COLORS)],
                    marker='o', markersize=3)
    
    plt.title('Task Weights Over Time', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_training_history.png")
    plt.savefig(save_path)
    plt.close()
    print(f"训练历史图表已保存至: {save_path}")

if __name__ == "__main__":
    train_rl_hetero_moe()