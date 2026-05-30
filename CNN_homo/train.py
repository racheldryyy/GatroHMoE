import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 不再使用 TkAgg，避免触发 Tkinter
import pandas as pd


from config import (
    DEVICE, GPU_IDS, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, SEED, USE_AMP,
    COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, UGI_DISEASE_PATH, RESULT_PATH,
    DISTILLATION_TEMP, ALPHA, LANCET_COLORS, FIG_SIZE, DPI, INPUT_SIZE,
    LABEL_SMOOTHING, PATIENCE, MIN_DELTA, LR_WARMUP_EPOCHS, LR_MIN_FACTOR,
    TASK_WEIGHT_STRATEGY, TASK_WEIGHT_ALPHA, BATCH_SIZE, USE_FOCAL_LOSS,
    FOCAL_LOSS_TYPE, FOCAL_GAMMA, FOCAL_ALPHA, USE_GRADIENT_CHECKPOINTING,
    MEMORY_OPTIMIZATION, MAX_MEMORY_USAGE_PERCENT
)
# 注释掉可能导致阻塞的GPU工具导入
# from gpu_utils import SmartGPUManager, monitor_gpu_usage
from focal_loss import create_focal_loss
from utils import (
    set_seed, ProgressBar, save_training_curve, save_model, load_model,
    save_detailed_curves, EarlyStopping
)
from data_loader import create_multi_task_loaders, mixup, cutmix
from models.moe_model import MixtureOfExperts
from models.student_model import StudentModel
from amp_utils import AmpHandler
from memory_utils import MemoryManager, memory_monitor, get_memory_info, generate_memory_config
from torch.utils.tensorboard import SummaryWriter

def train_single_epoch(model, data_loaders, task_weights, optimizer, criterion, 
                      epoch, writer, balance_loss_weight=0.1, amp_handler=None):
    """执行单轮训练，处理多任务数据并计算损失"""
    model.train()
    total_loss = 0.0
    task_losses = {task: 0.0 for task in data_loaders.keys()}
    total_acc = {task: 0.0 for task in data_loaders.keys()}
    samples_count = {task: 0 for task in data_loaders.keys()}
    
    # 找出所有任务中最长的数据加载器，用于统一训练步数
    max_len = max([len(data_loaders[task]['train_loader']) for task in data_loaders.keys()])
    
    # 初始化进度条显示
    pbar = ProgressBar(max_len, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    
    # 为每个任务创建数据迭代器
    iterators = {task: iter(data_loaders[task]['train_loader']) for task in data_loaders.keys()}
    
    # 梯度累积设置，用于模拟更大的批次大小 - 16GB内存优化
    accumulation_steps = 4  # 增加梯度累积步数以减少内存使用
    optimizer.zero_grad()
    
    for i in range(max_len):
        # 处理当前批次的所有任务
        batch_loss = 0.0
        batch_task_losses = {}
        
        # 依次处理每个任务的数据
        for task, task_weight in task_weights.items():
            try:
                inputs, labels = next(iterators[task])
            except StopIteration:
                # 当某个任务的数据用完时，重新开始遍历
                iterators[task] = iter(data_loaders[task]['train_loader'])
                inputs, labels = next(iterators[task])
            
            # 将数据转移到GPU，使用非阻塞传输提高效率
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            # 使用混合精度训练
            with amp_handler.autocast() if amp_handler else torch.no_grad():
                # 前向传播
                outputs, routing_weights = model(inputs, task=task)
                loss = criterion(outputs, labels)
                
                # 计算负载均衡损失
                if isinstance(model, nn.DataParallel):
                    # 如果模型被DataParallel包装
                    load_balancing_loss = model.module.calculate_load_balancing_loss(routing_weights)
                else:
                    # 如果模型没有被包装
                    load_balancing_loss = model.calculate_load_balancing_loss(routing_weights)
                    
                task_specific_loss = loss + balance_loss_weight * load_balancing_loss
                weighted_task_loss = task_weight * task_specific_loss / accumulation_steps
            
            # 记录任务损失
            batch_task_losses[task] = task_specific_loss.item()
            task_losses[task] += batch_task_losses[task]
            
            # 累积任务损失
            batch_loss += weighted_task_loss
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)
            batch_acc = batch_correct / batch_total * 100.0
            
            total_acc[task] += batch_correct
            samples_count[task] += batch_total
        
        # 反向传播 (使用梯度累积提高效率)
        if amp_handler:
            # 使用混合精度训练
            amp_handler.scale_loss(batch_loss).backward()
            
            # 仅在累积步骤完成后更新
            if (i + 1) % accumulation_steps == 0:
                # 梯度裁剪 - 处理混合精度情况
                if hasattr(amp_handler.scaler, "_enabled") and amp_handler.scaler._enabled:
                    amp_handler.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                amp_handler.step(optimizer)
                amp_handler.update()
                optimizer.zero_grad()
        else:
            # 标准训练
            batch_loss.backward()
            
            # 仅在累积步骤完成后更新
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        # 累积总损失
        total_loss += batch_loss.item() * accumulation_steps
        
        # 更新进度条描述
        if i % 10 == 0:  # 每10个批次更新一次显示
            task_desc = " | ".join([f"{task}: {batch_task_losses[task]:.3f}" for task in batch_task_losses.keys()])
            pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} [Train] Loss: {batch_loss.item()*accumulation_steps:.3f} | {task_desc}")
        
        # 更新进度条
        pbar.update()
    
    # 确保最后一批的梯度也被应用
    if (max_len % accumulation_steps) != 0:
        if amp_handler:
            if hasattr(amp_handler.scaler, "_enabled") and amp_handler.scaler._enabled:
                amp_handler.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_handler.step(optimizer)
            amp_handler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    pbar.close()
    
    # 计算平均损失和准确率
    avg_loss = total_loss / max_len
    avg_task_losses = {task: task_losses[task] / max_len for task in task_losses.keys()}
    avg_acc = {task: total_acc[task] / samples_count[task] * 100.0 for task in data_loaders.keys()}
    
    # 记录训练指标
    writer.add_scalar('Loss/train/total', avg_loss, epoch)
    for task in avg_task_losses.keys():
        writer.add_scalar(f'Loss/train/{task}', avg_task_losses[task], epoch)
        writer.add_scalar(f'Accuracy/train/{task}', avg_acc[task], epoch)
    
    # 打印训练结果
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} 训练结果:")
    print(f"总损失: {avg_loss:.4f}")
    for task in avg_task_losses.keys():
        print(f"{task} 损失: {avg_task_losses[task]:.4f} | 准确率: {avg_acc[task]:.2f}%")
    print("-" * 60)
    
    return avg_loss, avg_acc, avg_task_losses


def validate(model, data_loaders, task_weights, criterion, epoch, writer, amp_handler=None):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    task_losses = {task: 0.0 for task in data_loaders.keys()}
    total_acc = {task: 0.0 for task in data_loaders.keys()}
    samples_count = {task: 0 for task in data_loaders.keys()}
    
    # 计算最长的数据加载器长度
    max_len = max([len(data_loaders[task]['test_loader']) for task in data_loaders.keys()])
    
    # 创建进度条
    pbar = ProgressBar(max_len, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
    
    with torch.no_grad():
        # 使用迭代器的字典
        iterators = {task: iter(data_loaders[task]['test_loader']) for task in data_loaders.keys()}
        
        for i in range(max_len):
            # 为每个任务准备批次数据
            batch_loss = 0.0
            batch_task_losses = {}
            
            for task, task_weight in task_weights.items():
                try:
                    inputs, labels = next(iterators[task])
                except StopIteration:
                    # 如果某个数据集耗尽，重新初始化迭代器
                    iterators[task] = iter(data_loaders[task]['test_loader'])
                    inputs, labels = next(iterators[task])
                
                # 使用非阻塞传输加速数据移动
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                
                # 使用混合精度（验证时可选）
                with amp_handler.autocast() if amp_handler else torch.cuda.amp.autocast(enabled=False):
                    # 前向传播
                    outputs, _ = model(inputs, task=task)
                    loss = criterion(outputs, labels)
                
                # 记录任务损失
                batch_task_losses[task] = loss.item()
                task_losses[task] += batch_task_losses[task]
                
                # 累积任务损失
                batch_loss += task_weight * loss
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                batch_total = labels.size(0)
                batch_acc = batch_correct / batch_total * 100.0
                
                total_acc[task] += batch_correct
                samples_count[task] += batch_total
            
            # 累积总损失
            total_loss += batch_loss.item()
            
            # 更新进度条描述
            if i % 10 == 0:  # 每10个批次更新一次显示
                task_desc = " | ".join([f"{task}: {batch_task_losses[task]:.3f}" for task in batch_task_losses.keys()])
                pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} [Val] Loss: {batch_loss.item():.3f} | {task_desc}")
            
            # 更新进度条
            pbar.update()
    
    pbar.close()
    
    # 计算平均损失和准确率
    avg_loss = total_loss / max_len
    avg_task_losses = {task: task_losses[task] / max_len for task in task_losses.keys()}
    avg_acc = {task: total_acc[task] / samples_count[task] * 100.0 for task in data_loaders.keys()}
    
    # 记录验证指标
    writer.add_scalar('Loss/val/total', avg_loss, epoch)
    for task in avg_task_losses.keys():
        writer.add_scalar(f'Loss/val/{task}', avg_task_losses[task], epoch)
        writer.add_scalar(f'Accuracy/val/{task}', avg_acc[task], epoch)
    
    # 打印验证结果
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} 验证结果:")
    print(f"总损失: {avg_loss:.4f}")
    for task in avg_task_losses.keys():
        print(f"{task} 损失: {avg_task_losses[task]:.4f} | 准确率: {avg_acc[task]:.2f}%")
    print("-" * 60)
    
    # 计算整体准确率（所有任务的平均值）
    overall_acc = sum(avg_acc.values()) / len(avg_acc)
    
    return avg_loss, avg_acc, avg_task_losses, overall_acc


@memory_monitor
def train_teacher_model(base_model_name='resnet50', resume_training=False, enable_mixed_precision=USE_AMP):
    """
    训练教师模型 - 内存优化版本
    
    使用指定的骨干网络架构训练一个强大的教师模型，
    为后续的知识蒸馏提供高质量的软标签。
    """

    # 确保实验的可重现性
    set_seed(SEED)
    
    # 初始化内存管理器
    if MEMORY_OPTIMIZATION:
        memory_manager = MemoryManager(MAX_MEMORY_USAGE_PERCENT)
        memory_config = generate_memory_config()
        print("🧠 内存优化模式已启用")
        
        # 根据内存情况调整批次大小
        global BATCH_SIZE
        BATCH_SIZE = memory_manager.auto_batch_size(BATCH_SIZE)
        print(f"📊 自动调整批次大小为: {BATCH_SIZE}")
    else:
        memory_manager = None
    
    # 🚀 智能GPU设置
    print("🔧 配置GPU环境...")
    # gpu_manager = SmartGPUManager(min_memory_gb=2.0, enable_amp=enable_mixed_precision)
    # 简化版本，避免GPU管理器阻塞
    print("简化GPU配置，避免SmartGPUManager阻塞")
    # available_gpus = gpu_manager.initialize()
    
    # 简化GPU检测
    if torch.cuda.is_available():
        available_gpus = [0]  # 使用第一个GPU
        device = torch.device('cuda:0')
        actual_batch_size = BATCH_SIZE
        print(f"✓ 使用GPU: cuda:0, 批量大小: {actual_batch_size}")
    else:
        available_gpus = []
        device = torch.device('cpu')
        actual_batch_size = max(4, BATCH_SIZE // 4)  # CPU模式使用小批次
        print(f"⚠️ 使用CPU, 批量大小: {actual_batch_size}")
    
    # 准备日志和结果保存目录
    experiment_log_dir = os.path.join(RESULT_PATH, "logs", f"teacher_{base_model_name}")
    os.makedirs(experiment_log_dir, exist_ok=True)
    
    # 初始化训练过程记录器
    writer = SummaryWriter(experiment_log_dir)
    
    # 初始化混合精度训练管理器
    amp_handler = AmpHandler(
        enabled=enable_mixed_precision,
        init_scale=2**16,
        growth_factor=2.0,
        backoff_factor=0.5
    )
    # 记录训练开始的时刻
    training_start_time = time.time()
    
    # 加载多任务数据集
    print("正在加载数据...")
    multi_task_data_loaders = create_multi_task_loaders(
        COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, UGI_DISEASE_PATH
    )
    
    # 获取类别数量
    num_colon_classes = len(multi_task_data_loaders['colon']['classes'])
    num_ugi_classes = len(multi_task_data_loaders['ugi']['classes'])
    num_colon_disease_classes = len(multi_task_data_loaders['colon_disease']['classes'])
    num_ugi_disease_classes = len(multi_task_data_loaders['ugi_disease']['classes'])
    
    print(f"数据加载完成! 用时: {time.time() - training_start_time:.2f}秒")
    print(f"肠镜部位类别数: {num_colon_classes}")
    print(f"胃镜部位类别数: {num_ugi_classes}")
    print(f"肠镜疾病类别数: {num_colon_disease_classes}")
    print(f"胃镜疾病类别数: {num_ugi_disease_classes}")
    
    # 任务权重 (基于任务复杂度和样本量自动调整)
    task_weights = {
        'colon': 1.0,
        'ugi': 1.0,
        'colon_disease': 1.0,
        'ugi_disease': 1.0
    }
    
    # 根据各任务的样本数调整权重
    task_sample_counts = {}
    for task in task_weights.keys():
        train_loader = multi_task_data_loaders[task]['train_loader']
        # 获取样本数
        if hasattr(train_loader.dataset, 'samples'):
            task_sample_counts[task] = len(train_loader.dataset.samples)
        else:
            task_sample_counts[task] = len(train_loader.dataset)
    
    # 样本数少的任务给予更高权重
    if TASK_WEIGHT_STRATEGY == 'dynamic':
        total_samples = sum(task_sample_counts.values())
        for task in task_weights.keys():
            # 使用样本数的平方根倒数作为基础权重
            task_weights[task] = np.sqrt(total_samples / task_sample_counts[task])
        
        # 归一化权重
        weight_sum = sum(task_weights.values())
        for task in task_weights.keys():
            task_weights[task] = task_weights[task] / weight_sum * len(task_weights)
    
    print(f"初始任务权重: {task_weights}")
    
    # 创建MoE模型
    print(f"🏗️  创建MoE模型 (教师模型: {base_model_name})...")
    model = MixtureOfExperts(
        base_model_name=base_model_name,
        num_colon_classes=num_colon_classes,
        num_ugi_classes=num_ugi_classes,
        num_colon_disease_classes=num_colon_disease_classes,
        num_ugi_disease_classes=num_ugi_disease_classes,
        use_attention_gate=True
    )
    
    # 🚀 简化模型设置（避免GPU管理器阻塞）
    model = model.to(device)
    if len(available_gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=available_gpus)
        print(f"✓ 使用多GPU并行: {available_gpus}")
    
    # 设置混合精度训练
    scaler = torch.cuda.amp.GradScaler() if enable_mixed_precision and device.type == 'cuda' else None
    optimized_batch_size = actual_batch_size
    
    print(f"✓ 模型已部署到设备: {device}")
    if scaler:
        print("✓ 混合精度训练已启用")
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建优化器和学习率调度器
    # 使用AdamW优化器，配置更好的优化参数
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 使用余弦退火学习率调度器，带热重启
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # 首次重启的周期
        T_mult=2,  # 每次重启后周期长度的倍增因子
        eta_min=LEARNING_RATE * LR_MIN_FACTOR  # 最小学习率
    )
    
    # 🎯 智能损失函数选择
    if USE_FOCAL_LOSS:
        print(f"🎯 使用 {FOCAL_LOSS_TYPE} Focal Loss (gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA})")
        criterion = create_focal_loss(
            loss_type=FOCAL_LOSS_TYPE,
            gamma=FOCAL_GAMMA,
            alpha=FOCAL_ALPHA,
            reduction='mean'
        )
        if hasattr(criterion, 'to'):
            criterion = criterion.to(device)
    else:
        print("📝 使用标准交叉熵损失")
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # 创建早停策略
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA, verbose=True)
    
    # 如果恢复训练，加载之前的模型
    start_epoch = 0
    best_acc = 0.0
    model_name = f"teacher_{base_model_name}"
    
    if resume_training:
        start_epoch, best_acc = load_model(model, optimizer, model_name)
        print(f"从第 {start_epoch} 轮恢复训练，最佳精度: {best_acc:.4f}")
        
        # 调整学习率调度器状态
        for _ in range(start_epoch):
            scheduler.step()
    
    # 记录训练和验证指标
    train_history = {
        'total_loss': [],
        'task_losses': {task: [] for task in task_weights.keys()},
        'acc': {task: [] for task in task_weights.keys()},
        'lr': []
    }
    
    val_history = {
        'total_loss': [],
        'task_losses': {task: [] for task in task_weights.keys()},
        'acc': {task: [] for task in task_weights.keys()}
    }
    
    # 训练循环
    print(f"\n{'='*20} 开始训练教师模型 {base_model_name} {'='*20}")
    training_start = time.time()
    
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            epoch_start = time.time()
            
            # 🔥 GPU预热和监控
            if epoch == start_epoch:
                print("🔥 预热GPU以优化卷积算法...")
                # 创建一个随机张量并通过模型传播
                dummy_input = torch.randn(actual_batch_size, 3, INPUT_SIZE, INPUT_SIZE, device=device)
                with torch.no_grad():
                    for task in task_weights.keys():
                        model(dummy_input, task=task)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # 确保完成
                print("✅ GPU预热完成")
            
            # 定期监控GPU使用情况
            if epoch % 5 == 0 and available_gpus:
                print(f"\n📊 Epoch {epoch} GPU监控:")
                if torch.cuda.is_available():
                    for gpu_id in available_gpus:
                        memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                        print(f"  GPU {gpu_id}: {memory_allocated:.2f}GB/{memory_reserved:.2f}GB")
            
            # 训练单个epoch
            train_loss, train_acc, train_task_losses = train_single_epoch(
                model, multi_task_data_loaders, task_weights, optimizer, criterion, 
                epoch, writer, amp_handler=amp_handler
            )
            # 强制内存清理 - 16GB内存优化
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 额外的Python垃圾回收
            import gc
            gc.collect()

            # 验证
            val_loss, val_acc, val_task_losses, overall_acc = validate(
                model, multi_task_data_loaders, task_weights, criterion, epoch, writer,
                amp_handler=amp_handler
            )
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录指标
            train_history['total_loss'].append(train_loss)
            val_history['total_loss'].append(val_loss)
            train_history['lr'].append(current_lr)
            
            for task in task_weights.keys():
                train_history['task_losses'][task].append(train_task_losses[task])
                val_history['task_losses'][task].append(val_task_losses[task])
                train_history['acc'][task].append(train_acc[task])
                val_history['acc'][task].append(val_acc[task])
            
            # 保存训练曲线
            save_detailed_curves(train_history, val_history, model_name, epoch)
            
            # 如果是最佳模型，保存模型
            if overall_acc > best_acc:
                best_acc = overall_acc
                save_model(model, optimizer, epoch, best_acc, model_name)
                print(f"保存最佳模型，精度: {best_acc:.4f}")
            
            # 检查早停条件
            if early_stopping.check(-overall_acc):  # 使用负准确率使其与最小化损失兼容
                print(f"早停触发于第 {epoch+1} 轮")
                break
            
            # 计算本轮用时
            epoch_time = time.time() - epoch_start
            # 估计剩余时间
            remaining_epochs = NUM_EPOCHS - epoch - 1
            remaining_time = epoch_time * remaining_epochs
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} 用时: {epoch_time:.2f}秒 | 估计剩余时间: {remaining_time/60:.2f}分钟")
            print("=" * 80)
            
    except KeyboardInterrupt:
        print("\n训练被中断!")
    
    # 计算总训练时间
    total_time = time.time() - training_start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"训练完成! 总用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    

    # 过滤掉训练准确率中的 None 值并计算平均值
    valid_train_acc = [
        train_history['acc'][task][-1]
        for task in task_weights.keys()
        if train_history['acc'].get(task) and train_history['acc'][task] and train_history['acc'][task][-1] is not None
    ]
    avg_train_acc = sum(valid_train_acc) / len(valid_train_acc) if valid_train_acc else 0.0
    # 过滤掉验证准确率中的 None 值并计算平均值
    valid_val_acc = [
        val_history['acc'][task][-1]
        for task in task_weights.keys()
        if val_history['acc'].get(task) and val_history['acc'][task] and val_history['acc'][task][-1] is not None
    ]
    avg_val_acc = sum(valid_val_acc) / len(valid_val_acc) if valid_val_acc else 0.0

    # 保存最终训练曲线
    save_detailed_curves(train_history, val_history, model_name, NUM_EPOCHS-1)

    # 保存标准训练曲线
    save_training_curve(
        train_history['total_loss'], 
        val_history['total_loss'], 
        [avg_train_acc],  # 替换为计算后的平均训练准确率
        [avg_val_acc],    # 替换为计算后的平均验证准确率
        model_name
    )

    # 清除缓存，防止内存泄漏
    torch.cuda.empty_cache()
    
    return model, multi_task_data_loaders


def train_student_model(teacher_model, base_model_name='mobilenetv3_small', use_amp=USE_AMP):
    """训练学生模型 (知识蒸馏)"""
    # 设置随机种子
    set_seed(SEED)
    
    # 创建AMP处理器
    amp_handler = AmpHandler(
        enabled=use_amp,
        init_scale=2**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=1000,
        max_scale=2**20
    )
    
    # 创建结果目录
    # 创建结果目录
    log_dir = os.path.join(RESULT_PATH, "logs", f"student_{base_model_name}")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(log_dir)
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 加载数据
    print("正在加载数据...")
    data_loaders = create_multi_task_loaders(
        COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, UGI_DISEASE_PATH
    )        
    # 动态计算类别权重
    class_weights = self.calculate_class_weights(targets, num_classes)
    class_weights = class_weights.to(inputs.device)
    num_colon_classes = len(data_loaders['colon']['classes'])
    num_ugi_classes = len(data_loaders['ugi']['classes'])
    num_colon_disease_classes = len(data_loaders['colon_disease']['classes'])
    num_ugi_disease_classes = len(data_loaders['ugi_disease']['classes'])

    print(f"数据加载完成! 用时: {time.time() - start_time:.2f}秒")
    
    # 任务权重 - 使用与教师模型相同的策略
    task_weights = {
        'colon': 1.0,
        'ugi': 1.0,
        'colon_disease': 1.0,
        'ugi_disease': 1.0
    }
    
    # 根据各任务的样本数调整权重
    task_sample_counts = {}
    for task in task_weights.keys():
        train_loader = multi_task_data_loaders[task]['train_loader']
        # 获取样本数
        if hasattr(train_loader.dataset, 'samples'):
            task_sample_counts[task] = len(train_loader.dataset.samples)
        else:
            task_sample_counts[task] = len(train_loader.dataset)
    
    # 样本数少的任务给予更高权重
    if TASK_WEIGHT_STRATEGY == 'dynamic':
        total_samples = sum(task_sample_counts.values())
        for task in task_weights.keys():
            # 使用样本数的平方根倒数作为基础权重
            task_weights[task] = np.sqrt(total_samples / task_sample_counts[task])
        
        # 归一化权重
        weight_sum = sum(task_weights.values())
        for task in task_weights.keys():
            task_weights[task] = task_weights[task] / weight_sum * len(task_weights)
    start
    print(f"初始任务权重: {task_weights}")
    
    # 创建学生模型
    print(f"创建学生模型 (轻量级模型: {base_model_name})...")
    student_model = StudentModel(
        base_model_name=base_model_name,
        num_colon_classes=num_colon_classes,
        num_ugi_classes=num_ugi_classes,
        num_colon_disease_classes=num_colon_disease_classes,
        num_ugi_disease_classes=num_ugi_disease_classes
    ).to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"✅ 使用 {torch.cuda.device_count()} 张 GPU")
        # 确保GPU ID有效
        available_gpus = list(range(torch.cuda.device_count()))
        valid_gpu_ids = [gpu_id for gpu_id in GPU_IDS if gpu_id in available_gpus]
        if not valid_gpu_ids:
            valid_gpu_ids = available_gpus
            print(f"警告: 指定的GPU ID无效，使用所有可用GPU: {valid_gpu_ids}")
        
        student_model = nn.DataParallel(student_model, device_ids=valid_gpu_ids)

    # 确保教师模型处于评估模式
    teacher_model.eval()
    if torch.cuda.device_count() > 1 and not isinstance(teacher_model, nn.DataParallel):
        teacher_model = nn.DataParallel(teacher_model, device_ids=valid_gpu_ids)
    

    # 统计模型参数
    teacher_params = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
    student_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    compression_ratio = teacher_params / student_params
    
    print(f"教师模型参数: {teacher_params:,}")
    print(f"学生模型参数: {student_params:,}")
    print(f"压缩比例: {compression_ratio:.2f}x")
    
    # 创建优化器和学习率调度器
    optimizer = optim.AdamW(
        student_model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 使用余弦退火学习率调度器，带热重启
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # 首次重启的周期
        T_mult=2,  # 每次重启后周期长度的倍增因子
        eta_min=LEARNING_RATE * LR_MIN_FACTOR  # 最小学习率
    )
    
    # 定义蒸馏损失函数
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    
    # 创建早停策略
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA, verbose=True)
    
    # 记录训练和验证指标
    train_history = {
        'total_loss': [],
        'task_losses': {task: [] for task in task_weights.keys()},
        'hard_losses': {task: [] for task in task_weights.keys()},
        'soft_losses': {task: [] for task in task_weights.keys()},
        'acc': {task: [] for task in task_weights.keys()},
        'lr': []
    }
    
    val_history = {
        'total_loss': [],
        'task_losses': {task: [] for task in task_weights.keys()},
        'acc': {task: [] for task in task_weights.keys()}
    }
    
    # 训练循环
    print(f"\n{'='*20} 开始训练学生模型 {base_model_name} {'='*20}")
    training_start = time.time()
    best_acc = 0.0
    model_name = f"student_{base_model_name}"
    
    try:
        for epoch in range(NUM_EPOCHS):
            epoch_start = time.time()
            
            # 在训练前预热GPU (对卷积层进行一次前向传播，使CUDNN优化器找到最佳算法)
            if epoch == 0:
                print("预热GPU以优化卷积算法...")
                # 创建一个随机张量并通过模型传递
                dummy_input = torch.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)
                with torch.no_grad():
                    for task in task_weights.keys():
                        student_model(dummy_input, task=task)
                torch.cuda.synchronize()  # 确保完成
                print("GPU预热完成")
                
            # 学习率预热
            if epoch < LR_WARMUP_EPOCHS:
                lr_scale = min(1.0, (epoch + 1) / LR_WARMUP_EPOCHS)
                for pg in optimizer.param_groups:
                    pg['lr'] = LEARNING_RATE * lr_scale
            
            # 训练
            student_model.train()
            teacher_model.eval()
            
            total_loss = 0.0
            hard_losses = {task: 0.0 for task in task_weights.keys()}
            soft_losses = {task: 0.0 for task in task_weights.keys()}
            task_losses = {task: 0.0 for task in task_weights.keys()}
            total_acc = {task: 0.0 for task in task_weights.keys()}
            samples_count = {task: 0 for task in task_weights.keys()}
            
            # 计算最长的数据加载器长度
            max_len = max([len(data_loaders[task]['train_loader']) for task in data_loaders.keys()])
            
            # 创建进度条
            pbar = ProgressBar(max_len, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train Student]")
            
            # 重置数据迭代器
            iterators = {task: iter(data_loaders[task]['train_loader']) for task in data_loaders.keys()}
            
            # 批处理积累，用于更高效的梯度累积
            accumulation_steps = 1  # 可以调整为2或4以减少同步开销
            optimizer.zero_grad()
            
            for i in range(max_len):
                # 为每个任务准备批次数据
                batch_loss = 0.0
                batch_task_losses = {}
                batch_hard_losses = {}
                batch_soft_losses = {}
                
                for task, task_weight in task_weights.items():
                    try:
                        inputs, labels = next(iterators[task])
                    except StopIteration:
                        # 如果某个数据集耗尽，重新初始化迭代器
                        iterators[task] = iter(data_loaders[task]['train_loader'])
                        inputs, labels = next(iterators[task])
                    
                    # 使用非阻塞传输加速数据移动
                    inputs = inputs.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    
                    # 使用混合精度训练
                    with amp_handler.autocast() if amp_handler else torch.cuda.amp.autocast(enabled=False):
                        # 教师模型前向传播 (不计算梯度)
                        with torch.no_grad():
                            teacher_outputs, _ = teacher_model(inputs, task=task)
                            teacher_probs = F.softmax(teacher_outputs / DISTILLATION_TEMP, dim=1)
                        
                        # 学生模型前向传播
                        student_outputs = student_model(inputs, task=task)
                        student_log_probs = F.log_softmax(student_outputs / DISTILLATION_TEMP, dim=1)
                        
                        # 计算硬标签损失
                        hard_loss = ce_criterion(student_outputs, labels)
                        batch_hard_losses[task] = hard_loss.item()
                        hard_losses[task] += batch_hard_losses[task]
                        
                        # 计算软标签损失 (KL散度)
                        soft_loss = kl_criterion(student_log_probs, teacher_probs) * (DISTILLATION_TEMP ** 2)
                        batch_soft_losses[task] = soft_loss.item()
                        soft_losses[task] += batch_soft_losses[task]
                        
                        # 计算总损失
                        distill_loss = ALPHA * hard_loss + (1 - ALPHA) * soft_loss
                        task_loss = task_weight * distill_loss / accumulation_steps
                        
                        # 记录任务损失
                        batch_task_losses[task] = distill_loss.item()
                        task_losses[task] += batch_task_losses[task]
                        
                        # 累积任务损失
                        batch_loss += task_loss
                    
                    # 计算准确率
                    _, predicted = torch.max(student_outputs.data, 1)
                    batch_correct = (predicted == labels).sum().item()
                    batch_total = labels.size(0)
                    batch_acc = batch_correct / batch_total * 100.0
                    
                    total_acc[task] += batch_correct
                    samples_count[task] += batch_total
                
                # 反向传播和优化（使用梯度累积）
                if amp_handler:
                    # 使用混合精度训练
                    amp_handler.scale_loss(batch_loss).backward()
                    
                    # 仅在累积步骤完成后更新
                    if (i + 1) % accumulation_steps == 0:
                        # 梯度裁剪 - 处理混合精度情况
                        if hasattr(amp_handler.scaler, "_enabled") and amp_handler.scaler._enabled:
                            amp_handler.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                        amp_handler.step(optimizer)
                        amp_handler.update()
                        optimizer.zero_grad()
                else:
                    # 标准训练
                    batch_loss.backward()
                    
                    # 仅在累积步骤完成后更新
                    if (i + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                # 累积总损失
                total_loss += batch_loss.item() * accumulation_steps
                
                # 更新进度条描述
                if i % 10 == 0:  # 每10个批次更新一次显示
                    task_desc = " | ".join([f"{task}: {batch_task_losses[task]:.3f}" for task in batch_task_losses.keys()])
                    pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} [Train Student] Loss: {batch_loss.item()*accumulation_steps:.3f} | {task_desc}")
                
                # 更新进度条
                pbar.update()
            
            # 确保最后一批的梯度也被应用
            if (max_len % accumulation_steps) != 0:
                if amp_handler:
                    if hasattr(amp_handler.scaler, "_enabled") and amp_handler.scaler._enabled:
                        amp_handler.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                    amp_handler.step(optimizer)
                    amp_handler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                    optimizer.step()
            
            pbar.close()
            
            # 计算平均损失和准确率
            avg_loss = total_loss / max_len
            avg_task_losses = {task: task_losses[task] / max_len for task in task_losses.keys()}
            avg_hard_losses = {task: hard_losses[task] / max_len for task in task_losses.keys()}
            avg_soft_losses = {task: soft_losses[task] / max_len for task in task_losses.keys()}
            avg_acc = {task: total_acc[task] / samples_count[task] * 100.0 for task in data_loaders.keys()}
            
            # 记录训练指标
            writer.add_scalar('Loss/train/total', avg_loss, epoch)
            for task in avg_task_losses.keys():
                writer.add_scalar(f'Loss/train/{task}', avg_task_losses[task], epoch)
                writer.add_scalar(f'Loss/train/{task}_hard', avg_hard_losses[task], epoch)
                writer.add_scalar(f'Loss/train/{task}_soft', avg_soft_losses[task], epoch)
                writer.add_scalar(f'Accuracy/train/{task}', avg_acc[task], epoch)
            
            # 打印训练结果
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} 学生训练结果:")
            print(f"总损失: {avg_loss:.4f}")
            for task in avg_task_losses.keys():
                print(f"{task} 损失: {avg_task_losses[task]:.4f} (硬: {avg_hard_losses[task]:.4f} | 软: {avg_soft_losses[task]:.4f}) | 准确率: {avg_acc[task]:.2f}%")
            
            # 验证
            student_model.eval()
            val_loss = 0.0
            val_task_losses = {task: 0.0 for task in task_weights.keys()}
            val_acc = {task: 0.0 for task in task_weights.keys()}
            val_samples_count = {task: 0 for task in task_weights.keys()}
            
            max_val_len = max([len(data_loaders[task]['test_loader']) for task in data_loaders.keys()])
            pbar = ProgressBar(max_val_len, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val Student]")
            
            with torch.no_grad():
                # 重置数据迭代器
                val_iterators = {task: iter(data_loaders[task]['test_loader']) for task in data_loaders.keys()}
                
                for i in range(max_val_len):
                    # 为每个任务准备批次数据
                    batch_val_loss = 0.0
                    batch_val_task_losses = {}
                    
                    for task, task_weight in task_weights.items():
                        try:
                            inputs, labels = next(val_iterators[task])
                        except StopIteration:
                            # 如果某个数据集耗尽，重新初始化迭代器
                            val_iterators[task] = iter(data_loaders[task]['test_loader'])
                            inputs, labels = next(val_iterators[task])
                        
                        # 使用非阻塞传输加速数据移动
                        inputs = inputs.to(DEVICE, non_blocking=True)
                        labels = labels.to(DEVICE, non_blocking=True)
                        
                        # 使用混合精度
                        with amp_handler.autocast() if amp_handler else torch.cuda.amp.autocast(enabled=False):
                            # 学生模型前向传播
                            student_outputs = student_model(inputs, task=task)
                            loss = ce_criterion(student_outputs, labels)
                        
                        # 记录任务损失
                        batch_val_task_losses[task] = loss.item()
                        val_task_losses[task] += batch_val_task_losses[task]
                        
                        # 累积任务损失
                        batch_val_loss += task_weight * loss
                        
                        # 计算准确率
                        _, predicted = torch.max(student_outputs.data, 1)
                        batch_correct = (predicted == labels).sum().item()
                        batch_total = labels.size(0)
                        batch_acc = batch_correct / batch_total * 100.0
                        
                        val_acc[task] += batch_correct
                        val_samples_count[task] += batch_total
                    
                    # 累积总损失
                    val_loss += batch_val_loss.item()
                    
                    # 更新进度条描述
                    if i % 10 == 0:  # 每10个批次更新一次显示
                        task_desc = " | ".join([f"{task}: {batch_val_task_losses[task]:.3f}" for task in batch_val_task_losses.keys()])
                        pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} [Val Student] Loss: {batch_val_loss.item():.3f} | {task_desc}")
                    
                    # 更新进度条
                    pbar.update()
            
            pbar.close()
            
            # 计算平均验证损失和准确率
            avg_val_loss = val_loss / max_val_len
            avg_val_task_losses = {task: val_task_losses[task] / max_val_len for task in task_weights.keys()}
            avg_val_acc = {task: val_acc[task] / val_samples_count[task] * 100.0 for task in data_loaders.keys()}
            
            # 记录验证指标
            writer.add_scalar('Loss/val/total', avg_val_loss, epoch)
            for task in avg_val_task_losses.keys():
                writer.add_scalar(f'Loss/val/{task}', avg_val_task_losses[task], epoch)
                writer.add_scalar(f'Accuracy/val/{task}', avg_val_acc[task], epoch)
            
            # 打印验证结果
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} 学生验证结果:")
            print(f"总损失: {avg_val_loss:.4f}")
            for task in avg_val_task_losses.keys():
                print(f"{task} 损失: {avg_val_task_losses[task]:.4f} | 准确率: {avg_val_acc[task]:.2f}%")
            
            # 更新学习率
            if epoch >= LR_WARMUP_EPOCHS:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录指标
            train_history['total_loss'].append(avg_loss)
            val_history['total_loss'].append(avg_val_loss)
            train_history['lr'].append(current_lr)
            
            for task in task_weights.keys():
                train_history['task_losses'][task].append(avg_task_losses[task])
                train_history['hard_losses'][task].append(avg_hard_losses[task])
                train_history['soft_losses'][task].append(avg_soft_losses[task])
                val_history['task_losses'][task].append(avg_val_task_losses[task])
                train_history['acc'][task].append(avg_acc[task])
                val_history['acc'][task].append(avg_val_acc[task])
            
            # 保存训练曲线
            save_detailed_curves(train_history, val_history, model_name, epoch)
            
            # 计算整体验证准确率
            overall_val_acc = sum(avg_val_acc.values()) / len(avg_val_acc)
            
            # 如果是最佳模型，保存模型
            if overall_val_acc > best_acc:
                best_acc = overall_val_acc
                save_model(student_model, optimizer, epoch, best_acc, model_name)
                print(f"保存最佳学生模型，精度: {best_acc:.4f}")
            
            # 检查早停条件
            if early_stopping.check(-overall_val_acc):  # 使用负准确率使其与最小化损失兼容
                print(f"早停触发于第 {epoch+1} 轮")
                break
            
            # 计算本轮用时
            epoch_time = time.time() - epoch_start
            # 估计剩余时间
            remaining_epochs = NUM_EPOCHS - epoch - 1
            remaining_time = epoch_time * remaining_epochs
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} 用时: {epoch_time:.2f}秒 | 估计剩余时间: {remaining_time/60:.2f}分钟")
            print("=" * 80)
    
    except KeyboardInterrupt:
        print("\n训练被中断!")
    
    # 计算总训练时间
    total_time = time.time() - training_start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"训练完成! 总用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    
    # 保存最终训练曲线
    save_detailed_curves(train_history, val_history, model_name, NUM_EPOCHS-1)


    # 过滤掉训练准确率中的 None 值并计算平均值
    valid_train_acc = [
        train_history['acc'][task][-1]
        for task in task_weights.keys()
        if train_history['acc'].get(task) and train_history['acc'][task] and train_history['acc'][task][-1] is not None
    ]
    avg_train_acc = sum(valid_train_acc) / len(valid_train_acc) if valid_train_acc else 0.0
    # 过滤掉验证准确率中的 None 值并计算平均值
    valid_val_acc = [
        val_history['acc'][task][-1]
        for task in task_weights.keys()
        if val_history['acc'].get(task) and val_history['acc'][task] and val_history['acc'][task][-1] is not None
    ]
    avg_val_acc = sum(valid_val_acc) / len(valid_val_acc) if valid_val_acc else 0.0

    # 保存最终训练曲线
    save_detailed_curves(train_history, val_history, model_name, NUM_EPOCHS-1)

    # 保存标准训练曲线
    save_training_curve(
        train_history['total_loss'], 
        val_history['total_loss'], 
        [avg_train_acc],  # 替换为计算后的平均训练准确率
        [avg_val_acc],    # 替换为计算后的平均验证准确率
        model_name
    )
    
    # 清除缓存，防止内存泄漏
    torch.cuda.empty_cache()
    
    return student_model