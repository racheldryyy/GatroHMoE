import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE

from config import (
    DEVICE, RESULT_PATH, COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, 
    UGI_DISEASE_PATH, HETERO_ARCHITECTURES, GPU_IDS, USE_AMP
)
from data_loader import create_multi_task_loaders
from models.moe_model import MixtureOfExperts
from models.student_model import StudentModel
from models.hetero_moe_model import HeterogeneousMixtureOfExperts
from models.rl_hetero_moe_model import RLHeterogeneousMixtureOfExperts
from evaluate import evaluate_model
from utils import load_model, ProgressBar, set_seed
from amp_utils import AmpHandler

# 定义莫兰迪冷色系
MORANDI_COLD_COLORS = [
    "#8CAAB9",  # 灰蓝色
    "#B4C4CA",  # 淡蓝灰色
    "#7C9EA8",  # 孔雀蓝
    "#A5B8C0",  # 淡蓝色
    "#A6AEB3",  # 铅灰色
    "#8D9DAB",  # 蓝灰色
    "#B1B8BE",  # 灰色
    "#728C96",  # 深青灰色
]

# 测试模型的性能指标和各种特性
def comprehensive_model_evaluation(model_types, architectures=None, use_amp=USE_AMP, seed=42):
    """
    综合评估所有指定模型的性能
    
    参数:
        model_types (list): 要评估的模型类型列表
        architectures (list): 异构模型使用的架构
        use_amp (bool): 是否使用混合精度
        seed (int): 随机种子
    """
    # 设置随机种子确保结果可复现
    set_seed(seed)
    
    # 创建结果目录
    evaluation_dir = os.path.join(RESULT_PATH, "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    
    # 配置AMP处理器
    amp_handler = AmpHandler(enabled=use_amp)
    
    # 加载数据
    print("正在加载数据集...")
    data_loaders = create_multi_task_loaders(
        COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, UGI_DISEASE_PATH
    )
    
    # 获取类别数量
    num_colon_classes = len(data_loaders['colon']['classes'])
    num_ugi_classes = len(data_loaders['ugi']['classes'])
    num_colon_disease_classes = len(data_loaders['colon_disease']['classes'])
    num_ugi_disease_classes = len(data_loaders['ugi_disease']['classes'])
    
    # 初始化结果存储
    results = {}
    model_weights = {}
    inference_times = {}
    parameter_counts = {}
    
    # 评估每个模型
    for model_type in model_types:
        print(f"\n{'='*20} 评估模型: {model_type} {'='*20}")
        
        # 初始化模型
        if model_type.startswith('teacher_'):
            base_model_name = model_type.split('_')[1]
            model = MixtureOfExperts(
                base_model_name=base_model_name,
                num_colon_classes=num_colon_classes,
                num_ugi_classes=num_ugi_classes,
                num_colon_disease_classes=num_colon_disease_classes,
                num_ugi_disease_classes=num_ugi_disease_classes
            ).to(DEVICE)
            
        elif model_type.startswith('student_'):
            base_model_name = model_type.split('_')[1]
            model = StudentModel(
                base_model_name=base_model_name,
                num_colon_classes=num_colon_classes,
                num_ugi_classes=num_ugi_classes,
                num_colon_disease_classes=num_colon_disease_classes,
                num_ugi_disease_classes=num_ugi_disease_classes
            ).to(DEVICE)
            
        elif model_type == 'hetero_moe':
            model = HeterogeneousMixtureOfExperts(
                model_names=architectures if architectures else HETERO_ARCHITECTURES,
                num_colon_classes=num_colon_classes,
                num_ugi_classes=num_ugi_classes,
                num_colon_disease_classes=num_colon_disease_classes,
                num_ugi_disease_classes=num_ugi_disease_classes
            ).to(DEVICE)
            
        elif model_type == 'rl_hetero_moe':
            model = RLHeterogeneousMixtureOfExperts(
                model_names=architectures if architectures else HETERO_ARCHITECTURES,
                num_colon_classes=num_colon_classes,
                num_ugi_classes=num_ugi_classes,
                num_colon_disease_classes=num_colon_disease_classes,
                num_ugi_disease_classes=num_ugi_disease_classes
            ).to(DEVICE)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        # 多GPU支持
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 张 GPU 进行评估")
            model = nn.DataParallel(model, device_ids=GPU_IDS)
            
        # 加载预训练权重
        optimizer = torch.optim.AdamW(model.parameters())
        _, _ = load_model(model, optimizer, model_type)
        
        # 统计模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        parameter_counts[model_type] = {
            'total': total_params,
            'trainable': trainable_params
        }
        
        # 测量模型大小（MB）
        model_size = 0
        for param in model.parameters():
            model_size += param.nelement() * param.element_size()
        model_size_mb = model_size / (1024 * 1024)
        model_weights[model_type] = model_size_mb
        
        # 评估模型性能
        model.eval()
        start_time = time.time()
        eval_results = evaluate_model(model, data_loaders, model_type)
        eval_time = time.time() - start_time
        
        # 测量推理时间
        inference_times[model_type] = measure_inference_time(model, data_loaders)
        
        # 保存评估结果
        results[model_type] = eval_results
        
        # 为异构模型执行额外的专家分析
        if model_type in ['hetero_moe', 'rl_hetero_moe']:
            print(f"分析{model_type}的专家贡献...")
            analyze_experts_contribution(model, data_loaders, model_type)
            
        # 记录每个任务的最佳类别和失败案例
        for task, task_data in eval_results.items():
            # 找出性能最好和最差的类别
            analyze_class_performance(task_data, task, model_type)
            
            # 分析失败案例
            analyze_failure_cases(task_data, task, model_type, 
                                  data_loaders[task]['classes'],
                                  data_loaders[task]['test_loader'])
            
        # 测量推理速度：速度与批量大小的关系
        analyze_inference_scaling(model, model_type)
        
        # 可视化模型的特征空间
        visualize_feature_space(model, data_loaders, model_type)
        
        # 清除缓存，防止内存泄漏
        torch.cuda.empty_cache()
        
    # 模型比较和综合分析
    print("\n正在进行模型综合比较分析...")
    
    # 准备比较数据
    comparison_data = prepare_comparison_data(results, model_weights, 
                                            inference_times, parameter_counts)
    
    # 创建综合比较报告
    create_comparison_report(comparison_data, model_types)
    
    # 创建任务间相关性分析
    analyze_task_correlations(results)
    
    # 创建难例分析（所有模型共同的错误案例）
    analyze_common_errors(results, data_loaders)
    
    # 创建决策边界可视化
    visualize_decision_boundaries(model_types, results)
    
    # 计算性能的显著性差异（统计检验）
    perform_statistical_tests(results)
    
    print("\n综合评估完成！详细结果已保存到评估目录。")
    
    return results, comparison_data

# 测量推理时间
def measure_inference_time(model, data_loaders, num_runs=10):
    """测量模型在不同任务上的平均推理时间"""
    inference_times = {}
    
    model.eval()
    with torch.no_grad():
        for task, task_data in data_loaders.items():
            test_loader = task_data['test_loader']
            
            # 获取一批数据用于测试
            inputs, _ = next(iter(test_loader))
            inputs = inputs.to(DEVICE)
            
            # 预热
            for _ in range(5):
                if hasattr(model, 'module'):
                    model.module(inputs, task=task)
                else:
                    model(inputs, task=task)
            
            # 测量时间
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_runs):
                if hasattr(model, 'module'):
                    model.module(inputs, task=task)
                else:
                    model(inputs, task=task)
                    
            torch.cuda.synchronize()
            end_time = time.time()
            
            # 计算平均每批次推理时间（毫秒）
            avg_time = (end_time - start_time) * 1000 / num_runs
            inference_times[task] = avg_time
            
    return inference_times

# 分析专家贡献
def analyze_experts_contribution(model, data_loaders, model_name):
    """分析异构模型中不同专家的贡献和特点"""
    if not hasattr(model, 'module'):
        module = model
    else:
        module = model.module
        
    if not hasattr(module, 'num_experts'):
        print("该模型不支持专家贡献分析")
        return
    
    model.eval()
    expert_usage = {task: np.zeros(module.num_experts) for task in data_loaders.keys()}
    expert_accuracy = {task: np.zeros(module.num_experts) for task in data_loaders.keys()}
    expert_counts = {task: np.zeros(module.num_experts) for task in data_loaders.keys()}
    
    with torch.no_grad():
        for task, task_data in data_loaders.items():
            test_loader = task_data['test_loader']
            pbar = ProgressBar(len(test_loader), desc=f"分析{task}任务的专家贡献")
            
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                # 获取每个输入样本最有贡献的专家
                outputs, routing_weights = model(inputs, task=task)
                
                # 获取TopK专家
                topk = module.top_k if hasattr(module, 'top_k') else 2
                _, top_indices = torch.topk(routing_weights, topk, dim=1)
                
                # 计算预测准确性
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels)
                
                # 更新专家使用情况
                for i in range(inputs.size(0)):
                    for k in range(topk):
                        expert_idx = top_indices[i, k].item()
                        expert_usage[task][expert_idx] += 1
                        expert_counts[task][expert_idx] += 1
                        if correct[i]:
                            expert_accuracy[task][expert_idx] += 1
                
                pbar.update()
            
            pbar.close()
    
    # 计算平均准确率
    for task in expert_accuracy:
        for i in range(module.num_experts):
            if expert_counts[task][i] > 0:
                expert_accuracy[task][i] /= expert_counts[task][i]
    
    # 可视化专家贡献
    plt.figure(figsize=(15, 10))
    for i, task in enumerate(data_loaders.keys()):
        plt.subplot(2, 2, i+1)
        
        # 定义专家标签
        if hasattr(module, 'model_names'):
            expert_labels = [f"Expert {j+1}: {name}" for j, name in enumerate(module.model_names)]
        else:
            expert_labels = [f"Expert {j+1}" for j in range(module.num_experts)]
        
        # 计算专家使用率
        total_usage = sum(expert_usage[task])
        if total_usage > 0:
            usage_ratios = [usage / total_usage for usage in expert_usage[task]]
        else:
            usage_ratios = [0] * len(expert_usage[task])
        
        # 创建堆叠柱状图数据
        x = np.arange(len(expert_labels))
        width = 0.35
        
        # 绘制使用率柱状图
        plt.bar(x - width/2, usage_ratios, width, 
                label='使用率', color=MORANDI_COLD_COLORS[0])
        
        # 绘制准确率柱状图
        plt.bar(x + width/2, expert_accuracy[task], width, 
                label='准确率', color=MORANDI_COLD_COLORS[1])
        
        plt.xlabel('专家模型')
        plt.ylabel('比例')
        plt.title(f'{task.capitalize()} 任务专家贡献分析')
        plt.xticks(x, expert_labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(RESULT_PATH, "evaluation", f"{model_name}_expert_contribution.png")
    plt.savefig(save_path)
    plt.close()
    
    # 保存专家贡献数据
    contribution_data = []
    for task in expert_usage.keys():
        for i in range(module.num_experts):
            expert_name = expert_labels[i] if i < len(expert_labels) else f"Expert {i+1}"
            contribution_data.append({
                'Task': task.capitalize(),
                'Expert': expert_name,
                'Usage Ratio': usage_ratios[i],
                'Accuracy': expert_accuracy[task][i]
            })
    
    df = pd.DataFrame(contribution_data)
    csv_path = os.path.join(RESULT_PATH, "evaluation", f"{model_name}_expert_contribution.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"专家贡献分析已保存至: {save_path} 和 {csv_path}")

# 分析类别性能
def analyze_class_performance(task_data, task, model_name):
    """分析模型在不同类别上的性能"""
    report = task_data['classification_report']
    classes = task_data['classes']
    
    # 提取每个类别的精确率、召回率和F1分数
    class_metrics = {}
    for i, cls in enumerate(classes):
        if cls in report:
            class_metrics[cls] = {
                'precision': report[cls]['precision'],
                'recall': report[cls]['recall'],
                'f1-score': report[cls]['f1-score'],
                'support': report[cls]['support']
            }
    
    # 转换为DataFrame
    df = pd.DataFrame(class_metrics).T
    df = df.sort_values('f1-score', ascending=False)
    
    # 保存为CSV
    csv_path = os.path.join(RESULT_PATH, "evaluation", f"{model_name}_{task}_class_performance.csv")
    df.to_csv(csv_path)
    
    # 创建类别性能可视化
    plt.figure(figsize=(12, 8))
    plt.title(f"{task.capitalize()} 任务各类别性能分析")
    
    # 绘制性能指标
    x = np.arange(len(df))
    width = 0.25
    
    plt.bar(x - width, df['precision'], width, label='精确率', color=MORANDI_COLD_COLORS[0])
    plt.bar(x, df['recall'], width, label='召回率', color=MORANDI_COLD_COLORS[1])
    plt.bar(x + width, df['f1-score'], width, label='F1分数', color=MORANDI_COLD_COLORS[2])
    
    plt.xlabel('类别')
    plt.ylabel('得分')
    plt.xticks(x, df.index, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(RESULT_PATH, "evaluation", f"{model_name}_{task}_class_performance.png")
    plt.savefig(save_path)
    plt.close()
    
    # 找出性能最好和最差的类别
    best_class = df.index[0]
    worst_class = df.index[-1]
    
    print(f"{task.capitalize()} 任务最佳类别: {best_class}, F1: {df.loc[best_class, 'f1-score']:.4f}")
    print(f"{task.capitalize()} 任务最差类别: {worst_class}, F1: {df.loc[worst_class, 'f1-score']:.4f}")
    
    return best_class, worst_class

# 分析失败案例
def analyze_failure_cases(task_data, task, model_name, classes, test_loader, max_cases=10):
    """分析并可视化模型预测失败的典型案例"""
    predictions = task_data['predictions']
    labels = task_data['labels']
    
    # 找出预测错误的样本
    error_indices = np.where(predictions != labels)[0]
    
    if len(error_indices) == 0:
        print(f"{task.capitalize()} 任务无预测错误案例")
        return
    
    # 准备失败案例数据
    failure_cases = []
    
    # 分析错误模式
    error_types = {}
    for i in error_indices:
        true_class = classes[labels[i]]
        pred_class = classes[predictions[i]]
        error_type = f"{true_class} → {pred_class}"
        
        if error_type not in error_types:
            error_types[error_type] = 0
        error_types[error_type] += 1
        
        # 记录失败案例详情
        failure_cases.append({
            'Index': i,
            'True Class': true_class,
            'Predicted Class': pred_class
        })
    
    # 转换为DataFrame并保存
    failure_df = pd.DataFrame(failure_cases)
    fail_path = os.path.join(RESULT_PATH, "evaluation", f"{model_name}_{task}_failure_cases.csv")
    failure_df.to_csv(fail_path, index=False)
    
    # 可视化错误类型分布
    error_df = pd.DataFrame({
        'Error Type': list(error_types.keys()),
        'Count': list(error_types.values())
    })
    error_df = error_df.sort_values('Count', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Count', y='Error Type', data=error_df.head(20), 
                palette=sns.color_palette(MORANDI_COLD_COLORS))
    plt.title(f"{task.capitalize()} 任务主要错误类型")
    plt.tight_layout()
    
    # 保存错误类型分析图表
    error_path = os.path.join(RESULT_PATH, "evaluation", f"{model_name}_{task}_error_types.png")
    plt.savefig(error_path)
    plt.close()
    
    print(f"{task.capitalize()} 任务失败案例分析已保存至: {fail_path} 和 {error_path}")

# 分析推理扩展性
def analyze_inference_scaling(model, model_name):
    """分析模型推理时间与批量大小的关系"""
    model.eval()
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    times = []
    
    # 创建随机输入数据
    with torch.no_grad():
        for batch_size in batch_sizes:
            # 检查批量大小是否超出GPU内存限制
            try:
                inputs = torch.randn(batch_size, 3, 224, 224).to(DEVICE)
                
                # 预热
                for _ in range(5):
                    if hasattr(model, 'module'):
                        model.module(inputs, task='colon')
                    else:
                        model(inputs, task='colon')
                
                # 测量时间
                torch.cuda.synchronize()
                start_time = time.time()
                
                runs = 10
                for _ in range(runs):
                    if hasattr(model, 'module'):
                        model.module(inputs, task='colon')
                    else:
                        model(inputs, task='colon')
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                # 计算每样本平均推理时间（毫秒）
                avg_time_per_sample = (end_time - start_time) * 1000 / (runs * batch_size)
                times.append(avg_time_per_sample)
                
                print(f"批量大小 {batch_size}: 每样本 {avg_time_per_sample:.2f} ms")
                
            except RuntimeError as e:
                print(f"批量大小 {batch_size} 超出GPU内存限制")
                break
    
    # 绘制批量大小与推理时间的关系图
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes[:len(times)], times, marker='o', 
             color=MORANDI_COLD_COLORS[0], linewidth=2)
    plt.title(f"{model_name} 推理扩展性分析")
    plt.xlabel('批量大小')
    plt.ylabel('每样本推理时间 (ms)')
    plt.xscale('log', base=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(RESULT_PATH, "evaluation", f"{model_name}_inference_scaling.png")
    plt.savefig(save_path)
    plt.close()
    
    # 保存数据
    scaling_df = pd.DataFrame({
        'Batch Size': batch_sizes[:len(times)],
        'Time per Sample (ms)': times
    })
    csv_path = os.path.join(RESULT_PATH, "evaluation", f"{model_name}_inference_scaling.csv")
    scaling_df.to_csv(csv_path, index=False)
    
    print(f"推理扩展性分析已保存至: {save_path}")

# 可视化特征空间
def visualize_feature_space(model, data_loaders, model_name, task='colon', n_samples=500):
    """使用t-SNE可视化模型的特征空间"""
    model.eval()
    
    # 收集特征和标签
    features = []
    labels = []
    
    with torch.no_grad():
        test_loader = data_loaders[task]['test_loader']
        count = 0
        
        for inputs, targets in test_loader:
            if count >= n_samples:
                break
                
            inputs = inputs.to(DEVICE)
            
            # 获取特征表示
            if hasattr(model, 'module'):
                if isinstance(model.module, MixtureOfExperts) or isinstance(model.module, HeterogeneousMixtureOfExperts):
                    _, feature_vectors = model.module.encoder(inputs) if hasattr(model.module, 'encoder') else model(inputs, task=task)
                else:
                    # 学生模型或其他模型
                    if hasattr(model.module, 'encoder'):
                        feature_vectors = model.module.encoder(inputs)
                    else:
                        # 特别处理
                        feature_vectors = model(inputs, task=task)
            else:
                if isinstance(model, MixtureOfExperts) or isinstance(model, HeterogeneousMixtureOfExperts):
                    _, feature_vectors = model.encoder(inputs) if hasattr(model, 'encoder') else model(inputs, task=task)
                else:
                    # 学生模型或其他模型
                    if hasattr(model, 'encoder'):
                        feature_vectors = model.encoder(inputs)
                    else:
                        # 特别处理
                        feature_vectors = model(inputs, task=task)
            
            # 确保特征向量是2D张量
            if isinstance(feature_vectors, tuple):
                feature_vectors = feature_vectors[1]  # 使用第二个返回值作为特征
            
            # 处理DataParallel的输出
            if isinstance(feature_vectors, list):
                feature_vectors = feature_vectors[0]
            
            # 将特征向量展平
            if feature_vectors.dim() > 2:
                feature_vectors = feature_vectors.view(feature_vectors.size(0), -1)
            
            # 收集特征和标签
            features.append(feature_vectors.cpu().numpy())
            labels.append(targets.numpy())
            
            count += inputs.size(0)
    
    # 确保我们有足够的样本
    if not features:
        print(f"无法为{model_name}获取特征表示")
        return
        
    # 合并数据
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    # 使用t-SNE降维
    print(f"正在进行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # 可视化
    plt.figure(figsize=(12, 10))
    classes = data_loaders[task]['classes']
    
    # 为每个类别着色
    for i, cls in enumerate(np.unique(labels)):
        plt.scatter(
            features_2d[labels == cls, 0], 
            features_2d[labels == cls, 1],
            label=classes[cls] if cls < len(classes) else f"Class {cls}",
            color=MORANDI_COLD_COLORS[i % len(MORANDI_COLD_COLORS)],
            alpha=0.7
        )
    
    plt.title(f"{model_name} {task.capitalize()} 任务特征空间可视化")
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(RESULT_PATH, "evaluation", f"{model_name}_{task}_feature_space.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"特征空间可视化已保存至: {save_path}")

# 准备模型比较数据
def prepare_comparison_data(results, model_weights, inference_times, parameter_counts):
    """准备用于模型比较的数据"""
    comparison = {
        'Model': [],
        'Accuracy': [],
        'F1-Score': [],
        'Size (MB)': [],
        'Parameters': [],
        'Inference Time (ms)': [],
        'Task': []
    }
    
    for model_name, model_results in results.items():
        for task, task_data in model_results.items():
            report = task_data['classification_report']
            
            comparison['Model'].append(model_name)
            comparison['Task'].append(task)
            comparison['Accuracy'].append(report['accuracy'])
            comparison['F1-Score'].append(report['weighted avg']['f1-score'])
            comparison['Size (MB)'].append(model_weights[model_name])
            comparison['Parameters'].append(parameter_counts[model_name]['total'])
            
            # 计算平均推理时间
            if model_name in inference_times and task in inference_times[model_name]:
                comparison['Inference Time (ms)'].append(inference_times[model_name][task])
            else:
                comparison['Inference Time (ms)'].append(float('nan'))
    
    return pd.DataFrame(comparison)

# 创建比较报告
def create_comparison_report(comparison_data, model_types):
    """创建模型综合比较报告"""
    # 任务间平均性能
    avg_performance = comparison_data.groupby(['Model'])[['Accuracy', 'F1-Score']].mean().reset_index()
    
    # 计算效率指标
    efficiency = comparison_data.groupby(['Model'])[['Size (MB)', 'Parameters', 'Inference Time (ms)']].mean().reset_index()
    
    # 合并性能和效率数据
    comparison_summary = pd.merge(avg_performance, efficiency, on='Model')
    
    # 计算综合得分 (可根据需要调整权重)
    # 标准化各指标
    normalized = comparison_summary.copy()
    for col in ['Accuracy', 'F1-Score']:
        normalized[col] = normalized[col] / normalized[col].max()
    
    for col in ['Size (MB)', 'Parameters', 'Inference Time (ms)']:
        if normalized[col].min() > 0:
            normalized[col] = normalized[col].min() / normalized[col]
        else:
            normalized[col] = 1.0
            
    # 计算综合得分
    normalized['Overall Score'] = (
        normalized['Accuracy'] * 0.3 + 
        normalized['F1-Score'] * 0.3 + 
        normalized['Size (MB)'] * 0.15 + 
        normalized['Parameters'] * 0.1 + 
        normalized['Inference Time (ms)'] * 0.15
    )
    
    # 合并原始值和综合得分
    comparison_summary['Overall Score'] = normalized['Overall Score']
    comparison_summary = comparison_summary.sort_values('Overall Score', ascending=False)
    
    # 保存综合比较报告
    csv_path = os.path.join(RESULT_PATH, "evaluation", "model_comparison_summary.csv")
    comparison_summary.to_csv(csv_path, index=False)
    
    # 创建雷达图比较
    create_radar_comparison(comparison_summary, model_types)
    
    # 创建任务特定比较图
    create_task_specific_comparison(comparison_data, model_types)
    
    # 创建性能与效率权衡图
    create_performance_efficiency_tradeoff(comparison_summary)
    
    print(f"模型比较报告已保存至: {csv_path}")
    
# 创建雷达图比较
def create_radar_comparison(comparison_summary, model_types):
    """创建模型性能雷达图比较"""
    # 准备雷达图数据
    metrics = ['Accuracy', 'F1-Score', 'Size (MB)', 'Inference Time (ms)', 'Parameters']
    
    # 标准化数据 (0-1范围)
    radar_data = comparison_summary.copy()
    for col in metrics:
        if col in ['Accuracy', 'F1-Score']:
            radar_data[col] = radar_data[col] / radar_data[col].max()
        else:
            if radar_data[col].min() > 0:
                radar_data[col] = radar_data[col].min() / radar_data[col]
            else:
                radar_data[col] = 1.0
    
    # 设置雷达图参数
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 创建雷达图
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, polar=True)
    
    # 绘制每个模型的雷达图
    for i, model in enumerate(radar_data['Model']):
        values = radar_data.loc[radar_data['Model'] == model, metrics].values.flatten().tolist()
        values += values[:1]  # 闭合图形
        
        ax.plot(angles, values, 
                color=MORANDI_COLD_COLORS[i % len(MORANDI_COLD_COLORS)], 
                linewidth=2, 
                label=model)
        ax.fill(angles, values, 
                color=MORANDI_COLD_COLORS[i % len(MORANDI_COLD_COLORS)], 
                alpha=0.1)
    
    # 设置雷达图标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # 设置Y轴范围
    ax.set_ylim(0, 1)
    
    # 添加图例和标题
    plt.legend(loc='upper right')
    plt.title('模型性能比较雷达图')
    
    # 保存图表
    save_path = os.path.join(RESULT_PATH, "evaluation", "model_comparison_radar.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"模型性能雷达图已保存至: {save_path}")

# 创建任务特定比较图
def create_task_specific_comparison(comparison_data, model_types):
    """创建各任务上的模型性能比较图"""
    # 获取所有任务
    tasks = comparison_data['Task'].unique()
    
    # 创建任务特定比较图
    plt.figure(figsize=(15, 10))
    
    for i, task in enumerate(tasks):
        plt.subplot(2, 2, i+1)
        
        task_data = comparison_data[comparison_data['Task'] == task]
        
        # 创建分组柱状图
        x = np.arange(len(task_data))
        width = 0.35
        
        plt.bar(x - width/2, task_data['Accuracy'], width, 
                label='准确率', color=MORANDI_COLD_COLORS[0])
        plt.bar(x + width/2, task_data['F1-Score'], width, 
                label='F1分数', color=MORANDI_COLD_COLORS[1])
        
        plt.title(f'{task.capitalize()} 任务模型性能比较')
        plt.xlabel('模型')
        plt.ylabel('得分')
        plt.xticks(x, task_data['Model'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(RESULT_PATH, "evaluation", "task_specific_comparison.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"任务特定比较图已保存至: {save_path}")

# 创建性能与效率权衡图
def create_performance_efficiency_tradeoff(comparison_summary):
    """创建性能与效率权衡图"""
    plt.figure(figsize=(12, 10))
    
    # 创建散点图，大小反映模型参数量，颜色反映综合得分
    scatter = plt.scatter(
        comparison_summary['Inference Time (ms)'],
        comparison_summary['Accuracy'],
        s=comparison_summary['Size (MB)'] / comparison_summary['Size (MB)'].max() * 500,
        c=comparison_summary['Overall Score'],
        cmap='viridis',
        alpha=0.7
    )
    
    # 添加模型标签
    for i, model in enumerate(comparison_summary['Model']):
        plt.annotate(
            model,
            (comparison_summary['Inference Time (ms)'].iloc[i],
             comparison_summary['Accuracy'].iloc[i]),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.title('模型性能与效率权衡')
    plt.xlabel('推理时间 (ms)')
    plt.ylabel('准确率')
    plt.colorbar(scatter, label='综合得分')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(RESULT_PATH, "evaluation", "performance_efficiency_tradeoff.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"性能与效率权衡图已保存至: {save_path}")

# 分析任务间相关性
def analyze_task_correlations(results):
    """分析不同任务之间的性能相关性"""
    # 准备数据
    correlation_data = {
        'Model': [],
        'Task': [],
        'Accuracy': [],
        'F1-Score': []
    }
    
    for model_name, model_results in results.items():
        for task, task_data in model_results.items():
            report = task_data['classification_report']
            
            correlation_data['Model'].append(model_name)
            correlation_data['Task'].append(task)
            correlation_data['Accuracy'].append(report['accuracy'])
            correlation_data['F1-Score'].append(report['weighted avg']['f1-score'])
    
    corr_df = pd.DataFrame(correlation_data)
    
    # 计算任务间准确率相关性
    pivot_acc = corr_df.pivot_table(index='Model', columns='Task', values='Accuracy')
    corr_acc = pivot_acc.corr()
    
    # 计算任务间F1分数相关性
    pivot_f1 = corr_df.pivot_table(index='Model', columns='Task', values='F1-Score')
    corr_f1 = pivot_f1.corr()
    
    # 可视化相关性热图
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(corr_acc, annot=True, cmap='Blues', vmin=-1, vmax=1)
    plt.title('任务间准确率相关性')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(corr_f1, annot=True, cmap='Blues', vmin=-1, vmax=1)
    plt.title('任务间F1分数相关性')
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(RESULT_PATH, "evaluation", "task_correlations.png")
    plt.savefig(save_path)
    plt.close()
    
    # 保存相关性数据
    csv_path_acc = os.path.join(RESULT_PATH, "evaluation", "task_accuracy_correlations.csv")
    csv_path_f1 = os.path.join(RESULT_PATH, "evaluation", "task_f1_correlations.csv")
    
    corr_acc.to_csv(csv_path_acc)
    corr_f1.to_csv(csv_path_f1)
    
    print(f"任务相关性分析已保存至: {save_path}")

# 分析共同错误案例
def analyze_common_errors(results, data_loaders):
    """分析所有模型共同的错误案例"""
    # 获取所有预测结果
    predictions = {}
    true_labels = {}
    
    for task in data_loaders.keys():
        task_preds = {}
        for model_name, model_results in results.items():
            if task in model_results:
                task_preds[model_name] = model_results[task]['predictions']
                if task not in true_labels:
                    true_labels[task] = model_results[task]['labels']
        
        predictions[task] = task_preds
    
    # 找出所有模型都预测错误的样本
    common_errors = {}
    
    for task in predictions.keys():
        model_names = list(predictions[task].keys())
        if not model_names:
            continue
            
        labels = true_labels[task]
        n_samples = len(labels)
        
        # 初始化错误标记
        all_wrong = np.ones(n_samples, dtype=bool)
        
        # 标记每个样本是否被所有模型预测错误
        for model_name in model_names:
            preds = predictions[task][model_name]
            wrong = (preds != labels)
            all_wrong = all_wrong & wrong
        
        # 提取共同错误的样本索引
        error_indices = np.where(all_wrong)[0]
        
        common_errors[task] = {
            'indices': error_indices,
            'true_labels': labels[error_indices]
        }
    
    # 分析共同错误
    for task, errors in common_errors.items():
        if len(errors['indices']) == 0:
            print(f"{task.capitalize()} 任务无共同错误案例")
            continue
            
        classes = data_loaders[task]['classes']
        
        # 计算共同错误的类别分布
        class_counts = {}
        for label in errors['true_labels']:
            class_name = classes[label]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        # 创建类别分布图
        plt.figure(figsize=(10, 6))
        plt.bar(class_counts.keys(), class_counts.values(), color=MORANDI_COLD_COLORS[0])
        plt.title(f'{task.capitalize()} 任务共同错误案例的类别分布')
        plt.xlabel('真实类别')
        plt.ylabel('错误样本数')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(RESULT_PATH, "evaluation", f"{task}_common_errors.png")
        plt.savefig(save_path)
        plt.close()
        
        # 保存共同错误数据
        error_data = []
        for idx, label in zip(errors['indices'], errors['true_labels']):
            error_data.append({
                'Index': idx,
                'True Class': classes[label],
                'Model Predictions': {
                    model: classes[predictions[task][model][idx]]
                    for model in predictions[task].keys()
                }
            })
        
        # 转换为DataFrame并保存
        # 处理字典列
        error_df = pd.DataFrame(error_data)
        for model in predictions[task].keys():
            error_df[f'{model} Prediction'] = error_df['Model Predictions'].apply(lambda x: x.get(model, ''))
        error_df = error_df.drop('Model Predictions', axis=1)
        
        csv_path = os.path.join(RESULT_PATH, "evaluation", f"{task}_common_errors.csv")
        error_df.to_csv(csv_path, index=False)
        
        print(f"{task.capitalize()} 任务共同错误分析已保存至: {save_path}")

# 可视化决策边界
def visualize_decision_boundaries(model_types, results):
    """尝试可视化部分类别间的决策边界"""
    # 这个功能在高维空间中很难实现，特别是对于图像数据
    # 我们可以使用t-SNE将特征降维，然后在二维空间中近似决策边界
    # 由于实现复杂度较高，此处仅提供概念设计
    print("决策边界可视化功能需要更多特征工程，暂未实现")

# 执行统计检验
def perform_statistical_tests(results):
    """执行统计检验，分析模型间性能差异是否显著"""
    # 准备数据
    test_data = {
        'Model': [],
        'Task': [],
        'Accuracy': [],
        'Sample Size': []
    }
    
    for model_name, model_results in results.items():
        for task, task_data in model_results.items():
            report = task_data['classification_report']
            # 计算样本总数
            sample_size = sum(report[cls]['support'] for cls in report if cls not in ['accuracy', 'macro avg', 'weighted avg'])
            
            test_data['Model'].append(model_name)
            test_data['Task'].append(task)
            test_data['Accuracy'].append(report['accuracy'])
            test_data['Sample Size'].append(sample_size)
    
    test_df = pd.DataFrame(test_data)
    
    # 统计分析结果
    model_stats = test_df.groupby('Model').agg({
        'Accuracy': ['mean', 'std'],
        'Sample Size': 'mean'
    }).reset_index()
    
    # 计算95%置信区间
    model_stats['95% CI Lower'] = model_stats[('Accuracy', 'mean')] - 1.96 * model_stats[('Accuracy', 'std')] / np.sqrt(len(results))
    model_stats['95% CI Upper'] = model_stats[('Accuracy', 'mean')] + 1.96 * model_stats[('Accuracy', 'std')] / np.sqrt(len(results))
    
    # 可视化模型准确率及置信区间
    plt.figure(figsize=(12, 8))
    
    y_pos = np.arange(len(model_stats))
    plt.barh(y_pos, model_stats[('Accuracy', 'mean')], 
             xerr=[model_stats[('Accuracy', 'mean')] - model_stats['95% CI Lower'], 
                   model_stats['95% CI Upper'] - model_stats[('Accuracy', 'mean')]], 
             align='center', color=MORANDI_COLD_COLORS[0], alpha=0.7)
    
    plt.yticks(y_pos, model_stats['Model'])
    plt.xlabel('平均准确率')
    plt.title('模型准确率及95%置信区间')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(RESULT_PATH, "evaluation", "model_accuracy_confidence_intervals.png")
    plt.savefig(save_path)
    plt.close()
    
    # 保存统计分析结果
    stats_df = pd.DataFrame({
        'Model': model_stats['Model'],
        'Mean Accuracy': model_stats[('Accuracy', 'mean')],
        'Std Accuracy': model_stats[('Accuracy', 'std')],
        '95% CI Lower': model_stats['95% CI Lower'],
        '95% CI Upper': model_stats['95% CI Upper']
    })
    
    csv_path = os.path.join(RESULT_PATH, "evaluation", "model_statistical_analysis.csv")
    stats_df.to_csv(csv_path, index=False)
    
    print(f"统计分析结果已保存至: {save_path}")

# 主函数
def main():
    parser = argparse.ArgumentParser(description="胃肠道疾病诊断模型综合性能评估")
    
    parser.add_argument('--models', nargs='+', default=['teacher_resnet50', 'student_mobilenetv3_small', 'hetero_moe', 'rl_hetero_moe'],
                        help='要评估的模型列表')
    
    parser.add_argument('--architectures', nargs='+', default=None,
                        help='异构模型使用的架构列表')
    
    parser.add_argument('--amp', action='store_true', default=USE_AMP,
                        help='使用自动混合精度')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='GPU IDs，用逗号分隔')
    
    args = parser.parse_args()
    
    # 设置GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print(f"使用GPU: {args.gpus}")
    
    # 运行评估
    results, comparison = comprehensive_model_evaluation(
        model_types=args.models,
        architectures=args.architectures,
        use_amp=args.amp,
        seed=args.seed
    )
    
    print("评估完成!")

if __name__ == "__main__":
    main()