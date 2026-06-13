import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from itertools import cycle
from matplotlib.font_manager import FontProperties
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'

from config import (
    DEVICE, RESULT_PATH, COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, UGI_DISEASE_PATH,
    LANCET_COLORS, LANCET_PASTEL_COLORS, FIG_SIZE, DPI, LABEL_SMOOTHING
)
from data_loader import create_multi_task_loaders
from models.moe_model import MixtureOfExperts
from models.student_model import StudentModel
from models.hetero_moe_model import HeterogeneousMixtureOfExperts
from utils import load_model, ProgressBar, get_class_names

def evaluate_model(model, data_loaders, model_name):
    """评估模型性能并生成详细的可视化报告"""
    model.eval()
    results = {}
    
    # 初始化评估损失函数
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    for task, task_data in data_loaders.items():
        test_loader = task_data['test_loader']
        classes = task_data['classes']
        
        # 收集模型预测结果和真实标签
        all_preds = []
        all_labels = []
        all_probs = []
        all_losses = []
        
        pbar = ProgressBar(len(test_loader), desc=f"Evaluating {task} task")
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                # 适配多种GPU并行模型架构
                if isinstance(model, (MixtureOfExperts, HeterogeneousMixtureOfExperts)):
                    outputs, routing_weights = model(inputs, task=task)
                elif isinstance(model, nn.DataParallel) and isinstance(model.module, (MixtureOfExperts, HeterogeneousMixtureOfExperts)):
                    outputs, routing_weights = model(inputs, task=task)
                else:  # Student model or other models
                    outputs = model(inputs, task=task)
                
                # 计算损失
                loss = criterion(outputs, labels)
                all_losses.append(loss.item())
                
                # Get prediction probabilities
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                
                # Get predicted classes
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.update()
        
        pbar.close()
        
        
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.concatenate(all_probs, axis=0)
        avg_loss = np.mean(all_losses)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Calculate classification report
        report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
        
        # Save results
        results[task] = {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'confusion_matrix': cm,
            'classification_report': report,
            'classes': classes,
            'loss': avg_loss,
        }
        
        print(f"{task} task evaluation - Loss: {avg_loss:.4f}, Accuracy: {report['accuracy']:.4f}")
        
        # Generate visualizations
        plot_confusion_matrix(cm, classes, task, model_name)
        plot_normalized_confusion_matrix(cm, classes, task, model_name)
        plot_roc_curves(all_labels, all_probs, classes, task, model_name)
        plot_precision_recall_curves(all_labels, all_probs, classes, task, model_name)
        plot_classification_metrics(report, classes, task, model_name)
        
        # 生成错误分析表格
        create_error_analysis_table(all_labels, all_preds, classes, task, model_name)
    
    # Generate overall performance report
    generate_overall_report(results, model_name)
    
    # Generate expert usage analysis if applicable
    if isinstance(model, (MixtureOfExperts, HeterogeneousMixtureOfExperts)):
        analyze_expert_usage(model, data_loaders, model_name)
        
        # 如果是异构MoE，分析每个专家的特定性能
        if isinstance(model, HeterogeneousMixtureOfExperts) and hasattr(model, 'model_names'):
            analyze_expert_performance(model, data_loaders, model_name)
    
    return results

def plot_confusion_matrix(cm, classes, task, model_name):
    """Plot confusion matrix"""
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap=sns.light_palette(LANCET_COLORS[0], as_cmap=True),
                xticklabels=classes, yticklabels=classes)
    
    plt.title(f'Confusion Matrix - {task.capitalize()}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_{task}_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def plot_normalized_confusion_matrix(cm, classes, task, model_name):
    """Plot normalized confusion matrix"""
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap with normalized values
    sns.heatmap(cm_norm, annot=True, fmt='.2f', 
                cmap=sns.light_palette(LANCET_COLORS[2], as_cmap=True),
                xticklabels=classes, yticklabels=classes, vmin=0, vmax=1)
    
    plt.title(f'Normalized Confusion Matrix - {task.capitalize()}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_{task}_normalized_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Normalized confusion matrix saved to: {save_path}")

def plot_roc_curves(y_true, y_score, classes, task, model_name):
    """Plot ROC curves"""
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    
    # Binarize labels for multi-class ROC
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculate micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot all ROC curves
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
             color=LANCET_COLORS[4], linestyle=':', linewidth=3)
    
    # Plot random chance line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Plot individual ROC curves if there aren't too many classes
    if n_classes <= 10:
        colors = cycle(LANCET_COLORS)
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {task.capitalize()}', fontsize=14)
    
    # Adjust legend size if there are too many classes
    if n_classes <= 20:
        plt.legend(loc="lower right", fontsize='small')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_{task}_roc_curves.png")
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curves saved to: {save_path}")

def plot_precision_recall_curves(y_true, y_score, classes, task, model_name):
    """Plot precision-recall curves"""
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    
    # Binarize labels for multi-class precision-recall
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Calculate precision-recall curve for each class
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        avg_precision[i] = np.mean(precision[i])
    
    # Plot individual precision-recall curves if there aren't too many classes
    if n_classes <= 10:
        colors = cycle(LANCET_PASTEL_COLORS)
        for i, color in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                    label=f'{classes[i]} (AP = {avg_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves - {task.capitalize()}', fontsize=14)
    
    # Adjust legend size if there are too many classes
    if n_classes <= 20:
        plt.legend(loc="best", fontsize='small')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_{task}_precision_recall_curves.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Precision-recall curves saved to: {save_path}")

def plot_classification_metrics(report, classes, task, model_name):
    """Plot classification metrics for each class"""
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    
    # Prepare data for plotting
    metrics = ['precision', 'recall', 'f1-score']
    class_metrics = {cls: [report[cls][metric] for metric in metrics] for cls in classes}
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Class': np.repeat(classes, len(metrics)),
        'Metric': np.tile(metrics, len(classes)),
        'Score': [score for cls in classes for score in class_metrics[cls]]
    })
    
    # Create grouped bar plot
    ax = plt.subplot(111)
    sns.barplot(x='Class', y='Score', hue='Metric', data=df, palette=LANCET_COLORS[:3])
    
    plt.title(f'Classification Metrics - {task.capitalize()}', fontsize=14)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(title='Metric')
    
    # Rotate x-axis labels if there are many classes
    if len(classes) > 6:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_{task}_class_metrics.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Classification metrics plot saved to: {save_path}")

def create_error_analysis_table(y_true, y_pred, classes, task, model_name):
    """创建错误分析表格"""
    # 创建错误矩阵
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            errors.append({
                'True Class': classes[y_true[i]],
                'Predicted Class': classes[y_pred[i]]
            })
    
    if not errors:
        print(f"No errors found for {task} task.")
        return
    
    # 将错误转为DataFrame
    error_df = pd.DataFrame(errors)
    
    # 统计每种错误类型的数量
    error_counts = error_df.groupby(['True Class', 'Predicted Class']).size().reset_index(name='Count')
    error_counts = error_counts.sort_values('Count', ascending=False)
    
    # 保存错误分析表
    save_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_{task}_error_analysis.csv")
    error_counts.to_csv(save_path, index=False)
    
    # 可视化主要错误类型
    plt.figure(figsize=(12, 8), dpi=DPI)
    
    # 只显示前20种最常见的错误
    top_errors = error_counts.head(20)
    
    # 创建错误类型标签
    error_labels = [f"{true} → {pred}" for true, pred in zip(top_errors['True Class'], top_errors['Predicted Class'])]
    
    # 绘制条形图
    ax = sns.barplot(x='Count', y=range(len(error_labels)), data=top_errors, 
                    palette=sns.color_palette("viridis", len(error_labels)))
    
    plt.yticks(range(len(error_labels)), error_labels)
    plt.title(f'Top Error Types - {task.capitalize()}', fontsize=14)
    plt.xlabel('Error Count', fontsize=12)
    plt.ylabel('Error Type (True → Predicted)', fontsize=12)
    plt.tight_layout()
    
    # 添加计数标签
    for i, count in enumerate(top_errors['Count']):
        ax.text(count + 0.5, i, str(count), va='center')
    
    # 保存图表
    viz_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_{task}_error_analysis.png")
    plt.savefig(viz_path)
    plt.close()
    
    print(f"Error analysis saved to: {save_path} and {viz_path}")

def generate_overall_report(results, model_name):
    """Generate overall performance report"""
    # Prepare data
    tasks = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    
    # Create DataFrame
    df_data = []
    
    for task in tasks:
        report = results[task]['classification_report']
        row = {'Task': task.capitalize(), 'Loss': results[task]['loss']}
        
        for metric in metrics:
            row[metric] = report['weighted avg'][metric]
        
        df_data.append(row)
    
    report_df = pd.DataFrame(df_data)
    
    # Save as CSV
    csv_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_overall_report.csv")
    report_df.to_csv(csv_path, index=False)
    
    # Generate performance comparison chart
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    
    # Use Lancet colors for bar plot
    ax = plt.subplot(111)
    x = np.arange(len(tasks))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, report_df[metric], width,
               color=LANCET_COLORS[i % len(LANCET_COLORS)],
               label=metric.capitalize())
    
    plt.title(f'{model_name} Overall Performance', fontsize=14)
    plt.xlabel('Task', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(x + width * (len(metrics)-1)/2, report_df['Task'])
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_overall_performance.png")
    plt.savefig(fig_path)
    plt.close()
    
    # Create radar chart for overall performance
    create_radar_chart(report_df, model_name)
    
    print(f"Overall performance report saved to: {csv_path}")
    print(f"Overall performance chart saved to: {fig_path}")

def create_radar_chart(report_df, model_name):
    """Create radar chart for model performance"""
    # Prepare data for radar chart
    categories = report_df['Task'].tolist()
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    
    # Set up figure
    fig = plt.figure(figsize=(10, 8), dpi=DPI)
    ax = fig.add_subplot(111, polar=True)
    
    # Number of categories
    N = len(categories)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        values = report_df[metric].tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, label=metric.capitalize(), color=LANCET_COLORS[i % len(LANCET_COLORS)])
        ax.fill(angles, values, alpha=0.1, color=LANCET_COLORS[i % len(LANCET_COLORS)])
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add gridlines
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Add legend
    plt.legend(loc='upper right')
    
    plt.title(f'{model_name} Performance Radar Chart', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_radar_chart.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Radar chart saved to: {save_path}")

def analyze_expert_usage(model, data_loaders, model_name):
    """Analyze and visualize expert usage patterns"""
    if not hasattr(model, 'num_experts'):
        print("Model does not support expert usage analysis")
        return
    
    model.eval()
    expert_usage = {task: np.zeros(model.num_experts) for task in data_loaders.keys()}
    task_samples = {task: 0 for task in data_loaders.keys()}
    
    with torch.no_grad():
        for task, task_data in data_loaders.items():
            test_loader = task_data['test_loader']
            pbar = ProgressBar(len(test_loader), desc=f"Analyzing expert usage for {task}")
            
            for inputs, _ in test_loader:
                inputs = inputs.to(DEVICE)
                batch_size = inputs.size(0)
                
                # Get routing weights
                _, routing_weights = model(inputs, task=task)
                
                # Get top-k experts for each sample
                if hasattr(model, 'top_k'):
                    top_k = model.top_k
                else:
                    top_k = 2  # Default value
                
                _, top_indices = torch.topk(routing_weights, top_k, dim=1)
                
                # Count expert usage
                for indices in top_indices:
                    for idx in indices:
                        expert_usage[task][idx.item()] += 1
                
                task_samples[task] += batch_size
                pbar.update()
            
            pbar.close()
    
    # Normalize expert usage
    for task in expert_usage:
        if task_samples[task] > 0:
            expert_usage[task] = expert_usage[task] / (task_samples[task] * top_k)
    
    # Plot expert usage patterns
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    
    if isinstance(model, HeterogeneousMixtureOfExperts) and hasattr(model, 'model_names'):
        expert_labels = [f"{i+1}: {name}" for i, name in enumerate(model.model_names)]
    else:
        expert_labels = [f"Expert {i+1}" for i in range(model.num_experts)]
    
    # Create subplot for each task
    for i, (task, usage) in enumerate(expert_usage.items()):
        plt.subplot(len(expert_usage), 1, i+1)
        plt.bar(expert_labels, usage, color=LANCET_PASTEL_COLORS)
        plt.title(f'Expert Usage for {task.capitalize()}', fontsize=12)
        plt.ylabel('Usage Frequency', fontsize=10)
        
        if i == len(expert_usage) - 1:  # Last subplot
            plt.xlabel('Experts', fontsize=10)
        
        # Rotate x-axis labels if there are many experts
        if model.num_experts > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_expert_usage.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Expert usage analysis saved to: {save_path}")
    
    # 保存专家使用率数据
    usage_data = []
    for task in expert_usage.keys():
        for i, usage in enumerate(expert_usage[task]):
            expert_name = expert_labels[i] if i < len(expert_labels) else f"Expert {i+1}"
            usage_data.append({
                'Task': task.capitalize(),
                'Expert': expert_name,
                'Usage': usage
            })
    
    usage_df = pd.DataFrame(usage_data)
    csv_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_expert_usage.csv")
    usage_df.to_csv(csv_path, index=False)
    print(f"Expert usage data saved to: {csv_path}")
    
    # 创建热力图
    plt.figure(figsize=(10, 6), dpi=DPI)
    pivot_table = usage_df.pivot(index='Task', columns='Expert', values='Usage')
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis')
    plt.title(f'Expert Usage Heatmap - {model_name}', fontsize=14)
    plt.tight_layout()
    
    heatmap_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_expert_usage_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Expert usage heatmap saved to: {heatmap_path}")

def analyze_expert_performance(model, data_loaders, model_name):
    """分析每个专家的单独性能"""
    if not isinstance(model, HeterogeneousMixtureOfExperts) or not hasattr(model, 'model_names'):
        print("Model does not support expert performance analysis")
        return
    
    # 获取专家名称
    expert_names = model.model_names
    
    # 创建性能存储表
    performance_data = []
    
    for task, task_data in data_loaders.items():
        test_loader = task_data['test_loader']
        classes = task_data['classes']
        
        for expert_idx, expert_name in enumerate(expert_names):
            print(f"Analyzing performance of expert {expert_name} on {task} task...")
            
            # 收集预测和标签
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    
                    # 获取专家特征
                    expert_features = model.experts[expert_idx](inputs)
                    # 应用投影层
                    expert_features = model.projections[expert_idx](expert_features)
                    # 应用任务分类器
                    outputs = model.classifiers[task](expert_features)
                    
                    # 获取预测类别
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # 计算性能指标
            report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
            
            # 保存性能数据
            performance_data.append({
                'Task': task.capitalize(),
                'Expert': expert_name,
                'Accuracy': report['accuracy'],
                'Precision': report['weighted avg']['precision'],
                'Recall': report['weighted avg']['recall'],
                'F1-Score': report['weighted avg']['f1-score']
            })
    
    # 创建性能DataFrame
    perf_df = pd.DataFrame(performance_data)
    
    # 保存为CSV
    csv_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_expert_performance.csv")
    perf_df.to_csv(csv_path, index=False)
    
    # 为每个任务可视化专家性能
    for task in data_loaders.keys():
        task_perf = perf_df[perf_df['Task'] == task.capitalize()]
        
        plt.figure(figsize=(12, 8), dpi=DPI)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(task_perf))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, task_perf[metric], width,
                   color=LANCET_COLORS[i % len(LANCET_COLORS)],
                   label=metric)
        
        plt.title(f'Expert Performance on {task.capitalize()} Task', fontsize=14)
        plt.xlabel('Expert', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(x + width * (len(metrics)-1)/2, task_perf['Expert'])
        plt.legend()
        plt.ylim(0, 1.1)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 如果有多个专家，旋转标签
        if len(expert_names) > 4:
            plt.xticks(rotation=45, ha='right')
            
        plt.tight_layout()
        
        # 保存图表
        viz_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_{task}_expert_performance.png")
        plt.savefig(viz_path)
        plt.close()
    
    print(f"Expert performance analysis saved to: {csv_path}")

def evaluate_teacher(base_model_name='resnet50'):
    """Evaluate teacher model"""
    # Load data
    data_loaders = create_multi_task_loaders(
        COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, UGI_DISEASE_PATH
    )
    
    # Get number of classes
    num_colon_classes = len(data_loaders['colon']['classes'])
    num_ugi_classes = len(data_loaders['ugi']['classes'])
    num_colon_disease_classes = len(data_loaders['colon_disease']['classes'])
    num_ugi_disease_classes = len(data_loaders['ugi_disease']['classes'])
    
    # Create model
    model = MixtureOfExperts(
        base_model_name=base_model_name,
        num_colon_classes=num_colon_classes,
        num_ugi_classes=num_ugi_classes,
        num_colon_disease_classes=num_colon_disease_classes,
        num_ugi_disease_classes=num_ugi_disease_classes
    ).to(DEVICE)
    
    # Load model weights
    optimizer = torch.optim.AdamW(model.parameters())
    model_name = f"teacher_{base_model_name}"
    load_model(model, optimizer, model_name)
    
    # Evaluate model
    results = evaluate_model(model, data_loaders, model_name)
    
    return results

def evaluate_student(base_model_name='mobilenetv3_small'):
    """Evaluate student model"""
    # Load data
    data_loaders = create_multi_task_loaders(
        COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, UGI_DISEASE_PATH
    )
    
    # Get number of classes
    num_colon_classes = len(data_loaders['colon']['classes'])
    num_ugi_classes = len(data_loaders['ugi']['classes'])
    num_colon_disease_classes = len(data_loaders['colon_disease']['classes'])
    num_ugi_disease_classes = len(data_loaders['ugi_disease']['classes'])
    
    # Create model
    model = StudentModel(
        base_model_name=base_model_name,
        num_colon_classes=num_colon_classes,
        num_ugi_classes=num_ugi_classes,
        num_colon_disease_classes=num_colon_disease_classes,
        num_ugi_disease_classes=num_ugi_disease_classes
    ).to(DEVICE)
    
    # Load model weights
    optimizer = torch.optim.AdamW(model.parameters())
    model_name = f"student_{base_model_name}"
    load_model(model, optimizer, model_name)
    
    # Evaluate model
    results = evaluate_model(model, data_loaders, model_name)
    
    return results

def evaluate_heterogeneous_moe(model_names, task_weights=None):
    """Evaluate heterogeneous mixture of experts model"""
    # Load data
    data_loaders = create_multi_task_loaders(
        COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, UGI_DISEASE_PATH
    )
    
    # Get number of classes
    num_colon_classes = len(data_loaders['colon']['classes'])
    num_ugi_classes = len(data_loaders['ugi']['classes'])
    num_colon_disease_classes = len(data_loaders['colon_disease']['classes'])
    num_ugi_disease_classes = len(data_loaders['ugi_disease']['classes'])
    
    # Create model
    model = HeterogeneousMixtureOfExperts(
        model_names=model_names,
        num_colon_classes=num_colon_classes,
        num_ugi_classes=num_ugi_classes,
        num_colon_disease_classes=num_colon_disease_classes,
        num_ugi_disease_classes=num_ugi_disease_classes
    ).to(DEVICE)
    
    # Load model weights
    optimizer = torch.optim.AdamW(model.parameters())
    model_name = f"hetero_moe"
    load_model(model, optimizer, model_name)
    
    # Evaluate model
    results = evaluate_model(model, data_loaders, model_name)
    
    return results

def compare_all_models(model_combinations=None):
    """Compare all model types including heterogeneous MoE"""
    # Load data
    data_loaders = create_multi_task_loaders(
        COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, UGI_DISEASE_PATH
    )
    
    # Default models to compare if none specified
    if model_combinations is None:
        model_combinations = {
            "teacher_resnet50": ["teacher_resnet50"],
            "hetero_moe": ["hetero_moe"],
            "student_mobilenetv3_small": ["student_mobilenetv3_small"]
        }
    
    # Get all model names
    model_names = list(model_combinations.keys())
    
    # Compare models
    compare_models(model_names, data_loaders)
    
def compare_models(model_names, data_loaders):
    """Compare multiple models performance"""
    # Load all model performance data
    model_results = {}
    
    for model_name in model_names:
        csv_path = os.path.join(RESULT_PATH, "visualizations", f"{model_name}_overall_report.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            model_results[model_name] = df
    
    if not model_results:
        print("No model performance reports found. Please evaluate models first.")
        return
    
    # Compare models for each task
    tasks = data_loaders.keys()
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    
    for task in tasks:
        plt.figure(figsize=FIG_SIZE, dpi=DPI)
        
        # Prepare data
        task_data = []
        for model_name, df in model_results.items():
            task_row = df[df['Task'] == task.capitalize()]
            if not task_row.empty:
                row_data = {
                    'Model': model_name,
                    **{metric: task_row[metric].values[0] for metric in metrics}
                }
                task_data.append(row_data)
        
        if not task_data:
            continue
            
        comp_df = pd.DataFrame(task_data)
        
        # Create comparison plot
        ax = plt.subplot(111)
        x = np.arange(len(comp_df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, comp_df[metric], width,
                   color=LANCET_COLORS[i % len(LANCET_COLORS)],
                   label=metric.capitalize())
        
        plt.title(f'Model Comparison - {task.capitalize()} Task', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(x + width * (len(metrics)-1)/2, comp_df['Model'])
        plt.legend()
        plt.ylim(0, 1.1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(RESULT_PATH, "visualizations", f"model_comparison_{task}.png")
        plt.savefig(fig_path)
        plt.close()
        
        print(f"Model comparison for {task.capitalize()} task saved to: {fig_path}")
    
    # Create model comparison radar chart
    create_model_comparison_radar_chart(model_results, tasks)
    
    # Create comprehensive comparison dashboard
    create_comparison_dashboard(model_results, tasks)

def create_model_comparison_radar_chart(model_results, tasks):
    """Create radar chart comparing different models"""
    plt.figure(figsize=(12, 10), dpi=DPI)
    
    # Prepare radar chart data
    models = list(model_results.keys())
    task_metric_pairs = [(task, 'accuracy') for task in tasks]  # Use accuracy as the main metric
    
    # Set up angles for radar chart
    angles = np.linspace(0, 2*np.pi, len(tasks), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Set up radar chart
    ax = plt.subplot(111, polar=True)
    
    # Plot each model
    for i, model_name in enumerate(models):
        values = []
        for task in tasks:
            task_cap = task.capitalize()
            df = model_results[model_name]
            task_row = df[df['Task'] == task_cap]
            if not task_row.empty:
                values.append(task_row['accuracy'].values[0])
            else:
                values.append(0)
        
        values += values[:1]  # Close the loop
        
        # Plot and fill
        ax.plot(angles, values, linewidth=2, 
                label=model_name, 
                color=LANCET_COLORS[i % len(LANCET_COLORS)])
        ax.fill(angles, values, alpha=0.1, 
                color=LANCET_COLORS[i % len(LANCET_COLORS)])
    
    # Set labels and ticks
    task_labels = [task.capitalize() for task in tasks]
    plt.xticks(angles[:-1], task_labels)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add gridlines
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
    
    plt.title('Model Comparison - Accuracy by Task', fontsize=14)
    plt.legend(loc='upper right')
    
    # Save radar chart
    radar_path = os.path.join(RESULT_PATH, "visualizations", "model_comparison_radar.png")
    plt.savefig(radar_path)
    plt.close()
    
    print(f"Model comparison radar chart saved to: {radar_path}")

def create_comparison_dashboard(model_results, tasks):
    """Create a comprehensive comparison dashboard"""
    # Prepare data for all metrics and models
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    
    # Initialize figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=DPI)
    axs = axs.flatten()
    
    # Plot each metric separately
    for i, metric in enumerate(metrics):
        metric_data = []
        
        for model_name, df in model_results.items():
            for task in tasks:
                task_cap = task.capitalize()
                task_row = df[df['Task'] == task_cap]
                if not task_row.empty:
                    metric_data.append({
                        'Model': model_name,
                        'Task': task_cap,
                        'Score': task_row[metric].values[0]
                    })
        
        if not metric_data:
            continue
            
        metric_df = pd.DataFrame(metric_data)
        
        # Create grouped bar plot
        sns.barplot(x='Task', y='Score', hue='Model', data=metric_df, 
                   palette=LANCET_PASTEL_COLORS, ax=axs[i])
        
        axs[i].set_title(f'{metric.capitalize()}', fontsize=14)
        axs[i].set_xlabel('Task', fontsize=12)
        axs[i].set_ylabel('Score', fontsize=12)
        axs[i].set_ylim(0, 1.1)
        axs[i].grid(True, linestyle='--', alpha=0.5)
        
        # Adjust legend
        axs[i].legend(title='Model', fontsize='small')
    
    fig.suptitle('Comprehensive Model Comparison Dashboard', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save dashboard
    dashboard_path = os.path.join(RESULT_PATH, "visualizations", "model_comparison_dashboard.png")
    plt.savefig(dashboard_path)
    plt.close()
    
    print(f"Comprehensive comparison dashboard saved to: {dashboard_path}")