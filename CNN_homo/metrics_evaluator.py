"""
多指标评估工具模块
提供准确率、AUC、F1-score、精确率、召回率等评估指标，包括95%置信区间
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MultiMetricsEvaluator:
    """多指标评估器"""
    
    def __init__(self, num_classes, task_name="unknown"):
        self.num_classes = num_classes
        self.task_name = task_name
        self.reset()
    
    def reset(self):
        """重置累计统计"""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
    
    def update(self, predictions, labels, probabilities=None):
        """
        更新预测结果
        
        Args:
            predictions: torch.Tensor, 预测的类别标签
            labels: torch.Tensor, 真实标签
            probabilities: torch.Tensor, 预测概率 (用于计算AUC)
        """
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()  # 确保detach
        
        self.all_predictions.extend(predictions.flatten())
        self.all_labels.extend(labels.flatten())
        
        if probabilities is not None:
            if len(probabilities.shape) == 1:  # 二分类
                self.all_probabilities.extend(probabilities.flatten())
            else:  # 多分类
                self.all_probabilities.extend(probabilities)
    
    def _bootstrap_confidence_interval(self, metric_func, n_bootstrap=1000, confidence_level=0.95):
        """
        使用Bootstrap方法计算指标的置信区间
        
        Args:
            metric_func: 指标计算函数
            n_bootstrap: Bootstrap采样次数
            confidence_level: 置信水平
            
        Returns:
            tuple: (lower_bound, upper_bound, metric_value)
        """
        if len(self.all_predictions) == 0:
            return 0.0, 0.0, 0.0
        
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)
        y_prob = np.array(self.all_probabilities) if self.all_probabilities else None
        
        # 计算原始指标值
        try:
            if y_prob is not None:
                original_metric = metric_func(y_true, y_pred, y_prob)
            else:
                original_metric = metric_func(y_true, y_pred)
        except Exception:
            return 0.0, 0.0, 0.0
        
        # Bootstrap采样
        bootstrap_scores = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # 有放回采样
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            try:
                if y_prob is not None:
                    if len(y_prob.shape) == 1:  # 二分类
                        y_prob_boot = y_prob[indices]
                    else:  # 多分类
                        y_prob_boot = y_prob[indices]
                    score = metric_func(y_true_boot, y_pred_boot, y_prob_boot)
                else:
                    score = metric_func(y_true_boot, y_pred_boot)
                
                if not np.isnan(score) and not np.isinf(score):
                    bootstrap_scores.append(score)
            except Exception:
                continue
        
        if len(bootstrap_scores) == 0:
            return 0.0, 0.0, original_metric
        
        # 计算置信区间
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_scores, lower_percentile)
        upper_bound = np.percentile(bootstrap_scores, upper_percentile)
        
        return lower_bound, upper_bound, original_metric
    
    def _accuracy_func(self, y_true, y_pred, y_prob=None):
        """准确率计算函数"""
        return accuracy_score(y_true, y_pred)
    
    def _auc_func(self, y_true, y_pred, y_prob):
        """
AUC计算函数"""
        if self.num_classes == 2:
            if len(y_prob.shape) == 1:
                return roc_auc_score(y_true, y_prob)
            else:
                return roc_auc_score(y_true, y_prob[:, 1])
        else:
            # 多分类 AUC
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            if y_true_bin.shape[1] == 1:
                return 0.5
            else:
                if len(y_prob.shape) == 2 and y_prob.shape[1] == self.num_classes:
                    return roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
                else:
                    return 0.0
    
    def compute_metrics(self):
        """
        计算所有评估指标
        
        Returns:
            dict: 包含各种评估指标的字典
        """
        if len(self.all_predictions) == 0:
            return {}
        
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)
        y_prob = np.array(self.all_probabilities) if self.all_probabilities else None
        
        metrics = {}
        
        # 基础指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100.0
        
        # 计算准确率的95%置信区间
        if len(y_true) >= 30:  # 只有样本数量足够时才计算置信区间
            acc_lower, acc_upper, _ = self._bootstrap_confidence_interval(self._accuracy_func)
            metrics['accuracy_ci_lower'] = acc_lower * 100.0
            metrics['accuracy_ci_upper'] = acc_upper * 100.0
        else:
            metrics['accuracy_ci_lower'] = metrics['accuracy']
            metrics['accuracy_ci_upper'] = metrics['accuracy']
        
        # 多分类指标 - 使用不同的平均方式
        if self.num_classes > 2:
            # Macro平均（每个类别权重相同）
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
            
            # Weighted平均（按类别样本数加权）
            metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0
            metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0
            
            # Micro平均（全局计算）
            metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0) * 100.0
            metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0) * 100.0
            metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0) * 100.0
        else:
            # 二分类指标
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0) * 100.0
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0) * 100.0
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0) * 100.0
        
        # AUC计算
        if y_prob is not None and len(y_prob) > 0:
            try:
                if self.num_classes == 2:
                    # 二分类AUC
                    if len(y_prob.shape) == 1:
                        auc = roc_auc_score(y_true, y_prob)
                    else:
                        auc = roc_auc_score(y_true, y_prob[:, 1])  # 使用正类概率
                    metrics['auc'] = auc
                    
                    # 计算AUC的95%置信区间
                    if len(y_true) >= 30:
                        auc_lower, auc_upper, _ = self._bootstrap_confidence_interval(self._auc_func)
                        metrics['auc_ci_lower'] = auc_lower
                        metrics['auc_ci_upper'] = auc_upper
                    else:
                        metrics['auc_ci_lower'] = auc
                        metrics['auc_ci_upper'] = auc
                else:
                    # 多分类AUC - One-vs-Rest
                    y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                    if y_true_bin.shape[1] == 1:  # 只有一个类别的情况
                        metrics['auc_ovr'] = 0.5
                        metrics['auc_ovr_ci_lower'] = 0.5
                        metrics['auc_ovr_ci_upper'] = 0.5
                    else:
                        if len(y_prob.shape) == 2 and y_prob.shape[1] == self.num_classes:
                            auc_ovr = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
                            metrics['auc_ovr'] = auc_ovr
                            
                            # 计算AUC的95%置信区间
                            if len(y_true) >= 30:
                                auc_lower, auc_upper, _ = self._bootstrap_confidence_interval(self._auc_func)
                                metrics['auc_ovr_ci_lower'] = auc_lower
                                metrics['auc_ovr_ci_upper'] = auc_upper
                            else:
                                metrics['auc_ovr_ci_lower'] = auc_ovr
                                metrics['auc_ovr_ci_upper'] = auc_ovr
                        else:
                            metrics['auc_ovr'] = 0.0  # 概率形状不匹配
                            metrics['auc_ovr_ci_lower'] = 0.0
                            metrics['auc_ovr_ci_upper'] = 0.0
            except Exception as e:
                print(f"AUC计算失败 ({self.task_name}): {e}")
                metrics['auc'] = 0.0 if self.num_classes == 2 else None
                metrics['auc_ci_lower'] = 0.0 if self.num_classes == 2 else None
                metrics['auc_ci_upper'] = 0.0 if self.num_classes == 2 else None
                if self.num_classes > 2:
                    metrics['auc_ovr'] = 0.0
                    metrics['auc_ovr_ci_lower'] = 0.0
                    metrics['auc_ovr_ci_upper'] = 0.0
        
        # 混淆矩阵信息
        try:
            cm = confusion_matrix(y_true, y_pred)
            # 计算每个类别的支持数
            class_support = np.sum(cm, axis=1)
            metrics['class_support'] = class_support.tolist()
            
            # 计算平衡准确率 (Balanced Accuracy)
            if self.num_classes == 2:
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['sensitivity'] = sensitivity * 100.0
                metrics['specificity'] = specificity * 100.0
                metrics['balanced_accuracy'] = (sensitivity + specificity) / 2 * 100.0
            else:
                # 多分类平衡准确率
                per_class_acc = []
                for i in range(self.num_classes):
                    if class_support[i] > 0:
                        per_class_acc.append(cm[i, i] / class_support[i])
                    else:
                        per_class_acc.append(0.0)
                metrics['balanced_accuracy'] = np.mean(per_class_acc) * 100.0
                metrics['per_class_accuracy'] = [acc * 100.0 for acc in per_class_acc]
        except Exception as e:
            print(f"混淆矩阵计算失败 ({self.task_name}): {e}")
        
        return metrics
    
    def get_summary_metrics(self):
        """获取主要的汇总指标"""
        metrics = self.compute_metrics()
        if not metrics:
            return {}
        
        summary = {
            'accuracy': metrics.get('accuracy', 0.0),
            'accuracy_ci_lower': metrics.get('accuracy_ci_lower', 0.0),
            'accuracy_ci_upper': metrics.get('accuracy_ci_upper', 0.0),
        }
        
        if self.num_classes == 2:
            summary.update({
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1': metrics.get('f1', 0.0),
                'auc': metrics.get('auc', 0.0),
                'auc_ci_lower': metrics.get('auc_ci_lower', 0.0),
                'auc_ci_upper': metrics.get('auc_ci_upper', 0.0),
                'balanced_accuracy': metrics.get('balanced_accuracy', 0.0)
            })
        else:
            summary.update({
                'precision_macro': metrics.get('precision_macro', 0.0),
                'recall_macro': metrics.get('recall_macro', 0.0),
                'f1_macro': metrics.get('f1_macro', 0.0),
                'f1_weighted': metrics.get('f1_weighted', 0.0),
                'auc_ovr': metrics.get('auc_ovr', 0.0),
                'auc_ovr_ci_lower': metrics.get('auc_ovr_ci_lower', 0.0),
                'auc_ovr_ci_upper': metrics.get('auc_ovr_ci_upper', 0.0),
                'balanced_accuracy': metrics.get('balanced_accuracy', 0.0)
            })
        
        return summary
    
    def print_metrics(self, prefix=""):
        """打印详细的评估指标"""
        metrics = self.compute_metrics()
        if not metrics:
            print(f"{prefix}没有可用的评估数据")
            return
        
        print(f"{prefix}=== {self.task_name.upper()} 评估指标 ===")
        # 显示准确率及其置信区间
        if 'accuracy_ci_lower' in metrics and 'accuracy_ci_upper' in metrics:
            print(f"{prefix}准确率: {metrics.get('accuracy', 0):.2f}% (95% CI: {metrics['accuracy_ci_lower']:.2f}%-{metrics['accuracy_ci_upper']:.2f}%)")
        else:
            print(f"{prefix}准确率: {metrics.get('accuracy', 0):.2f}%")
        
        if self.num_classes == 2:
            print(f"{prefix}精确率: {metrics.get('precision', 0):.2f}%")
            print(f"{prefix}召回率: {metrics.get('recall', 0):.2f}%")
            print(f"{prefix}F1分数: {metrics.get('f1', 0):.2f}%")
            # 显示AUC及其置信区间
            if 'auc' in metrics:
                if 'auc_ci_lower' in metrics and 'auc_ci_upper' in metrics:
                    print(f"{prefix}AUC: {metrics['auc']:.4f} (95% CI: {metrics['auc_ci_lower']:.4f}-{metrics['auc_ci_upper']:.4f})")
                else:
                    print(f"{prefix}AUC: {metrics['auc']:.4f}")
            if 'balanced_accuracy' in metrics:
                print(f"{prefix}平衡准确率: {metrics['balanced_accuracy']:.2f}%")
            if 'sensitivity' in metrics:
                print(f"{prefix}敏感性: {metrics['sensitivity']:.2f}%")
            if 'specificity' in metrics:
                print(f"{prefix}特异性: {metrics['specificity']:.2f}%")
        else:
            print(f"{prefix}精确率(Macro): {metrics.get('precision_macro', 0):.2f}%")
            print(f"{prefix}召回率(Macro): {metrics.get('recall_macro', 0):.2f}%")
            print(f"{prefix}F1分数(Macro): {metrics.get('f1_macro', 0):.2f}%")
            print(f"{prefix}F1分数(Weighted): {metrics.get('f1_weighted', 0):.2f}%")
            # 显示多分类AUC及其置信区间
            if 'auc_ovr' in metrics:
                if 'auc_ovr_ci_lower' in metrics and 'auc_ovr_ci_upper' in metrics:
                    print(f"{prefix}AUC(OvR): {metrics['auc_ovr']:.4f} (95% CI: {metrics['auc_ovr_ci_lower']:.4f}-{metrics['auc_ovr_ci_upper']:.4f})")
                else:
                    print(f"{prefix}AUC(OvR): {metrics['auc_ovr']:.4f}")
            if 'balanced_accuracy' in metrics:
                print(f"{prefix}平衡准确率: {metrics['balanced_accuracy']:.2f}%")


def create_task_evaluators(task_info):
    """
    为每个任务创建评估器
    
    Args:
        task_info: dict, 任务信息，格式为 {'task_name': num_classes}
    
    Returns:
        dict: 评估器字典
    """
    evaluators = {}
    for task, num_classes in task_info.items():
        evaluators[task] = MultiMetricsEvaluator(num_classes, task)
    return evaluators


def log_metrics_to_tensorboard(writer, metrics_dict, epoch, phase='train'):
    """
    将指标记录到TensorBoard
    
    Args:
        writer: TensorBoard SummaryWriter
        metrics_dict: dict, 指标字典
        epoch: int, 当前轮次
        phase: str, 阶段名称 ('train' 或 'val')
    """
    for task, metrics in metrics_dict.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    writer.add_scalar(f'{metric_name.title()}/{phase}/{task}', value, epoch)


def print_epoch_summary(metrics_dict, epoch, phase='Train'):
    """
    打印轮次总结
    
    Args:
        metrics_dict: dict, 任务指标字典
        epoch: int, 当前轮次
        phase: str, 阶段名称
    """
    print(f"\n=== Epoch {epoch+1} {phase} 指标总结 ===")
    
    # 计算总体指标
    all_accuracy = []
    all_f1 = []
    all_auc = []
    
    for task, metrics in metrics_dict.items():
        if isinstance(metrics, dict) and metrics:
            print(f"\n{task.upper()}:")
            # 显示准确率及其置信区间
            if 'accuracy_ci_lower' in metrics and 'accuracy_ci_upper' in metrics:
                print(f"  准确率: {metrics.get('accuracy', 0):.2f}% (95% CI: {metrics['accuracy_ci_lower']:.2f}%-{metrics['accuracy_ci_upper']:.2f}%)")
            else:
                print(f"  准确率: {metrics.get('accuracy', 0):.2f}%")
            
            if 'f1' in metrics:  # 二分类
                print(f"  精确率: {metrics.get('precision', 0):.2f}%")
                print(f"  召回率: {metrics.get('recall', 0):.2f}%")
                print(f"  F1分数: {metrics.get('f1', 0):.2f}%")
                # 显示AUC及其置信区间
                if 'auc_ci_lower' in metrics and 'auc_ci_upper' in metrics:
                    print(f"  AUC: {metrics.get('auc', 0):.4f} (95% CI: {metrics['auc_ci_lower']:.4f}-{metrics['auc_ci_upper']:.4f})")
                else:
                    print(f"  AUC: {metrics.get('auc', 0):.4f}")
                all_f1.append(metrics.get('f1', 0))
                all_auc.append(metrics.get('auc', 0))
            else:  # 多分类
                print(f"  F1(Macro): {metrics.get('f1_macro', 0):.2f}%")
                print(f"  F1(Weighted): {metrics.get('f1_weighted', 0):.2f}%")
                # 显示多分类AUC及其置信区间
                if 'auc_ovr_ci_lower' in metrics and 'auc_ovr_ci_upper' in metrics:
                    print(f"  AUC(OvR): {metrics.get('auc_ovr', 0):.4f} (95% CI: {metrics['auc_ovr_ci_lower']:.4f}-{metrics['auc_ovr_ci_upper']:.4f})")
                else:
                    print(f"  AUC(OvR): {metrics.get('auc_ovr', 0):.4f}")
                all_f1.append(metrics.get('f1_macro', 0))
                all_auc.append(metrics.get('auc_ovr', 0))
            
            all_accuracy.append(metrics.get('accuracy', 0))
    
    # 总体平均指标
    if all_accuracy:
        print(f"\n总体平均:")
        print(f"  平均准确率: {np.mean(all_accuracy):.2f}%")
        if all_f1:
            print(f"  平均F1分数: {np.mean(all_f1):.2f}%")
        if all_auc:
            print(f"  平均AUC: {np.mean(all_auc):.4f}")
    
    print("=" * 50)