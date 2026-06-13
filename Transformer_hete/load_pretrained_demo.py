#!/usr/bin/env python3
"""
演示如何正确加载第一阶段MoE预训练模型的脚本
"""

import torch
import os
import sys

# 添加项目路径
sys.path.append('/mnt/e/胃肠数据集/zuizhong/Transformer')

def load_pretrained_moe_model():
    """演示如何正确加载第一阶段的MoE预训练模型"""
    
    # 模型文件路径
    model_path = '/mnt/e/胃肠数据集/zuizhong/Transformer/models/transformer.pth'
    
    print("=== 第一阶段MoE预训练模型加载演示 ===\n")
    
    try:
        # 1. 加载保存的checkpoint
        print("1. 加载checkpoint文件...")
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        print(f"✅ 成功加载checkpoint")
        print(f"   - 训练轮次: {checkpoint['epoch']}")
        print(f"   - 最佳准确率: {checkpoint['best_acc']:.4f}")
        print(f"   - 参数总数: {len(state_dict)}")
        print()
        
        # 2. 分析专家网络结构
        print("2. 分析预训练的专家网络结构...")
        
        # 统计每个专家的参数
        expert_stats = {}
        for key in state_dict.keys():
            if key.startswith('experts.'):
                parts = key.split('.')
                if len(parts) >= 2:
                    expert_idx = int(parts[1])
                    if expert_idx not in expert_stats:
                        expert_stats[expert_idx] = {
                            'param_count': 0,
                            'components': set(),
                            'sample_keys': []
                        }
                    
                    tensor = state_dict[key]
                    if hasattr(tensor, 'numel'):
                        expert_stats[expert_idx]['param_count'] += tensor.numel()
                    
                    if len(parts) >= 3:
                        expert_stats[expert_idx]['components'].add(parts[2])
                    
                    if len(expert_stats[expert_idx]['sample_keys']) < 3:
                        expert_stats[expert_idx]['sample_keys'].append(key)
        
        for idx in sorted(expert_stats.keys()):
            stats = expert_stats[idx]
            print(f"   专家 {idx}:")
            print(f"     - 参数数量: {stats['param_count']:,}")
            print(f"     - 主要组件: {sorted(stats['components'])}")
            print(f"     - 示例参数: {stats['sample_keys'][:2]}")
        print()
        
        # 3. 分析门控网络
        print("3. 分析门控网络结构...")
        gate_tasks = set()
        for key in state_dict.keys():
            if key.startswith('gates.'):
                parts = key.split('.')
                if len(parts) >= 2:
                    gate_tasks.add(parts[1])
        
        print(f"   发现 {len(gate_tasks)} 个任务的门控网络:")
        for task in sorted(gate_tasks):
            # 找到该任务的专家选择层
            for key in state_dict.keys():
                if key.startswith(f'gates.{task}.') and key.endswith('.4.weight'):
                    tensor = state_dict[key]
                    if hasattr(tensor, 'shape'):
                        print(f"     - {task}: 选择 {tensor.shape[0]} 个专家")
                        break
        print()
        
        # 4. 分析分类器
        print("4. 分析任务特定分类器...")
        for key in state_dict.keys():
            if key.startswith('classifiers.') and key.endswith('.weight'):
                parts = key.split('.')
                if len(parts) >= 2:
                    task = parts[1]
                    tensor = state_dict[key]
                    if hasattr(tensor, 'shape'):
                        print(f"   任务 '{task}': {tensor.shape[0]} 个类别")
        print()
        
        # 5. 专家重要性权重
        print("5. 专家重要性权重...")
        if 'expert_importance' in state_dict:
            importance = state_dict['expert_importance']
            if hasattr(importance, 'tolist'):
                values = importance.tolist()
                print(f"   专家重要性值: {[f'{v:.3f}' for v in values]}")
                print(f"   最重要专家: 专家{values.index(max(values))} (权重: {max(values):.3f})")
        print()
        
        # 6. 如何在新模型中使用这些权重
        print("6. 在新MoE模型中使用预训练权重的建议:")
        print("   方法1: 直接加载整个MoE模型")
        print("   ```python")
        print("   # 创建相同结构的MoE模型")
        print("   moe_model = HeterogeneousMixtureOfExperts(...)")
        print("   # 加载预训练权重")
        print("   moe_model.load_state_dict(state_dict, strict=False)")
        print("   ```")
        print()
        print("   方法2: 选择性加载特定专家")
        print("   ```python")
        print("   # 只加载特定专家的权重")
        print("   expert_0_state = {k[10:]: v for k, v in state_dict.items() ")
        print("                    if k.startswith('experts.0.')}")
        print("   new_expert.load_state_dict(expert_0_state, strict=False)")
        print("   ```")
        print()
        print("   方法3: 加载门控网络权重")
        print("   ```python")
        print("   # 加载预训练的门控网络")
        print("   gate_state = {k[6:]: v for k, v in state_dict.items() ")
        print("                if k.startswith('gates.')}")
        print("   ```")
        print()
        
        return checkpoint, state_dict
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None, None

def demonstrate_loading_strategy():
    """演示不同的加载策略"""
    
    print("=== 不同加载策略的适用场景 ===\n")
    
    scenarios = [
        {
            "name": "场景1: 继续第一阶段训练",
            "description": "在相同的MoE架构上继续训练",
            "strategy": "直接加载完整的state_dict",
            "code": """
# 创建相同的MoE模型
from models.hetero_moe_model import HeterogeneousMixtureOfExperts
model = HeterogeneousMixtureOfExperts(...)

# 加载完整的预训练权重
checkpoint = torch.load('transformer.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
"""
        },
        {
            "name": "场景2: 开始第二阶段训练",
            "description": "使用预训练专家进行新的MoE训练",
            "strategy": "选择性加载专家权重，重新初始化门控网络",
            "code": """
# 创建新的MoE模型
new_moe_model = NewMoEArchitecture(...)

# 选择性加载专家权重
checkpoint = torch.load('transformer.pth', map_location='cpu')
state_dict = checkpoint['model_state_dict']

# 加载专家网络权重
expert_weights = {k: v for k, v in state_dict.items() 
                 if k.startswith('experts.')}
new_moe_model.load_state_dict(expert_weights, strict=False)

# 门控网络和分类器将使用新的随机初始化
"""
        },
        {
            "name": "场景3: 迁移学习",
            "description": "在新任务上使用预训练专家",
            "strategy": "加载专家权重，根据新任务调整分类器",
            "code": """
# 为新任务创建模型
task_model = TaskSpecificModel(...)

# 加载特定专家的权重
expert_state = {k[10:]: v for k, v in state_dict.items() 
               if k.startswith('experts.0.')}  # 加载专家0
task_model.backbone.load_state_dict(expert_state, strict=False)

# 分类器针对新任务重新训练
"""
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"   描述: {scenario['description']}")
        print(f"   策略: {scenario['strategy']}")
        print(f"   代码示例:{scenario['code']}")
        print()

if __name__ == "__main__":
    # 执行分析
    checkpoint, state_dict = load_pretrained_moe_model()
    
    if checkpoint is not None:
        print("\n" + "="*60)
        demonstrate_loading_strategy()
        
        print("="*60)
        print("📋 总结:")
        print("transformer.pth 是一个完整的第一阶段MoE训练结果，包含:")
        print("✅ 6个不同架构的预训练专家网络")
        print("✅ 4个任务的门控网络权重")
        print("✅ 任务特定的分类器")
        print("✅ 专家重要性权重")
        print("✅ 训练状态信息")
        print()
        print("🔧 修复加载问题的建议:")
        print("1. 确保新模型与保存时的MoE结构完全一致")
        print("2. 使用 strict=False 参数允许部分加载")
        print("3. 根据具体使用场景选择合适的加载策略")
        print("4. 检查模型名称和配置是否匹配保存时的设置")
        print()
    else:
        print("❌ 无法分析模型文件，请检查文件路径和格式")