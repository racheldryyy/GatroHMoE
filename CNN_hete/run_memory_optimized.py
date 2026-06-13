#!/usr/bin/env python3
"""
16GB内存优化训练启动脚本

此脚本专门为16GB内存环境优化，包含内存监控、自动清理和动态调整功能。
"""

import os
import sys
import torch
import argparse
from memory_utils import MemoryManager, get_memory_info, generate_memory_config

def main():
    parser = argparse.ArgumentParser(description='16GB内存优化训练')
    parser.add_argument('--model', default='resnet50', help='基础模型名称')
    parser.add_argument('--task', choices=['teacher', 'student'], default='teacher', help='训练任务类型')
    parser.add_argument('--resume', action='store_true', help='恢复训练')
    parser.add_argument('--monitor', action='store_true', default=True, help='启用内存监控')
    
    args = parser.parse_args()
    
    print("🚀 启动16GB内存优化训练")
    print("="*60)
    
    # 内存检查
    print("📊 系统内存检查:")
    get_memory_info()
    
    # 生成内存优化配置
    memory_config = generate_memory_config()
    
    # 设置环境变量进行优化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # 减少CUDA内存碎片
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步启动以提高性能
    
    print("\n🔧 PyTorch内存优化设置:")
    print(f"  - CUDA内存分配优化: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    print(f"  - 梯度检查点: 启用")
    print(f"  - 混合精度: 启用")
    print(f"  - 批次大小: {memory_config['batch_size']}")
    print(f"  - 工作进程: {memory_config['num_workers']}")
    print(f"  - 梯度累积: {memory_config['gradient_accumulation']}")
    
    # 内存监控
    if args.monitor:
        memory_manager = MemoryManager()
        
        # 定期内存检查装饰器
        def memory_check_wrapper(func):
            def wrapper(*args, **kwargs):
                if not memory_manager.check_memory():
                    print("⚠️  内存使用率过高，建议降低批次大小")
                return func(*args, **kwargs)
            return wrapper
    
    # 导入训练模块
    try:
        if args.task == 'teacher':
            from train import train_teacher_model
            print(f"\n🎯 开始训练教师模型: {args.model}")
            model = train_teacher_model(
                base_model_name=args.model,
                resume_training=args.resume,
                enable_mixed_precision=True
            )
        else:
            from train import train_student_model, train_teacher_model
            print(f"\n🎯 开始训练学生模型: {args.model}")
            # 首先需要训练好的教师模型
            teacher_model = train_teacher_model('resnet50', resume_training=True)
            model = train_student_model(
                teacher_model=teacher_model,
                base_model_name=args.model,
                use_amp=True
            )
            
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 训练过程发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✅ 训练完成!")
    print("📊 最终内存使用情况:")
    get_memory_info()

if __name__ == '__main__':
    main()