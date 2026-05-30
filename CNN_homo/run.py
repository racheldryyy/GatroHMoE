import os
import argparse
import torch
from config import (
    SEED, DEVICE, GPU_IDS, USE_AMP, COLON_DATA_PATH, UGI_DATA_PATH, 
    COLON_DISEASE_PATH, UGI_DISEASE_PATH, RESULT_PATH, 
    LIGHTWEIGHT_MODELS, CNN_EXPERT_MODELS
)
from utils import set_seed, count_samples
from train import train_teacher_model, train_student_model
from train_rl_hetero_moe import train_rl_hetero_moe as train_hetero_moe
from train_rl_hetero_moe import train_rl_hetero_moe  # 导入新的强化学习训练函数
from evaluate import (
    evaluate_teacher, evaluate_student, evaluate_heterogeneous_moe,
    compare_all_models
)

def check_data_paths():
    """Check if data paths exist"""
    paths = [COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, UGI_DISEASE_PATH]
    path_names = ["Colon site data", "UGI site data", "Colon Disease data", "UGI Disease data"]
    
    all_exist = True
    
    for path, name in zip(paths, path_names):
        if not os.path.exists(path):
            print(f"Error: {name} path does not exist: {path}")
            all_exist = False
    
    return all_exist

def print_dataset_stats():
    """Print dataset statistics"""
    print("Dataset statistics:")
    
    for path, name in zip(
        [COLON_DATA_PATH, UGI_DATA_PATH, COLON_DISEASE_PATH, UGI_DISEASE_PATH],
        ["Colon site", "UGI site", "Colon Disease data", "UGI Disease data"]
    ):
        counts = count_samples(path)
        
        print(f"\n{name} dataset:")
        print("Training set:")
        for class_name, count in counts['train'].items():
            print(f"  - {class_name}: {count} images")
        
        print("Test set:")
        for class_name, count in counts['test'].items():
            print(f"  - {class_name}: {count} images")
    
    print("\n")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Gastrointestinal Disease Diagnosis with MoE Models")
    
    parser.add_argument('--mode', type=str, default='train_teacher',
                        choices=['train_teacher', 'train_student', 'train_hetero_moe',
                                'evaluate_teacher', 'evaluate_student', 'evaluate_hetero_moe',
                                'evaluate_all', 'run_all', 'train_rl_hetero_moe'],  # 添加新模式
                        help='Run mode')
    
    parser.add_argument('--teacher_model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet50', 'vit_b_16'],
                        help='Teacher model architecture')
    
    parser.add_argument('--student_model', type=str, default='mobilenetv3_small',
                        choices=LIGHTWEIGHT_MODELS,
                        help='Student model architecture')
    
    parser.add_argument('--hetero_architectures', nargs='+', default=None,
                        help='Architectures to use in heterogeneous MoE')
    
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    
    # GPU and AMP parameters
    parser.add_argument('--gpus', type=str, default=None,
                        help='GPU id to use (e.g., "0"), defaults to GPU 0')
    parser.add_argument('--amp', action='store_true', default=USE_AMP,
                        help='Use Automatic Mixed Precision')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(SEED)

    # GPU configuration - single GPU mode only
    if args.gpus is not None:
        # User specified GPU(s) - use only the first one
        gpu_list = args.gpus.split(',')
        if len(gpu_list) > 1:
            print(f"Multiple GPUs specified: {args.gpus}, using only first GPU: {gpu_list[0]}")
            gpu_list = [gpu_list[0]]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[0]
        print(f"Using GPU: {gpu_list[0]}")
    else:
        # Auto-detect and use only first GPU
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_list = ["0"]  # Force use only GPU 0
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print(f"Detected {num_gpus} GPU(s), using GPU 0 only (single GPU mode)")
            print(f"Use --gpus parameter to specify different GPU")
        else:
            gpu_list = []
            print("No CUDA GPU detected")
    
    # 重新获取GPU信息（在设置环境变量后）
    num_gpus = torch.cuda.device_count()
    
    if num_gpus > 0:
        DEVICE = torch.device("cuda")
        print(f"Single GPU mode configured")
        
        # Show current GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"   Batch size: {16}")
    else:
        DEVICE = torch.device("cpu")
        print("GPU not available, using CPU (slower training)")
    
    # Check data paths
    if not check_data_paths():
        print("Error: Data path check failed, ensure all datasets exist")
        return
    
    # Print dataset statistics
    print_dataset_stats()
    
    # Run based on mode
    if args.mode == 'train_teacher':
        print(f"Starting teacher model training ({args.teacher_model})...")
        train_teacher_model(args.teacher_model, args.resume, enable_mixed_precision=args.amp)
        torch.cuda.empty_cache()
    
    elif args.mode == 'train_student':
        print(f"Starting student model training ({args.student_model})...")    
        # 首先加载教师模型
        teacher_model, data_loaders = train_teacher_model(args.teacher_model, resume=True, use_amp=args.amp)
        torch.cuda.empty_cache()
        
        # 然后训练学生模型
        train_student_model(teacher_model, args.student_model, use_amp=args.amp)
    
    elif args.mode == 'train_hetero_moe':
        print("Starting heterogeneous MoE training...")
        train_hetero_moe(args.hetero_architectures, args.resume, use_amp=args.amp)
    
    elif args.mode == 'train_rl_hetero_moe':  # 新增强化学习训练模式
        print("Starting RL-enhanced heterogeneous MoE training...")
        train_rl_hetero_moe(args.hetero_architectures, args.resume, use_amp=args.amp)
    
    elif args.mode == 'evaluate_teacher':
        print(f"Evaluating teacher model ({args.teacher_model})...")
        evaluate_teacher(args.teacher_model)
    
    elif args.mode == 'evaluate_student':
        print(f"Evaluating student model ({args.student_model})...")
        evaluate_student(args.student_model)
    
    elif args.mode == 'evaluate_hetero_moe':
        print("Evaluating heterogeneous MoE model...")
        evaluate_heterogeneous_moe()
    
    elif args.mode == 'evaluate_all':
        print("Evaluating all models and comparing performance...")
        evaluate_teacher(args.teacher_model)
        evaluate_student(args.student_model)
        evaluate_heterogeneous_moe()
        compare_all_models()
    
    elif args.mode == 'run_all':
        print("Running complete pipeline...")
        # Train teacher model
        teacher_model, data_loaders = train_teacher_model(args.teacher_model, args.resume, use_amp=args.amp)
        torch.cuda.empty_cache()  # 释放训练教师模型的显存缓存

        # Train student model
        student_model = train_student_model(teacher_model, args.student_model, use_amp=args.amp)
        torch.cuda.empty_cache()  # 释放训练学生模型的显存缓存

        # Train heterogeneous MoE
        hetero_model, _ = train_hetero_moe(args.hetero_architectures, args.resume, use_amp=args.amp)
        torch.cuda.empty_cache()  # 释放训练 MoE 模型的显存缓存

        # Train RL-enhanced heterogeneous MoE (新增)
        rl_hetero_model, _, _ = train_rl_hetero_moe(args.hetero_architectures, args.resume, use_amp=args.amp)
        torch.cuda.empty_cache()  # 释放 RL 增强 MoE 模型的显存缓存
        
        # Evaluate all models and compare
        evaluate_teacher(args.teacher_model)
        evaluate_student(args.student_model)
        evaluate_heterogeneous_moe()
        compare_all_models()

if __name__ == "__main__":
    main()