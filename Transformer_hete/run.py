import os
import argparse
import torch
from config import (
    SEED, DEVICE, GPU_IDS, USE_AMP, COLON_DATA_PATH, UGI_DATA_PATH, 
    COLON_DISEASE_PATH, UGI_DISEASE_PATH, RESULT_PATH, 
    LIGHTWEIGHT_MODELS, HETERO_ARCHITECTURES
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
        print("训练集:")
        for class_name, count in counts['train'].items():
            print(f"  - {class_name}: {count} 张图像")
        
        print("测试集:")
        for class_name, count in counts['test'].items():
            print(f"  - {class_name}: {count} 张图像")
    
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
    
    # 添加GPU和AMP参数
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='GPU ids to use (e.g., "0,1")')
    parser.add_argument('--amp', action='store_true', default=USE_AMP,
                        help='Use Automatic Mixed Precision')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(SEED)

    # 显式设置使用 GPU (包括用于显示的）
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    num_gpus = torch.cuda.device_count()
    
    if num_gpus > 0:
        DEVICE = torch.device("cuda")
        print(f"使用 {num_gpus} 张 GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        DEVICE = torch.device("cpu")
        print("警告: 无可用GPU，使用CPU训练")
    
    # Check data paths
    if not check_data_paths():
        print("Error: Data path check failed, ensure all datasets exist")
        return
    
    # Print dataset statistics
    print_dataset_stats()
    
    # Run based on mode
    if args.mode == 'train_teacher':
        print(f"Starting teacher model training ({args.teacher_model})...")
        train_teacher_model(args.teacher_model, args.resume, use_amp=args.amp)
    
    elif args.mode == 'train_student':
        print(f"Starting student model training ({args.student_model})...")    
        # 首先加载教师模型
        teacher_model, data_loaders = train_teacher_model(args.teacher_model, resume=True, use_amp=args.amp)
        
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
        evaluate_heterogeneous_moe(args.hetero_architectures)
    
    elif args.mode == 'evaluate_all':
        print("Evaluating all models and comparing performance...")
        evaluate_teacher(args.teacher_model)
        evaluate_student(args.student_model)
        evaluate_heterogeneous_moe(args.hetero_architectures)
        compare_all_models()
    
    elif args.mode == 'run_all':
        print("Running complete pipeline...")
        # Train teacher model
        teacher_model, data_loaders = train_teacher_model(args.teacher_model, args.resume, use_amp=args.amp)
        
        # Train student model
        student_model = train_student_model(teacher_model, args.student_model, use_amp=args.amp)
        
        # Train heterogeneous MoE
        hetero_model, _ = train_hetero_moe(args.hetero_architectures, args.resume, use_amp=args.amp)
        
        # Train RL-enhanced heterogeneous MoE (新增)
        rl_hetero_model, _, _ = train_rl_hetero_moe(args.hetero_architectures, args.resume, use_amp=args.amp)
        
        # Evaluate all models and compare
        evaluate_teacher(args.teacher_model)
        evaluate_student(args.student_model)
        evaluate_heterogeneous_moe(args.hetero_architectures)
        compare_all_models()

if __name__ == "__main__":
    main()