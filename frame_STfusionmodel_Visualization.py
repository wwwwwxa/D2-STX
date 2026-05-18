import torch
import torch.nn as nn

torch.cuda.empty_cache()
import numpy as np
import os, sys
# from Rep_count_loader import Rep_count
# from Repcount_multishot_loader import Rep_count
from Repcount_fusion_loader import Rep_count
# from Repcountpose_multishot_loader import Rep_count
from Countix_multishot_loader import Countix
from UCFRep_multishot_loader import UCFRep
from tqdm import tqdm
from video_mae_cross_full_attention import SupervisedMAE
from pose_mae_cross_full_attention import SupervisedMAE_p
from frame_spatialtemporal_1111_fusion import SpatioTemporalCrossModalFusion
from Attention_visualizer import AttentionVisualizer
from slowfast.utils.parser import load_config
import timm.optim.optim_factory as optim_factory
import argparse
import wandb
import torch.optim as optim
import math
import random
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
# from util.lr_sched import adjust_learning_rate
import pandas as pd
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib import rc
from thop import profile, clever_format
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(0)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--encodings', default='mae', type=str, help=['swin', 'mae'])
    # 模型参数
    parser.add_argument('--fusion_mode', type=str, default='pose_as_query',
                        choices=['pose_as_query', 'video_as_query', 'bidirectional_gating'],
                        help='融合模式')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='融合模块的特征维度')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--only_test', action='store_true',
                        help='Only testing')
    parser.add_argument('--trained_model', default='', type=str,
                        help='path to a trained model')
    parser.add_argument('--scale_counts', default=100, type=int, help='scaling the counts')

    parser.add_argument('--dataset', default='RepCount', type=str, help='Repcount, Countix, UCFRep')

    parser.add_argument('--get_overlapping_segments', action='store_true', help='whether to get overlapping segments')

    parser.add_argument('--peak_at_random_locations', default=False, type=bool,
                        help='whether to have density peaks at random locations')

    parser.add_argument('--multishot', action='store_true')

    parser.add_argument('--iterative_shots', action='store_true', help='will show the examples one by one')

    parser.add_argument('--density_peak_width', default=0.5, type=float,
                        help='sigma for the peak of density maps, lesser sigma gives sharp peaks')

    # Model parameters
    parser.add_argument('--save_path', default='./saved_models_v+p_repcountfull/1016_p+v_bi', type=str,
                        help="Path to save the model")

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=5e-6, metavar='LR',
                        help='learning rate (peaklr)')
    parser.add_argument('--eval_freq', default=2, type=int)

    # Dataset parameters
    parser.add_argument('--precomputed', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='flag to specify if precomputed tokens will be loaded')
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--slurm_job_id', default=None, type=str,
                        help='job id')
    parser.add_argument('--tokens_dir', default='saved_tokens_reencoded', type=str,
                        help='ground truth density map directory')
    parser.add_argument('--exemplar_dir', default='exemplar_tokens_reencoded', type=str,
                        help='ground truth density map directory')
    parser.add_argument('--pose_tokens_dir', type=str, default='saved_tokens_reencoded',
                        help='Directory for pose tokens')
    parser.add_argument('--pose_exemplar_dir', type=str, default='exemplar_tokens_reencoded',
                        help='Directory for pose exemplar tokens')
    parser.add_argument('--threshold', default=0.0, type=float,
                        help='p, cut off to decide if select exemplar from different video')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Training parameters
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--pretrained_encoder', default='pretrained_models/VIT_B_16x4_MAE_PT.pyth', type=str)

    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--num_gpus', default=2, type=int, help='number of gpus')

    # Logging parameters
    parser.add_argument('--log_dir', default='./logs/260512_STpq_notemp_stage3_1',
                        help='path where to tensorboard log')
    parser.add_argument("--title", default="", type=str)
    parser.add_argument("--use_wandb", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--use_tensorboard", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--wandb", default="", type=str)
    parser.add_argument("--team", default="", type=str)
    parser.add_argument("--wandb_id", default='', type=str)

    parser.add_argument("--token_pool_ratio", default=0.4, type=float)
    parser.add_argument("--rho", default=0.7, type=float)
    parser.add_argument("--window_size", default=(4, 7, 7), type=int, nargs='+',
                        help='window size for windowed self attention')

    # 👇 添加两阶段训练参数
    parser.add_argument('--training_stage', type=int, default=1,
                        choices=[1, 2, 3],
                        help='1: Train video model only, 2: Train pose model only, 3: Train fusion module')
    parser.add_argument('--stage1_epochs', type=int, default=50,
                        help='Number of epochs for stage 1 (video model)')
    parser.add_argument('--stage2_epochs', type=int, default=35,
                        help='Number of epochs for stage 2 (pose model)')
    parser.add_argument('--stage3_epochs', type=int, default=50,
                        help='Number of epochs for stage 3 (fusion module)')
    parser.add_argument('--stage1_checkpoint', type=str, default='',
                        help='Path to stage1 checkpoint for stage2+ training')
    parser.add_argument('--stage2_checkpoint', type=str, default='',
                        help='Path to stage2 checkpoint for stage3 training')
    # 新增融合控制参数
    parser.add_argument('--enable_spatial_fusion', type=bool, default=True,
                        help='是否启用空间融合')
    parser.add_argument('--enable_temporal_fusion', type=bool, default=True,
                        help='是否启用时间融合')
    parser.add_argument('--spatial_fusion_weight', type=float, default=0.6,
                        help='空间融合的初始权重(0-1之间)')
    parser.add_argument('--temporal_fusion_weight', type=float, default=0.5,
                        help='时间融合的初始权重(0-1之间)')
    # 注意力可视化参数
    parser.add_argument('--enable_attention_visualization', action='store_true',
                        help='Enable attention weight visualization during testing')
    parser.add_argument('--attention_save_dir', type=str, default='./attention_visualizations',
                        help='Directory to save attention visualizations')
    parser.add_argument('--visualize_epoch_freq', type=int, default=10,
                        help='Frequency of epochs to visualize attention (e.g., every 10 epochs)')
    return parser


# 在测试部分之前添加计算模型参数和FLOPs的函数
def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_flops(model, model_p, fusion_model, video_data, video_example, pose_data, pose_example,
                    video_thw, pose_thw, video_shot_num, pose_shot_num):
    """计算FLOPs"""
    # 为pose模型计算FLOPs
    flops_pose, _ = profile(model_p, inputs=(pose_data, pose_example, pose_thw, pose_shot_num), verbose=False)

    # 为video模型计算FLOPs
    flops_video, _ = profile(model, inputs=(video_data, video_example, video_thw, video_shot_num), verbose=False)

    # 为fusion模型计算FLOPs (需要先运行一次获取特征维度)
    with torch.no_grad():
        _, pose_feat = model_p(pose_data, pose_example, pose_thw, shot_num=pose_shot_num)
        _, video_feat = model(video_data, video_example, video_thw, shot_num=video_shot_num)

    flops_fusion, _ = profile(fusion_model, inputs=(pose_feat, video_feat), verbose=False)

    total_flops = flops_pose + flops_video + flops_fusion
    return total_flops


def save_density_map(density_map, save_path, filename_prefix):
    """保存密度图为图像"""
    density_map = density_map.squeeze()

    plt.figure(figsize=(15, 4))

    # 如果是1D数据,绘制为折线图
    if len(density_map.shape) == 1:
        plt.plot(density_map, linewidth=2, color='#2E86AB')
        plt.fill_between(range(len(density_map)), density_map, alpha=0.3, color='#2E86AB')
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('Density Value', fontsize=12)
        plt.title(f'{filename_prefix}', fontsize=14, pad=15)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
    else:
        # 如果是2D数据,使用热力图
        plt.imshow(density_map, cmap='jet', aspect='auto')
        plt.colorbar(label='Density Value')
        plt.xlabel('Spatial Dimension', fontsize=12)
        plt.ylabel('Temporal Dimension', fontsize=12)
        plt.title(f'{filename_prefix}', fontsize=14, pad=15)
        plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_comparison_density_map(gt_density, pred_density, save_path, filename, gt_count, pred_count):
    """保存GT和预测密度图的对比图"""
    gt_density = gt_density.squeeze()
    pred_density = pred_density.squeeze()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

    # 如果是1D数据
    if len(gt_density.shape) == 1:
        # Ground Truth
        ax1.plot(gt_density, linewidth=2, color='#2E86AB', label='Ground Truth')
        ax1.fill_between(range(len(gt_density)), gt_density, alpha=0.3, color='#2E86AB')
        ax1.set_xlabel('Frame Index', fontsize=11)
        ax1.set_ylabel('Density Value', fontsize=11)
        ax1.set_title(f'Ground Truth - {filename} (Count: {gt_count:.1f})', fontsize=12, pad=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend()

        # Prediction
        ax2.plot(pred_density, linewidth=2, color='#F18F01', label='Prediction')
        ax2.fill_between(range(len(pred_density)), pred_density, alpha=0.3, color='#F18F01')
        ax2.set_xlabel('Frame Index', fontsize=11)
        ax2.set_ylabel('Density Value', fontsize=11)
        ax2.set_title(f'Predicted - {filename} (Count: {pred_count:.1f})', fontsize=12, pad=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend()
    else:
        # 2D热力图
        im1 = ax1.imshow(gt_density, cmap='jet', aspect='auto')
        plt.colorbar(im1, ax=ax1, label='Density Value')
        ax1.set_xlabel('Spatial Dimension', fontsize=11)
        ax1.set_ylabel('Temporal Dimension', fontsize=11)
        ax1.set_title(f'Ground Truth - {filename} (Count: {gt_count:.1f})', fontsize=12, pad=10)

        im2 = ax2.imshow(pred_density, cmap='jet', aspect='auto')
        plt.colorbar(im2, ax=ax2, label='Density Value')
        ax2.set_xlabel('Spatial Dimension', fontsize=11)
        ax2.set_ylabel('Temporal Dimension', fontsize=11)
        ax2.set_title(f'Predicted - {filename} (Count: {pred_count:.1f})', fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==================== 2. 参数冻结/解冻函数 ====================
def freeze_model(model):
    """冻结模型的所有参数"""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    """解冻模型的所有参数"""
    for param in model.parameters():
        param.requires_grad = True


def print_trainable_params(model, model_name):
    """打印可训练参数数量"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name}: {trainable_params:,} / {total_params:,} trainable")


# ==================== 3. 修改main函数中的优化器初始化 ====================
def setup_training_stage(args, model, model_p, fusion_model):
    """根据训练阶段设置模型和优化器"""

    if args.training_stage == 1:
        print("\n" + "=" * 80)
        print("STAGE 1: Training Video Model Only")
        print("=" * 80)

        # Stage 1: 训练 video，冻结 pose 和 fusion
        unfreeze_model(model)
        freeze_model(model_p)
        freeze_model(fusion_model)

        print_trainable_params(model, "Video Model")
        print_trainable_params(model_p, "Pose Model")
        print_trainable_params(fusion_model, "Fusion Model")

        # 只优化 video 的参数
        param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
        all_params = param_groups

    elif args.training_stage == 2:
        print("\n" + "=" * 80)
        print("STAGE 2: Training Pose Model Only")
        print("=" * 80)

        # Stage 2: 训练 pose，冻结 video 和 fusion
        freeze_model(model)
        unfreeze_model(model_p)
        freeze_model(fusion_model)

        print_trainable_params(model, "Video Model")
        print_trainable_params(model_p, "Pose Model")
        print_trainable_params(fusion_model, "Fusion Model")

        # 只优化 pose 的参数
        param_groups_p = optim_factory.add_weight_decay(model_p, args.weight_decay)
        all_params = param_groups_p

    elif args.training_stage == 3:
        print("\n" + "=" * 80)
        print("STAGE 3: Training Fusion Module Only")
        print("=" * 80)

        # Stage 3: 训练 fusion，冻结 video 和 pose
        freeze_model(model)
        freeze_model(model_p)
        unfreeze_model(fusion_model)

        print_trainable_params(model, "Video Model")
        print_trainable_params(model_p, "Pose Model")
        print_trainable_params(fusion_model, "Fusion Model")

        # 只优化 fusion 的参数
        fusion_params = [{'params': fusion_model.parameters(), 'weight_decay': args.weight_decay}]
        all_params = fusion_params

    else:
        raise ValueError(f"Invalid training_stage: {args.training_stage}")

    # 创建优化器
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, betas=(0.9, 0.95))
    return optimizer


def test_with_attention_visualization(model, model_p, fusion_model, test_loader, args):
    """
    测试函数，保存最好的3个样本的注意力权重并可视化
    完全按照训练代码的数据加载和模型调用方式
    """
    print("\n" + "=" * 80)
    print("开始测试并收集注意力权重...")
    print("=" * 80)

    model.eval()
    model_p.eval()
    fusion_model.eval()

    # 开启注意力权重保存模式
    fusion_model.save_attention_weights = True

    # 初始化可视化器
    visualizer = AttentionVisualizer(save_dir=args.attention_save_dir)

    all_predictions = []
    all_ground_truths = []
    all_errors = []
    all_filenames = []
    all_attention_weights = []

    # 使用与训练相同的格式
    bformat = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'

    with torch.no_grad():
        with tqdm(total=len(test_loader), bar_format=bformat, ascii='░▒█') as pbar:
            for i, item in enumerate(test_loader):
                with torch.cuda.amp.autocast(enabled=True):
                    # ==================== 数据加载（完全按照原代码）====================
                    pose_data = item['pose_data'].cuda().type(torch.cuda.FloatTensor)
                    pose_example = item['pose_example'].cuda().type(torch.cuda.FloatTensor)
                    video_data = item['video_data'].cuda().type(torch.cuda.FloatTensor)
                    video_example = item['video_example'].cuda().type(torch.cuda.FloatTensor)
                    density_map = item['density_map'].cuda().type(torch.cuda.FloatTensor).half() * args.scale_counts
                    actual_counts = item['actual_counts'].cuda()
                    video_name = item['filename']
                    pose_thw = item['pose_thw']
                    video_thw = item['video_thw']
                    pose_shot_num = item['pose_shot_num'][0] if isinstance(item['pose_shot_num'], list) else item[
                        'pose_shot_num']
                    video_shot_num = item['video_shot_num'][0] if isinstance(item['video_shot_num'], list) else item[
                        'video_shot_num']

                    # ==================== 前向传播（完全按照原代码）====================
                    pose_y, pose_feat = model_p(pose_data, pose_example, pose_thw, shot_num=pose_shot_num)
                    video_y, video_feat = model(video_data, video_example, video_thw, shot_num=video_shot_num)

                    # 融合并获取注意力权重
                    y, fusion_info = fusion_model(pose_feat, video_feat)

                    # ==================== 预测计数 ====================
                    predict_count = torch.sum(y, dim=1).type(torch.cuda.FloatTensor) / args.scale_counts

                    # 计算误差
                    error = torch.abs(predict_count - actual_counts).item()

                    # 获取文件名（处理不同格式）
                    if isinstance(video_name, list):
                        filename = video_name[0]
                    elif isinstance(video_name, tuple):
                        filename = video_name[0]
                    else:
                        filename = video_name

                    # 保存结果
                    all_predictions.append(predict_count.cpu().numpy())
                    all_ground_truths.append(actual_counts.cpu().numpy())
                    all_errors.append(error)
                    all_filenames.append(filename)

                    # 保存注意力权重
                    if 'attention_weights' in fusion_info:
                        all_attention_weights.append({
                            'filename': filename,
                            'error': error,
                            'prediction': predict_count.cpu().numpy(),
                            'ground_truth': actual_counts.cpu().numpy(),
                            'weights': fusion_info
                        })

                    # 更新进度条
                    pbar.set_postfix({
                        "Processing": filename[:30] if len(filename) > 30 else filename,
                        "Error": f"{error:.3f}"
                    })
                    pbar.update()

    # 找出预测最好的10个样本（误差最小）
    sorted_indices = np.argsort(all_errors)
    best_10_indices = sorted_indices[:25]

    print("\n" + "=" * 80)
    print("预测最好的3个样本:")
    print("=" * 80)

    for rank, idx in enumerate(best_10_indices, 1):
        filename = all_filenames[idx]
        error = all_errors[idx]
        pred = all_predictions[idx]
        gt = all_ground_truths[idx]

        print(f"\n第{rank}名:")
        print(f"  文件名: {filename}")
        print(f"  真实计数: {gt[0]:.2f}")
        print(f"  预测计数: {pred[0]:.2f}")
        print(f"  绝对误差: {error:.4f}")

        # 找到对应的注意力权重
        attention_data = None
        for item in all_attention_weights:
            if item['filename'] == filename:
                attention_data = item
                break

        if attention_data is None:
            print(f"  警告: 未找到注意力权重数据")
            continue

        print(f"\n  开始可视化注意力权重...")

        attention_weights = attention_data['weights']['attention_weights']  # 从 fusion_info 中获取

        # 1. 可视化空间注意力: Video → Pose
        if 'spatial_video_to_pose' in attention_weights:
            print(f"  [1/4] 可视化 Spatial Video→Pose...")
            visualizer.visualize_spatial_video_to_pose(
                attention_weights['spatial_video_to_pose'],
                filename,
                aggregate_time=True
            )

        # 2. 可视化空间注意力: Pose → Video
        # if 'spatial_pose_to_video' in attention_weights:
        #     print(f"  [2/4] 可视化 Spatial Pose→Video...")
        #     spatial_attn_outputs = None
        #     if 'attn_outputs' in attention_data['weights']:
        #         if 'spatial_pose_to_video' in attention_data['weights']['attn_outputs']:
        #             spatial_attn_outputs = attention_data['weights']['attn_outputs']['spatial_pose_to_video']
        #             print(f"    ✓ Successfully got spatial_attn_outputs, shape: {spatial_attn_outputs.shape}")
        #
        #     visualizer.visualize_spatial_pose_to_video(
        #         attention_weights['spatial_pose_to_video'],
        #         filename,
        #         aggregate_time=True,
        #         attn_outputs=spatial_attn_outputs  # ✅ 传入 attn_outputs
        #     )

        # 3. 可视化时间注意力: Video → Pose
        if 'temporal_video_to_pose' in attention_weights:
            print(f"  [3/4] 可视化 Temporal Video→Pose...")
            visualizer.visualize_temporal_video_to_pose(
                attention_weights['temporal_video_to_pose'],
                filename
            )

        # 4. 可视化时间注意力: Pose → Video
        if 'temporal_pose_to_video' in attention_weights:
            print(f"  [4/4] 可视化 Temporal Pose→Video...")

            # 获取对应的 attn_outputs
            attn_outputs_p2v = None
            if 'attn_outputs' in attention_data['weights']:
                if 'temporal_pose_to_video' in attention_data['weights']['attn_outputs']:
                    attn_outputs_p2v = attention_data['weights']['attn_outputs']['temporal_pose_to_video']
                    print(f"    ✓ Successfully got attn_outputs_p2v, shape: {attn_outputs_p2v.shape}")

            visualizer.visualize_temporal_pose_to_video(
                attention_weights['temporal_pose_to_video'],
                filename,
                attn_outputs=attn_outputs_p2v
            )
            # ✅ 新增：绘制身体部位激活时间线（参考图风格）
            if attn_outputs_p2v is not None:
                print(f"  [4+] 可视化身体部位激活时间线...")
                visualizer.visualize_body_parts_activation_timeline(
                    attn_outputs_p2v,
                    filename,
                    source='temporal'
                )

        # 5. 保存原始权重数据
        # print(f"  保存原始注意力权重数据...")
        # visualizer.save_attention_weights_raw(attention_weights, filename)

        print(f"  ✓ 完成 {filename} 的所有可视化")

    # 关闭注意力权重保存模式
    fusion_model.save_attention_weights = False

    # 计算总体指标
    all_predictions = np.concatenate(all_predictions)
    all_ground_truths = np.concatenate(all_ground_truths)

    mae = np.mean(np.abs(all_predictions - all_ground_truths))
    rmse = np.sqrt(np.mean((all_predictions - all_ground_truths) ** 2))
    obo = np.mean(np.abs(np.round(all_predictions) - np.round(all_ground_truths)) <= 1)
    obz = np.mean(np.abs(np.round(all_predictions) - np.round(all_ground_truths)) == 0)

    print("\n" + "=" * 80)
    print("测试结果:")
    print("=" * 80)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"OBO:  {obo:.4f}")
    print(f"OBZ:  {obz:.4f}")
    print("=" * 80)

    return {
        'mae': mae,
        'rmse': rmse,
        'obo': obo,
        'obz': obz,
        'best_10_filenames': [all_filenames[idx] for idx in best_10_indices]
    }


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    args.window_size = tuple(args.window_size)
    args.opts = None
    g = torch.Generator()
    g.manual_seed(args.seed)
    # 在类初始化或文件开头添加这些列表来跟踪最佳模型
    if not hasattr(args, 'best_obo_checkpoints'):
        args.best_obo_checkpoints = []  # 存储 (obo_value, checkpoint_path) 的列表
    if not hasattr(args, 'best_mae_checkpoints'):
        args.best_mae_checkpoints = []  # 存储 (mae_value, checkpoint_path) 的列表
    cfg = load_config(args, path_to_config='configs/pretrain_config_video.yaml')
    cfg_1 = load_config(args, path_to_config='configs/pretrain_config_pose.yaml')

    if args.precomputed:
        if args.dataset == 'Countix':
            dataset_train = Countix(split="train",
                                    tokens_dir=args.tokens_dir,
                                    exemplar_dir=args.exemplar_dir,
                                    select_rand_segment=False,
                                    compact=True,
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    threshold=args.threshold)

            dataset_valid = Countix(split="valid",
                                    tokens_dir=args.tokens_dir,
                                    exemplar_dir=args.exemplar_dir,
                                    select_rand_segment=False,
                                    compact=True,
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    encodings=args.encodings)
            dataset_test = Countix(split="test",
                                   tokens_dir=args.tokens_dir,
                                   exemplar_dir=args.exemplar_dir,
                                   select_rand_segment=False,
                                   compact=True,
                                   pool_tokens_factor=args.token_pool_ratio,
                                   peak_at_random_location=args.peak_at_random_locations,
                                   get_overlapping_segments=args.get_overlapping_segments,
                                   multishot=args.multishot,
                                   encodings=args.encodings)
        elif args.dataset == 'RepCount':
            dataset_train = Rep_count(split="train",
                                      tokens_dir=args.tokens_dir,
                                      exemplar_dir=args.exemplar_dir,
                                      pose_tokens_dir=args.pose_tokens_dir,
                                      pose_exemplar_dir=args.pose_exemplar_dir,
                                      select_rand_segment=False,
                                      compact=True,
                                      pool_tokens_factor=args.token_pool_ratio,
                                      peak_at_random_location=args.peak_at_random_locations,
                                      get_overlapping_segments=args.get_overlapping_segments,
                                      multishot=args.multishot,
                                      threshold=args.threshold)

            dataset_valid = Rep_count(split="valid",
                                      tokens_dir=args.tokens_dir,
                                      exemplar_dir=args.exemplar_dir,
                                      pose_tokens_dir=args.pose_tokens_dir,
                                      pose_exemplar_dir=args.pose_exemplar_dir,
                                      select_rand_segment=False,
                                      compact=True,
                                      pool_tokens_factor=args.token_pool_ratio,
                                      peak_at_random_location=args.peak_at_random_locations,
                                      get_overlapping_segments=args.get_overlapping_segments,
                                      multishot=args.multishot,
                                      density_peak_width=args.density_peak_width)
            dataset_test = Rep_count(split="test",
                                     tokens_dir=args.tokens_dir,
                                     exemplar_dir=args.exemplar_dir,
                                     pose_tokens_dir=args.pose_tokens_dir,
                                     pose_exemplar_dir=args.pose_exemplar_dir,
                                     select_rand_segment=False,
                                     compact=True,
                                     pool_tokens_factor=args.token_pool_ratio,
                                     peak_at_random_location=args.peak_at_random_locations,
                                     get_overlapping_segments=args.get_overlapping_segments,
                                     multishot=args.multishot,
                                     density_peak_width=args.density_peak_width)

        elif args.dataset == 'UCFRep':
            dataset_train = UCFRep(split="train",
                                   tokens_dir=args.tokens_dir,
                                   exemplar_dir=args.exemplar_dir,
                                   select_rand_segment=False,
                                   compact=True,
                                   pool_tokens_factor=args.token_pool_ratio,
                                   peak_at_random_location=args.peak_at_random_locations,
                                   get_overlapping_segments=args.get_overlapping_segments,
                                   multishot=args.multishot,
                                   threshold=args.threshold)

            dataset_valid = UCFRep(split="valid",
                                   tokens_dir=args.tokens_dir,
                                   exemplar_dir=args.exemplar_dir,
                                   select_rand_segment=False,
                                   compact=True,
                                   pool_tokens_factor=args.token_pool_ratio,
                                   peak_at_random_location=args.peak_at_random_locations,
                                   get_overlapping_segments=args.get_overlapping_segments,
                                   multishot=args.multishot,
                                   density_peak_width=args.density_peak_width)
            dataset_test = UCFRep(split="test",
                                  tokens_dir=args.tokens_dir,
                                  exemplar_dir=args.exemplar_dir,
                                  select_rand_segment=False,
                                  compact=True,
                                  pool_tokens_factor=args.token_pool_ratio,
                                  peak_at_random_location=args.peak_at_random_locations,
                                  get_overlapping_segments=args.get_overlapping_segments,
                                  multishot=args.multishot,
                                  density_peak_width=args.density_peak_width)
        # Create dict of dataloaders for train and val
        dataloaders = {'train': torch.utils.data.DataLoader(dataset_train,
                                                            batch_size=args.batch_size,
                                                            num_workers=args.num_workers,
                                                            shuffle=True,
                                                            pin_memory=False,
                                                            drop_last=False,
                                                            collate_fn=dataset_train.collate_fn,
                                                            worker_init_fn=seed_worker,
                                                            persistent_workers=False,
                                                            generator=g),
                       'val': torch.utils.data.DataLoader(dataset_valid,
                                                          batch_size=args.batch_size,
                                                          num_workers=args.num_workers,
                                                          shuffle=False,
                                                          pin_memory=False,
                                                          drop_last=False,
                                                          collate_fn=dataset_valid.collate_fn,
                                                          worker_init_fn=seed_worker,
                                                          generator=g),
                       'test': torch.utils.data.DataLoader(dataset_test,
                                                           batch_size=1,
                                                           num_workers=args.num_workers,
                                                           shuffle=False,
                                                           pin_memory=False,
                                                           drop_last=False,
                                                           collate_fn=dataset_valid.collate_fn,
                                                           worker_init_fn=seed_worker,
                                                           generator=g)}

    # scaler = torch.cuda.amp.GradScaler() # use mixed percision for efficiency
    # scaler = NativeScaler()

    model = SupervisedMAE(cfg=cfg, use_precomputed=args.precomputed, token_pool_ratio=args.token_pool_ratio,
                          iterative_shots=args.iterative_shots, encodings=args.encodings,
                          window_size=args.window_size).cuda()
    model_p = SupervisedMAE_p(cfg=cfg_1, use_precomputed=args.precomputed, token_pool_ratio=args.token_pool_ratio,
                              iterative_shots=args.iterative_shots, encodings=args.encodings,
                              window_size=args.window_size).cuda()
    # 创建融合模型
    fusion_model = SpatioTemporalCrossModalFusion(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        fusion_mode=args.fusion_mode,
        enable_spatial_fusion=args.enable_spatial_fusion,  # 使用命令行参数
        enable_temporal_fusion=args.enable_temporal_fusion,  # 使用命令行参数
        spatial_fusion_weight=args.spatial_fusion_weight,  # 使用命令行参数
        temporal_fusion_weight=args.temporal_fusion_weight,  # 使用命令行参数
        save_attention_weights = False,
        video_spatial_size=(5, 5)
    ).cuda()

    # ==================== 加载前阶段的检查点 ====================
    if args.training_stage == 2 and args.stage1_checkpoint:
        print(f"\nLoading Stage 1 checkpoint: {args.stage1_checkpoint}")
        checkpoint = torch.load(args.stage1_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Stage 1 checkpoint loaded for Video model")

    if args.training_stage == 3:
        if args.stage1_checkpoint:
            print(f"\nLoading Stage 1 checkpoint: {args.stage1_checkpoint}")
            checkpoint = torch.load(args.stage1_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Stage 1 checkpoint loaded for Video model")

        if args.stage2_checkpoint:
            print(f"Loading Stage 2 checkpoint: {args.stage2_checkpoint}")
            checkpoint = torch.load(args.stage2_checkpoint)
            model_p.load_state_dict(checkpoint['model_p_state_dict'])
            print("✓ Stage 2 checkpoint loaded for Pose model")
    # else:
    #     model = SupervisedMAE(cfg=cfg,use_precomputed=args.precomputed, token_pool_ratio=args.token_pool_ratio, iterative_shots=args.iterative_shots, encodings=args.encodings, no_exemplars=args.no_exemplars).cuda()
    if args.num_gpus > 1:
        model = nn.parallel.DataParallel(model, device_ids=[i for i in range(args.num_gpus)])

    train_step = 0
    val_step = 0

    # ==================== 测试模式 ====================
    if args.only_test:
        print("\n" + "=" * 80)
        print("开始测试模式...")
        print("=" * 80)

        # 加载检查点
        print(f"\n加载检查点: {args.trained_model}")
        if not os.path.exists(args.trained_model):
            raise FileNotFoundError(f"Checkpoint not found: {args.trained_model}")

        checkpoint = torch.load(args.trained_model, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        model_p.load_state_dict(checkpoint['model_p_state_dict'])
        fusion_model.load_state_dict(checkpoint['fusion_layer_state_dict'])
        print("✓ 模型加载完成")

        # 设置为评估模式
        model.eval()
        model_p.eval()
        fusion_model.eval()

        # 运行测试
        if args.enable_attention_visualization:
            # 使用注意力可视化的测试
            print("\n启用注意力权重可视化...")
            results = test_with_attention_visualization(
                model, model_p, fusion_model, dataloaders['test'], args
            )
            print(f"\n✓ 可视化结果已保存到: {args.attention_save_dir}")
            print(f"✓ 最好的10个样本: {results['best_10_filenames']}")
        else:
            # 标准测试（不保存注意力权重）
            print("\n运行标准测试（不保存注意力权重）...")
            # 这里可以调用你原来的测试代码
            # 或者也可以调用 test_with_attention_visualization，只是 fusion_model.save_attention_weights 已经是 False
            pass

        return

    # ==================== 训练模式 ====================
    if args.use_wandb:
        wandb_run = wandb.init(
            config=args,
            resume="allow",
            project=args.wandb,
            entity=args.team,
            id=f"{args.wandb_id}_{args.dataset}_{args.encodings}_{args.lr}_{args.threshold}",
        )

    # 在训练开始前初始化writer（通常在main函数或训练循环开始处）
    if args.use_tensorboard:  # 建议将参数名改为use_tensorboard
        writer = SummaryWriter(log_dir=args.log_dir)  # args.log_dir是你想保存tensorboard日志的路径
    optimizer = setup_training_stage(args, model, model_p, fusion_model)
    stage_epochs = {1: args.stage1_epochs, 2: args.stage2_epochs, 3: args.stage3_epochs}
    num_epochs = stage_epochs[args.training_stage]
    # param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    # param_groups_p = optim_factory.add_weight_decay(model_p, args.weight_decay)
    # fusion_params = [{'params': fusion_model.parameters(), 'weight_decay': args.weight_decay}]
    # all_params = param_groups + param_groups_p + fusion_params
    # optimizer = torch.optim.AdamW(all_params, lr=args.lr, betas=(0.9, 0.95))
    milestones = [i for i in range(0, num_epochs, 60)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones,
                                               gamma=0.8)  ### reduce learning rate by 0.8 every 60 epochs
    lossMSE = nn.MSELoss().cuda()
    lossSL1 = nn.SmoothL1Loss().cuda()
    best_loss = np.inf

    os.makedirs(args.save_path, exist_ok=True)
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        scheduler.step()

        print(f"Epoch: {epoch:02d} | Stage: {args.training_stage}")

        for phase in ['train', 'val']:
            if phase == 'val':
                if epoch % args.eval_freq != 0:
                    continue
                model.eval()
                model_p.eval()
                fusion_model.eval()
                ground_truth = list()
                predictions = list()
            else:
                model.train()
                model_p.train()
                fusion_model.train()

            with torch.set_grad_enabled(phase == 'train'):
                total_loss_all = 0
                total_loss1 = 0
                total_loss2 = 0
                total_loss3 = 0
                off_by_zero = 0
                off_by_one = 0
                mse = 0
                count = 0
                mae = 0

                bformat = '{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}'
                dataloader = dataloaders[phase]

                with tqdm(total=len(dataloader), bar_format=bformat, ascii='░▒█') as pbar:
                    for i, item in enumerate(dataloader):
                        with torch.cuda.amp.autocast(enabled=True):
                            # 数据加载
                            pose_data = item['pose_data'].cuda().type(torch.cuda.FloatTensor)
                            pose_example = item['pose_example'].cuda().type(torch.cuda.FloatTensor)
                            video_data = item['video_data'].cuda().type(torch.cuda.FloatTensor)
                            video_example = item['video_example'].cuda().type(torch.cuda.FloatTensor)

                            density_map = item['density_map'].cuda().type(
                                torch.cuda.FloatTensor).half() * args.scale_counts
                            actual_counts = item['actual_counts'].cuda()
                            pose_thw = item['pose_thw']
                            video_thw = item['video_thw']
                            pose_shot_num = item['pose_shot_num'][0] if isinstance(item['pose_shot_num'], list) else \
                                item['pose_shot_num']
                            video_shot_num = item['video_shot_num'][0] if isinstance(item['video_shot_num'], list) else \
                                item['video_shot_num']

                            # ==================== 前向传播 ====================
                            pose_y, pose_feat = model_p(pose_data, pose_example, pose_thw, shot_num=pose_shot_num)
                            video_y, video_feat = model(video_data, video_example, video_thw, shot_num=video_shot_num)
                            # 所有阶段都使用融合模块进行预测（只是参数冻结/解冻不同）
                            y, fusion_info = fusion_model(pose_feat, video_feat)

                            # 随机掩码
                            if phase == 'train':
                                mask = np.random.binomial(n=1, p=0.8, size=[1, density_map.shape[1]])
                            else:
                                mask = np.ones([1, density_map.shape[1]])

                            masks = np.tile(mask, (density_map.shape[0], 1))
                            masks = torch.from_numpy(masks).cuda()

                            # 损失计算
                            loss = ((y - density_map) ** 2)
                            loss = ((loss * masks) / density_map.shape[1]).sum() / density_map.shape[0]

                            predict_count = torch.sum(y, dim=1).type(torch.cuda.FloatTensor) / args.scale_counts

                            if phase == 'val':
                                ground_truth.append(actual_counts.detach().cpu().numpy())
                                predictions.append(predict_count.detach().cpu().numpy())

                            loss2 = lossSL1(predict_count, actual_counts)
                            loss3 = torch.sum(torch.div(torch.abs(predict_count - actual_counts),
                                                        actual_counts + 1e-1)) / predict_count.flatten().shape[0]

                            if phase == 'train':
                                loss1 = (loss + 1.0 * loss3) / args.accum_iter
                                loss1.backward()

                                if (i + 1) % args.accum_iter == 0:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                    torch.nn.utils.clip_grad_norm_(model_p.parameters(), max_norm=1.0)
                                    torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)

                                    optimizer.step()
                                    optimizer.zero_grad()
                                    torch.cuda.empty_cache()

                            # 指标计算
                            b = pose_data.shape[0]
                            count += b
                            total_loss_all += loss.item() * b
                            total_loss1 += loss.item() * b
                            total_loss2 += loss2.item() * b
                            total_loss3 += loss3.item() * b
                            off_by_zero += (torch.abs(actual_counts.round() - predict_count.round()) == 0).sum().item()
                            off_by_one += (torch.abs(actual_counts.round() - predict_count.round()) <= 1).sum().item()
                            mse += ((actual_counts - predict_count.round()) ** 2).sum().item()
                            mae += torch.sum(torch.div(torch.abs(predict_count.round() - actual_counts),
                                                       actual_counts + 1e-1)).item()

                            pbar.set_postfix({
                                "LOSS": f"{total_loss_all / count:.2f}",
                                "MAE": f"{mae / count:.2f}",
                                "OBZ": f"{off_by_zero / count:.2f}",
                                "OBO": f"{off_by_one / count:.2f}",
                                "RMSE": f"{np.sqrt(mse / count):.3f}"
                            })
                            pbar.update()

                # TensorBoard 日志
                if args.use_tensorboard:
                    stage_prefix = f'Stage{args.training_stage}'
                    if phase == 'train':
                        writer.add_scalar(f'{stage_prefix}/Loss/train', total_loss_all / float(count), epoch)
                        writer.add_scalar(f'{stage_prefix}/Metrics/train_mae', mae / count, epoch)
                    else:
                        writer.add_scalar(f'{stage_prefix}/Loss/val', total_loss_all / float(count), epoch)
                        writer.add_scalar(f'{stage_prefix}/Metrics/val_mae', mae / count, epoch)
                        writer.add_scalar(f'{stage_prefix}/Metrics/val_obo', off_by_one / count, epoch)
                        writer.add_scalar(f'{stage_prefix}/Metrics/val_obz', off_by_zero / count, epoch)

                # ==================== 保存最佳检查点 ====================
                if phase == 'val' and epoch % args.eval_freq == 0:
                    stage_name = f'stage{args.training_stage}'

                    current_obo = off_by_one / count
                    current_mae = mae / count
                    current_obz = off_by_zero / count
                    current_rmse = np.sqrt(mse / count)

                    # 维护 OBO 最佳的 2 个模型
                    current_obo_path = os.path.join(
                        args.save_path,
                        f'best_obo_{current_obo:.3f}_mae{current_mae:.3f}_obz{current_obz:.3f}_rmse{current_rmse:.3f}_{stage_name}_epoch{epoch:03d}.pyth'
                    )
                    torch.save({
                        'epoch': epoch,
                        'stage': args.training_stage,
                        'model_state_dict': model.state_dict(),
                        'model_p_state_dict': model_p.state_dict(),
                        'fusion_layer_state_dict': fusion_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, current_obo_path)

                    args.best_obo_checkpoints.append((current_obo, current_obo_path))
                    args.best_obo_checkpoints.sort(key=lambda x: x[0], reverse=True)

                    if len(args.best_obo_checkpoints) > 2:
                        _, worst_obo_path = args.best_obo_checkpoints.pop()
                        if os.path.exists(worst_obo_path):
                            os.remove(worst_obo_path)

                    # 维护 MAE 最佳的 2 个模型
                    current_mae_path = os.path.join(
                        args.save_path,
                        f'best_mae_{current_mae:.3f}_obo{current_obo:.3f}_obz{current_obz:.3f}_rmse{current_rmse:.3f}_{stage_name}_epoch{epoch:03d}.pyth'
                    )
                    torch.save({
                        'epoch': epoch,
                        'stage': args.training_stage,
                        'model_state_dict': model.state_dict(),
                        'model_p_state_dict': model_p.state_dict(),
                        'fusion_layer_state_dict': fusion_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, current_mae_path)

                    args.best_mae_checkpoints.append((current_mae, current_mae_path))
                    args.best_mae_checkpoints.sort(key=lambda x: x[0])

                    if len(args.best_mae_checkpoints) > 2:
                        _, worst_mae_path = args.best_mae_checkpoints.pop()
                        if os.path.exists(worst_mae_path):
                            os.remove(worst_mae_path)

    if args.use_tensorboard:
        writer.close()

    if args.use_wandb:
        wandb_run.finish()

    print("\n" + "=" * 80)
    print(f"Stage {args.training_stage} Training Completed!")
    print("=" * 80)

    # 打印下一步命令
    if args.training_stage == 1:
        print(f"\nNext: Train Stage 2 (Pose Model)")
        print(f"Command:")
        print(f"  python train.py \\")
        print(f"    --training_stage 2 \\")
        print(f"    --stage1_checkpoint {args.save_path}/stage1_checkpoint.pyth \\")
        print(f"    --stage2_epochs 50")

    elif args.training_stage == 2:
        print(f"\nNext: Train Stage 3 (Fusion Module)")
        print(f"Command:")
        print(f"  python train.py \\")
        print(f"    --training_stage 3 \\")
        print(f"    --stage1_checkpoint {args.save_path}/stage1_checkpoint.pyth \\")
        print(f"    --stage2_checkpoint {args.save_path}/stage2_checkpoint.pyth \\")
        print(f"    --stage3_epochs 50")

    elif args.training_stage == 3:
        print(f"\nAll stages completed successfully!")
        print(f"Final checkpoint: {args.save_path}/stage3_checkpoint.pyth")


if __name__ == '__main__':
    main()
