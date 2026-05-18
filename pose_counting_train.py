import torch
import torch.nn as nn
torch.cuda.empty_cache()
import numpy as np
import os, sys
# from Rep_count_loader import Rep_count
# from Repcount_multishot_loader import Rep_count
# from Repcount_fusion_loader import Rep_count
from Repcountpose_multishot_loader import Rep_count
from Countix_multishot_loader import Countix
from UCFRep_multishot_loader import UCFRep
from tqdm import tqdm
from pose_mae_cross_full_attention import SupervisedMAE_p
from slowfast.utils.parser import load_config
import timm.optim.optim_factory as optim_factory
import argparse
import wandb
import torch.optim as optim
import torch.nn.functional as F
import math
import random
import time
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
# from util.lr_sched import adjust_learning_rate
import pandas as pd
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from scipy.signal import find_peaks
import matplotlib.pyplot as plt 
from matplotlib import rc
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

torch.manual_seed(0)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--encodings', default='mae', type=str, help=['swin','mae'])
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--only_test', action='store_true',
                        help='Only testing')
    parser.add_argument('--trained_model', default='', type=str,
                        help='path to a trained model')
    parser.add_argument('--scale_counts', default=100, type=int, help='scaling the counts')

    parser.add_argument('--dataset', default='RepCount', type=str, help='Repcount, Countix, UCFRep')

    parser.add_argument('--get_overlapping_segments', action='store_true', help='whether to get overlapping segments')

    parser.add_argument('--peak_at_random_locations', default=False, type=bool, help='whether to have density peaks at random locations')

    parser.add_argument('--multishot', action='store_true')


    parser.add_argument('--iterative_shots', action='store_true', help='will show the examples one by one')


    parser.add_argument('--density_peak_width', default=0.5, type=float, help='sigma for the peak of density maps, lesser sigma gives sharp peaks')


    # Model parameters
    parser.add_argument('--save_path', default='./saved_models_video_repcountfull', type=str, help="Path to save the model")

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
    parser.add_argument('--log_dir', default='./logs/1021_posedir',
                        help='path where to tensorboard log')
    parser.add_argument("--title", default="", type=str)
    parser.add_argument("--use_wandb", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--use_tensorboard", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--wandb", default="", type=str)
    parser.add_argument("--team", default="", type=str)
    parser.add_argument("--wandb_id", default='', type=str)
    
    parser.add_argument("--token_pool_ratio", default=0.4, type=float)
    parser.add_argument("--rho", default=0.7, type=float)
    parser.add_argument("--window_size", default=(4,7,7), type=int, nargs='+', help='window size for windowed self attention')

    return parser


# 在模型初始化部分添加融合参数
class MultiModalFusionLayer(nn.Module):
    def __init__(self):
        super(MultiModalFusionLayer, self).__init__()
        # 可学习的融合权重参数
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))  # 初始化为0.5
        self.sigmoid = nn.Sigmoid()

    def forward(self, video_pred, pose_pred):
        # 使用sigmoid确保权重在0-1之间
        alpha = self.sigmoid(self.fusion_weight)
        # 加权融合
        fused_pred = alpha * video_pred + (1 - alpha) * pose_pred
        return fused_pred

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
    cfg = load_config(args, path_to_config='configs/pretrain_config.yaml')
    
    '''
    create dataloaders
    '''
    if args.precomputed:
        if args.dataset == 'Countix':
            dataset_train = Countix(split="train",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    encodings=args.encodings,
                                    threshold=args.threshold)
            
            dataset_valid = Countix(split="val",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    encodings=args.encodings)
            dataset_test = Countix(split="test",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    encodings=args.encodings)
        elif args.dataset == 'RepCount':
            dataset_train = Rep_count(split="train",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    # pose_tokens_dir=args.pose_tokens_dir,
                                    # pose_exemplar_dir=args.pose_exemplar_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    threshold=args.threshold)
            
            dataset_valid = Rep_count(split="valid",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    # pose_tokens_dir=args.pose_tokens_dir,
                                    # pose_exemplar_dir=args.pose_exemplar_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    density_peak_width = args.density_peak_width)
            dataset_test = Rep_count(split="test",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    select_rand_segment=False,
                                    compact=True,
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    density_peak_width = args.density_peak_width)

        elif args.dataset == 'UCFRep':
            dataset_train = UCFRep(split="train",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    threshold=args.threshold)
            
            dataset_valid = UCFRep(split="valid",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    density_peak_width = args.density_peak_width)
            dataset_test = UCFRep(split="test",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    density_peak_width = args.density_peak_width)
        # Create dict of dataloaders for train and val
        dataloaders = {'train':torch.utils.data.DataLoader(dataset_train,
                                                           batch_size=args.batch_size,
                                                           num_workers=args.num_workers,
                                                           shuffle=True,
                                                           pin_memory=False,
                                                           drop_last=False,
                                                           collate_fn=dataset_train.collate_fn,
                                                           worker_init_fn=seed_worker,
                                                           persistent_workers=False,
                                                           generator=g),
                       'val':torch.utils.data.DataLoader(dataset_valid,
                                                         batch_size=args.batch_size,
                                                         num_workers=args.num_workers,
                                                         shuffle=False,
                                                         pin_memory=False,
                                                         drop_last=False,
                                                         collate_fn=dataset_valid.collate_fn,
                                                         worker_init_fn=seed_worker,
                                                         generator=g),
                        'test':torch.utils.data.DataLoader(dataset_test,
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
    # 初始化融合层
    fusion_layer = MultiModalFusionLayer().cuda()
    model = SupervisedMAE_p(cfg=cfg,use_precomputed=args.precomputed, token_pool_ratio=args.token_pool_ratio, iterative_shots=args.iterative_shots, encodings=args.encodings, window_size=args.window_size).cuda()
    # else:
    #     model = SupervisedMAE(cfg=cfg,use_precomputed=args.precomputed, token_pool_ratio=args.token_pool_ratio, iterative_shots=args.iterative_shots, encodings=args.encodings, no_exemplars=args.no_exemplars).cuda()
    if args.num_gpus > 1:
        model = nn.parallel.DataParallel(model, device_ids=[i for i in range(args.num_gpus)])
    
    train_step = 0
    val_step = 0
    if args.only_test:  #### only for testing
        model.load_state_dict(torch.load(args.trained_model)['model_state_dict'])  ### load trained model
        videos = []
        loss = []
        model.eval()
        print(f"Testing")

        # ========== 参数量统计 ==========
        def count_parameters(model):
            """统计可训练参数量"""
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        total_params = count_parameters(model)
        print(f"\n{'=' * 60}")
        print(f"Model Parameters:")
        print(f"  Total: {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f"{'=' * 60}\n")

        dataloader = dataloaders['test']
        gt_counts = list()
        predictions = list()
        predict_mae = list()
        predict_mse = list()
        clips = list()
        inference_times = []  # 存储所有推理时间

        # ========== 添加：用于保存所有样本的列表 ==========
        all_samples = []  # 存储所有样本信息

        # ========== 添加：密度图可视化函数 ==========
        def save_comparison_density_map(gt_density, pred_density, save_path, filename, gt_count, pred_count):
            """
            保存GT和预测密度图的对比图 - 矩形颜色条形式
            使用绿色colormap，上下两行分别显示ground truth和prediction
            """
            # 转换为numpy数组
            gt_density = gt_density.squeeze().cpu().numpy() if torch.is_tensor(gt_density) else gt_density.squeeze()
            pred_density = pred_density.squeeze().cpu().numpy() if torch.is_tensor(
                pred_density) else pred_density.squeeze()

            # 确保是1D数据
            if len(gt_density.shape) > 1:
                gt_density = gt_density.flatten()
                pred_density = pred_density.flatten()

            # 归一化到0-1范围以便更好地显示
            gt_max = gt_density.max() if gt_density.max() > 0 else 1.0
            pred_max = pred_density.max() if pred_density.max() > 0 else 1.0
            max_val = max(gt_max, pred_max)

            gt_normalized = gt_density / max_val
            pred_normalized = pred_density / max_val

            # 创建2D数组用于显示 (增加高度使得条带更明显)
            bar_height = 10
            gt_bar = np.tile(gt_normalized, (bar_height, 1))
            pred_bar = np.tile(pred_normalized, (bar_height, 1))

            # 创建图形
            fig = plt.figure(figsize=(16, 4), constrained_layout=True)

            # 创建自定义的绿色colormap - 从浅到深的绿色
            cmap_green = plt.cm.Greens

            # Ground Truth (上半部分)
            ax1 = plt.subplot(2, 1, 1)
            im1 = ax1.imshow(gt_bar, cmap=cmap_green, aspect='auto',
                             vmin=0, vmax=1, interpolation='bilinear')
            ax1.set_title(f'Ground Truth - {filename} (Count: {gt_count:.1f})',
                          fontsize=13, pad=10, fontweight='bold')
            ax1.set_ylabel('Ground\nTruth', fontsize=11, rotation=0, ha='right', va='center')
            ax1.set_yticks([])
            ax1.set_xticks([])

            # Prediction (下半部分)
            ax2 = plt.subplot(2, 1, 2)
            im2 = ax2.imshow(pred_bar, cmap=cmap_green, aspect='auto',
                             vmin=0, vmax=1, interpolation='bilinear')
            ax2.set_title(f'Prediction (Count: {pred_count:.1f}, Error: {abs(pred_count - gt_count):.2f})',
                          fontsize=13, pad=10, fontweight='bold')
            ax2.set_ylabel('Prediction', fontsize=11, rotation=0, ha='right', va='center')
            ax2.set_xlabel('Frame Index', fontsize=11)
            ax2.set_yticks([])

            # 设置x轴刻度
            n_frames = len(gt_density)
            if n_frames <= 50:
                tick_interval = 5
            elif n_frames <= 100:
                tick_interval = 10
            else:
                tick_interval = 20

            tick_positions = list(range(0, n_frames, tick_interval))
            ax2.set_xticks(tick_positions)
            ax2.set_xticklabels(tick_positions)

            # 添加colorbar
            cbar = plt.colorbar(im2, ax=[ax1, ax2], orientation='vertical',
                                shrink=0.8, aspect=30)
            cbar.set_label('Normalized Density Value', fontsize=10)

            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()

        # ========== 开始测试循环 ==========
        bformat = '{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}'
        with tqdm(total=len(dataloader), bar_format=bformat, ascii='░▒█') as pbar:
            for i, item in enumerate(dataloader):
                if args.get_overlapping_segments:
                    data, data2 = item[0][0], item[0][1]
                else:
                    data = item[0].cuda().type(torch.cuda.FloatTensor)  # B x (THW) x C

                example = item[1].cuda().type(torch.cuda.FloatTensor)  # B x (THW) x C
                density_map = item[2].cuda().type(torch.cuda.FloatTensor).half() * args.scale_counts
                actual_counts = item[3].cuda()  # B x 1
                video_name = item[4]

                videos.append(video_name[0])

                shot_num = item[6][0]
                b, n, c = data.shape

                thw = item[5]

                with torch.no_grad():
                    if args.get_overlapping_segments:
                        data = data.cuda().type(torch.cuda.FloatTensor)
                        data2 = data2.cuda().type(torch.cuda.FloatTensor)

                        # 测量推理时间（重叠片段）
                        if i >= 5:  # 跳过前5个batch预热
                            torch.cuda.synchronize()
                            start_time = time.time()

                        pred1, feat_v = model(data, example, thw, shot_num=shot_num)
                        pred2, feat_v = model(data2, example, thw, shot_num=shot_num)

                        if i >= 5:
                            torch.cuda.synchronize()
                            end_time = time.time()
                            inference_times.append((end_time - start_time) * 1000)  # ms

                        if pred1.shape != pred2.shape:
                            pred2 = torch.cat([torch.zeros(1, 4).cuda(), pred2], 1)
                        else:
                            print('equal')
                        pred = (pred1 + pred2) / 2

                    else:
                        # 测量推理时间（单片段）
                        if i >= 5:  # 跳过前5个batch预热
                            torch.cuda.synchronize()
                            start_time = time.time()

                        pred, feat_v = model(data, example, thw, shot_num=shot_num)

                        if i >= 5:
                            torch.cuda.synchronize()
                            end_time = time.time()
                            inference_times.append((end_time - start_time) * 1000)  # ms

                # 计算损失和指标
                mse = ((pred - density_map) ** 2).mean(-1)
                predict_counts = torch.sum(pred, dim=1).type(torch.FloatTensor).cuda() / args.scale_counts
                predict_counts = predict_counts.round()
                predictions.extend(predict_counts.detach().cpu().numpy())
                gt_counts.extend(actual_counts.detach().cpu().numpy())
                mae = torch.div(torch.abs(predict_counts - actual_counts), actual_counts + 1e-1)
                predict_mae.extend(mae.cpu().numpy())
                predict_mse.extend(np.sqrt(mse.cpu().numpy()))
                loss.append(mse.cpu().numpy())

                # ========== 收集所有样本信息 ==========
                for j in range(b):  # 遍历batch中的每个样本
                    sample_mse = mse[j].item()
                    sample_video_name = video_name[j] if isinstance(video_name, (list, tuple)) else video_name
                    sample_gt_count = actual_counts[j].item()
                    sample_pred_count = predict_counts[j].item()
                    sample_density_map = density_map[j].detach().cpu()
                    sample_pred = pred[j].detach().cpu()
                    sample_feat_v = feat_v[j].detach().cpu() if feat_v is not None else None
                    sample_mae = mae[j].item()

                    all_samples.append({
                        'mse': sample_mse,
                        'mae': sample_mae,
                        'video_name': sample_video_name,
                        'gt_count': sample_gt_count,
                        'pred_count': sample_pred_count,
                        'density_map': sample_density_map,
                        'pred': sample_pred,
                        'feat_v': sample_feat_v,
                        'error': abs(sample_pred_count - sample_gt_count)
                    })

                pbar.update()

        # ========== 计算最终指标 ==========
        predict_mae = np.array(predict_mae)
        predictions = np.array(predictions).round()
        gt_counts = np.array(gt_counts)
        predict_mse = np.array(predict_mse)
        diff = np.abs(predictions.round() - gt_counts)
        diff_z = np.abs(predictions.round() - gt_counts.round())

        # ========== 推理时间统计 ==========
        if len(inference_times) > 0:
            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)
            min_inference_time = np.min(inference_times)
            max_inference_time = np.max(inference_times)
            fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0

            print(f"\n{'=' * 60}")
            print(f"Inference Time Statistics (excluding first 5 warm-up batches):")
            print(f"  Average: {avg_inference_time:.2f} ms ± {std_inference_time:.2f} ms")
            print(f"  Min:     {min_inference_time:.2f} ms")
            print(f"  Max:     {max_inference_time:.2f} ms")
            print(f"  FPS:     {fps:.2f}")
            print(f"  Samples: {len(inference_times)}")
            print(f"{'=' * 60}\n")
        else:
            print("\n⚠️  No inference time data collected (dataset too small)\n")
            avg_inference_time = None

        # ========== 打印测试结果 ==========
        print(f"\n{'=' * 60}")
        print(f"Test Results:")
        print(f"  Overall MAE:  {predict_mae.mean():.4f}")
        print(f"  OBO (≤1):     {(diff <= 1).sum() / len(diff):.4f}")
        print(f"  OBZ (=0):     {(diff_z == 0).sum() / len(diff):.4f}")
        print(f"  RMSE:         {np.sqrt((diff ** 2).mean()):.4f}")
        print(f"{'=' * 60}\n")

        # ========== 汇总统计 ==========
        print(f"\n{'=' * 60}")
        print(f"Model Summary:")
        print(f"  Parameters:     {total_params / 1e6:.2f}M")
        if avg_inference_time is not None:
            print(f"  Inference Time: {avg_inference_time:.2f} ms")
            print(f"  FPS:            {fps:.2f}")
        print(f"  MAE:            {predict_mae.mean():.4f}")
        print(f"  RMSE:           {np.sqrt((diff ** 2).mean()):.4f}")
        print(f"{'=' * 60}\n")

        # ========== 新增：可视化所有样本的密度图 ==========
        print(f"\n{'=' * 60}")
        print("Generating Density Map Visualizations for ALL Samples...")
        print(f"{'=' * 60}\n")

        # 创建密度图保存目录
        vis_dir = os.path.join(os.path.dirname(args.trained_model), 'density_visualizations_20260130')
        os.makedirs(vis_dir, exist_ok=True)

        # 按误差分类保存所有样本
        print("1. 按误差值分类所有样本...")

        # 按error升序排序（最好到最差）
        all_samples_sorted = sorted(all_samples, key=lambda x: x['error'])

        # 定义误差范围分类
        error_ranges = [
            (0, 0.5, 'excellent'),  # 优秀：误差 < 0.5
            (0.5, 1.0, 'good'),  # 良好：误差 0.5-1.0
            (1.0, 2.0, 'moderate'),  # 一般：误差 1.0-2.0
            (2.0, float('inf'), 'poor')  # 差：误差 >= 2.0
        ]

        # 计算各分类的样本数
        error_distribution = {category: 0 for _, _, category in error_ranges}
        for sample in all_samples_sorted:
            error = sample['error']
            for min_err, max_err, category in error_ranges:
                if min_err <= error < max_err:
                    error_distribution[category] += 1
                    break

        print("   误差分布统计：")
        for min_err, max_err, category in error_ranges:
            count = error_distribution[category]
            percentage = (count / len(all_samples_sorted)) * 100
            print(f"   - {category:10s} (error < {max_err:6.1f}): {count:4d} 样本 ({percentage:5.1f}%)")
        print()

        # 2. 生成所有样本的密度图
        print("2. 生成所有样本的密度图...")
        print(f"   总样本数: {len(all_samples_sorted)}")

        # 创建总体目录
        all_samples_dir = os.path.join(vis_dir, 'all_samples')
        os.makedirs(all_samples_dir, exist_ok=True)

        # 使用进度条显示生成进度
        bformat_vis = '{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}'

        with tqdm(total=len(all_samples_sorted), bar_format=bformat_vis, ascii='░▒█',
                  desc='   Generating visualizations') as pbar:
            for rank, sample in enumerate(all_samples_sorted, 1):
                video_name = sample['video_name']
                gt_count = sample['gt_count']
                pred_count = sample['pred_count']
                error = sample['error']

                # 确定分类
                category = 'poor'
                for min_err, max_err, cat in error_ranges:
                    if min_err <= error < max_err:
                        category = cat
                        break

                # 清理文件名
                clean_name = os.path.splitext(video_name)[0].replace('/', '_').replace('\\', '_')

                # 保存路径：rank_序号_分类_样本名_密度图.png
                save_path = os.path.join(
                    all_samples_dir,
                    f'rank_{rank:04d}_error{error:.3f}_{category}_{clean_name}_density_map.png'
                )

                # 保存密度图
                save_comparison_density_map(
                    gt_density=sample['density_map'],
                    pred_density=sample['pred'],
                    save_path=save_path,
                    filename=f"#{rank} - {video_name}",
                    gt_count=gt_count,
                    pred_count=pred_count
                )

                pbar.update()

        # 3. 按分类创建子目录和符号链接（便于查看）
        print(f"\n3. 按误差分类组织样本...")

        # 创建分类目录
        category_dirs = {}
        for min_err, max_err, category in error_ranges:
            cat_dir = os.path.join(vis_dir, f'by_category_{category}')
            os.makedirs(cat_dir, exist_ok=True)
            category_dirs[category] = cat_dir

        # 遍历所有样本，按分类创建组织
        for rank, sample in enumerate(all_samples_sorted, 1):
            video_name = sample['video_name']
            error = sample['error']

            # 确定分类
            category = 'poor'
            for min_err, max_err, cat in error_ranges:
                if min_err <= error < max_err:
                    category = cat
                    break

            clean_name = os.path.splitext(video_name)[0].replace('/', '_').replace('\\', '_')

            # 源文件名
            src_filename = f'rank_{rank:04d}_error{error:.3f}_{category}_{clean_name}_density_map.png'
            src_path = os.path.join(all_samples_dir, src_filename)

            # 目标路径（在分类目录下）
            dst_path = os.path.join(
                category_dirs[category],
                f'rank_{rank:04d}_error{error:.3f}_{clean_name}_density_map.png'
            )

            # 复制文件（而不是创建符号链接，以保证跨平台兼容性）
            if os.path.exists(src_path):
                import shutil
                shutil.copy2(src_path, dst_path)

        print("   ✓ 按分类创建子目录完成")
        print()

        # 4. 生成统计报告
        print("4. 生成详细的统计报告...")

        report_path = os.path.join(vis_dir, 'visualization_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("密度图可视化统计报告\n")
            f.write("=" * 80 + "\n\n")

            f.write("【总体统计】\n")
            f.write(f"  总样本数: {len(all_samples_sorted)}\n")
            f.write(f"  平均误差: {np.mean([s['error'] for s in all_samples_sorted]):.4f}\n")
            f.write(f"  最小误差: {np.min([s['error'] for s in all_samples_sorted]):.4f}\n")
            f.write(f"  最大误差: {np.max([s['error'] for s in all_samples_sorted]):.4f}\n\n")

            f.write("【误差分布】\n")
            for min_err, max_err, category in error_ranges:
                count = error_distribution[category]
                percentage = (count / len(all_samples_sorted)) * 100
                f.write(f"  {category:10s} (error < {max_err:6.1f}): {count:4d} 样本 ({percentage:5.1f}%)\n")
            f.write("\n")

            f.write("【样本排名（按误差从小到大）】\n")
            f.write(f"{'Rank':<8}{'Video Name':<50}{'GT Count':<12}{'Pred Count':<12}{'Error':<10}{'Category':<12}\n")
            f.write("-" * 104 + "\n")

            for rank, sample in enumerate(all_samples_sorted, 1):
                # 确定分类
                error = sample['error']
                category = 'poor'
                for min_err, max_err, cat in error_ranges:
                    if min_err <= error < max_err:
                        category = cat
                        break

                f.write(f"{rank:<8}{sample['video_name']:<50}{sample['gt_count']:<12.2f}"
                        f"{sample['pred_count']:<12.2f}{error:<10.4f}{category:<12}\n")

            f.write("\n" + "=" * 80 + "\n")
            try:
                import pandas as pd
                f.write("报告生成时间: " + str(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')) + "\n")
            except:
                f.write("报告生成时间: " + str(time.strftime('%Y-%m-%d %H:%M:%S')) + "\n")
            f.write("=" * 80 + "\n")

        print(f"   ✓ 统计报告已保存: {report_path}")
        print()

        # 5. 最终总结
        print(f"{'=' * 60}")
        print(f"✅ 所有样本的密度图可视化完成！")
        print(f"{'=' * 60}")
        print(f"\n📊 可视化结果统计：")
        print(f"  总样本数: {len(all_samples_sorted)}")
        print(f"  保存位置: {vis_dir}")
        print(f"\n📁 目录结构：")
        print(f"  ├── all_samples/")
        print(f"  │   └── rank_0001~rank_{len(all_samples_sorted):04d}_*.png (所有样本)")
        for min_err, max_err, category in error_ranges:
            count = error_distribution[category]
            if count > 0:
                print(f"  ├── by_category_{category}/")
                print(f"  │   └── {count} 个样本")
        print(f"  └── visualization_report.txt (详细报告)")
        print(f"\n💾 文件说明：")
        print(f"  all_samples/: 包含所有样本的密度图，按误差值排序")
        print(f"  by_category_*/: 按误差分类组织，便于查看不同质量的样本")
        print(f"  visualization_report.txt: 包含完整的统计信息和排名\n")
        print(f"{'=' * 60}\n")

        return

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

    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    milestones = [i for i in range(0, args.epochs, 60)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.8)  ### reduce learning rate by 0.8 every 60 epochs
    lossMSE = nn.MSELoss().cuda()
    lossSL1 = nn.SmoothL1Loss().cuda()
    best_loss = np.inf

    os.makedirs(args.save_path, exist_ok=True)
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        scheduler.step()
        

        print(f"Epoch: {epoch:02d}")
        for phase in ['train', 'val']:
            if phase == 'val':
                if epoch % args.eval_freq != 0:
                    continue
                model.eval()
                ground_truth = list()
                predictions = list()
            else:
                model.train()
            
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
                
                bformat='{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}'
                dataloader = dataloaders[phase]
                with tqdm(total=len(dataloader),bar_format=bformat,ascii='░▒█') as pbar:
                    for i, item in enumerate(dataloader):
                        if phase == 'train':
                            train_step+=1
                        elif phase == 'val':
                            val_step+=1
                        with torch.cuda.amp.autocast(enabled=True):
                            data = item[0].cuda().type(torch.cuda.FloatTensor) # B x (THW) x C
                            example = item[1].cuda().type(torch.cuda.FloatTensor) # B x (THW) x C
                            density_map = item[2].cuda().type(torch.cuda.FloatTensor).half() * args.scale_counts  ###scaling up args.scale_counts.This helps in magnifying the loss
                            actual_counts = item[3].cuda() # B x 1
                            filename = item[4]  # 文件名
                            thw = item[5]
                            shot_num = item[6][0] ## number of shots
                            b,n,c = data.shape
                            # video_y = model(data, example, thw, shot_num=shot_num)
                            y, pose_feat = model(data, example, thw, shot_num=shot_num)
                            # target_len = video_y.shape[1]
                            # # 加载对应的pose预测结果
                            # fname = filename[0] if isinstance(filename, list) else filename
                            #
                            # # 构建pose文件路径，假设pose文件在pose_results目录下
                            # if '.' not in fname:
                            #     pose_file = fname + '.npy'
                            # else:
                            #     # 如果有扩展名，则替换为.npy
                            #     pose_file = fname.replace('.mp4', '.npy').replace('.avi', '.npy').replace('.npz',
                            #                                                                               '.npy')
                            # pose_path = os.path.join('./pose_density_predictions/', pose_file)  # 调整路径
                            #
                            # if os.path.exists(pose_path):
                            #     pose_pred = np.load(pose_path)  # 形状: (s,)
                            #     pose_pred = torch.from_numpy(pose_pred).cuda().float()
                            #     # 调整形状以匹配video预测 (s,) -> (1, s)
                            #     pose_y = pose_pred.unsqueeze(0)  # 形状: (1, s)
                            #     pose_y = pose_y.unsqueeze(1)  # [B, 1, s]
                            #     pose_output_interp = F.interpolate(pose_y, size=target_len, mode='linear',
                            #                                        align_corners=True)
                            #     pose_y = pose_output_interp.squeeze(1)  # [B, s]
                            # else:
                            #     print(f"Warning: Pose file not found: {pose_path}")
                            #     # 如果pose文件不存在，创建零张量作为默认值
                            #     pose_y = torch.zeros_like(video_y).cuda()  # 形状: (1, s)
                            # # 多模态融合
                            # y = fusion_layer(video_y, pose_y)
                            # print(f"y shape: {y.shape}")  ## shape of predicted density maps 0512
                            if phase == 'train':
                                mask = np.random.binomial(n=1, p=0.8, size=[1,density_map.shape[1]])  ### random masking of 20% density map
                            else:
                                mask = np.ones([1, density_map.shape[1]])
                            
                            masks = np.tile(mask, (density_map.shape[0], 1))
                            
                            
                            masks = torch.from_numpy(masks).cuda() #[1,45]
                            loss = ((y - density_map) ** 2)
                            loss = ((loss * masks) / density_map.shape[1]).sum() / density_map.shape[0]  ### mse

                            predict_count = torch.sum(y, dim=1).type(torch.cuda.FloatTensor) / args.scale_counts # sum density map
                            # loss_mse = torch.mean((predict_count - actual_counts)**2)
                            if phase == 'val':
                                ground_truth.append(actual_counts.detach().cpu().numpy())
                                predictions.append(predict_count.detach().cpu().numpy())
                            
                            loss2 = lossSL1(predict_count, actual_counts)  ###L1 loss between count and predicted count
                            loss3 = torch.sum(torch.div(torch.abs(predict_count - actual_counts), actual_counts + 1e-1)) / \
                            predict_count.flatten().shape[0]    #### reduce the mean absolute error (mae loss)
                            if phase=='train':

                                loss1 = (loss + 1.0 * loss3) / args.accum_iter  ### mse between density maps + mae loss (loss3)
                                loss1.backward()    ### call backward
                                if (i + 1) % args.accum_iter == 0: ### accumulate gradient
                                    optimizer.step() ##update parameters
                                    optimizer.zero_grad()
                                    torch.cuda.empty_cache()
                            
                            epoch_loss = loss.item()
                            count += b
                            total_loss_all += loss.item() * b
                            total_loss1 += loss.item() * b
                            total_loss2 += loss2.item() * b
                            total_loss3 += loss3.item() * b
                            off_by_zero += (torch.abs(actual_counts.round() - predict_count.round()) ==0).sum().item()  ## off by zero
                            off_by_one += (torch.abs(actual_counts.round() - predict_count.round()) <=1 ).sum().item()   ## off by one
                            mse += ((actual_counts - predict_count.round())**2).sum().item()
                            mae += torch.sum(torch.div(torch.abs(predict_count.round() - actual_counts), (actual_counts) + 1e-1)).item()  ##mean absolute error


                            
                            pbar.set_description(f"EPOCH: {epoch:02d} | PHASE: {phase} ")
                            # pbar.set_postfix_str(f" LOSS: {total_loss_all/count:.2f} | MAE:{mae/count:.2f} | LOSS ITER: {loss.item():.2f} | OBZ: {off_by_zero/count:.2f} | OBO: {off_by_one/count:.2f} | RMSE: {np.sqrt(mse/count):.3f}")
                            pbar.set_postfix({
                                "LOSS": f"{total_loss_all / count:.2f}",
                                "MAE": f"{mae / count:.2f}",
                                "LOSS_ITER": f"{loss.item():.2f}",
                                "OBZ": f"{off_by_zero / count:.2f}",
                                "OBO": f"{off_by_one / count:.2f}",
                                "RMSE": f"{np.sqrt(mse / count):.3f}"
                            })
                            pbar.update()

                if args.use_tensorboard:
                    if phase == 'train':
                        writer.add_scalar('Loss/train_total', total_loss_all / float(count), epoch)
                        writer.add_scalar('Loss/train_loss1', total_loss1 / float(count), epoch)
                        writer.add_scalar('Loss/train_loss2', total_loss2 / float(count), epoch)
                        writer.add_scalar('Loss/train_loss3', total_loss3 / float(count), epoch)
                        writer.add_scalar('Metrics/train_obz', off_by_zero / count, epoch)
                        writer.add_scalar('Metrics/train_obo', off_by_one / count, epoch)
                        writer.add_scalar('Metrics/train_rmse', np.sqrt(mse / count), epoch)
                        writer.add_scalar('Metrics/train_mae', mae / count, epoch)

                    if phase == 'val':
                        if not os.path.isdir(args.save_path):
                            os.makedirs(args.save_path)
                        writer.add_scalar('Loss/val_total', total_loss_all / float(count), epoch)
                        writer.add_scalar('Loss/val_loss1', total_loss1 / float(count), epoch)
                        writer.add_scalar('Loss/val_loss2', total_loss2 / float(count), epoch)
                        writer.add_scalar('Loss/val_loss3', total_loss3 / float(count), epoch)
                        writer.add_scalar('Metrics/val_obz', off_by_zero / count, epoch)
                        writer.add_scalar('Metrics/val_obo', off_by_one / count, epoch)
                        writer.add_scalar('Metrics/val_mae', mae / count, epoch)
                        writer.add_scalar('Metrics/val_rmse', np.sqrt(mse / count), epoch)

                        ### Savind checkpoints
                        # if total_loss_all / float(count) < best_loss:
                        #     best_loss = total_loss_all / float(count)
                        #     best_filename = f'best_1_obo{off_by_one / count:.3f}_mae{mae / count:.3f}.pyth'
                        #     torch.save({
                        #         'epoch': epoch,
                        #         'model_state_dict': model.state_dict(),
                        #         'optimizer_state_dict': optimizer.state_dict(),
                        #     }, os.path.join(args.save_path, best_filename))
                        # torch.save({
                        #     'model_state_dict': model.state_dict(),
                        #     'optimizer_state_dict': optimizer.state_dict(),
                        # }, os.path.join(args.save_path, 'epoch_{}.pyth'.format(str(epoch).zfill(3))))
                        current_obo = off_by_one / count
                        current_mae = mae / count

                        # 维护 OBO 最大的 3 个模型
                        current_obo_path = os.path.join(args.save_path,
                                                        f'best_obo_{current_obo:.3f}_mae{mae / count:.3f}_obz{off_by_zero / count:.3f}_rmse{np.sqrt(mse / count):.3f}_epoch{epoch:03d}.pyth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, current_obo_path)

                        args.best_obo_checkpoints.append((current_obo, current_obo_path))
                        args.best_obo_checkpoints.sort(key=lambda x: x[0], reverse=True)

                        if len(args.best_obo_checkpoints) > 3:
                            _, worst_obo_path = args.best_obo_checkpoints.pop()
                            if os.path.exists(worst_obo_path):
                                os.remove(worst_obo_path)

                        # 维护 MAE 最小的 3 个模型
                        current_mae_path = os.path.join(args.save_path,
                                                        f'best_mae_{current_mae:.3f}_obo{off_by_one / count:.3f}_obz{off_by_zero / count:.3f}_rmse{np.sqrt(mse / count):.3f}_epoch{epoch:03d}.pyth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, current_mae_path)

                        args.best_mae_checkpoints.append((current_mae, current_mae_path))
                        args.best_mae_checkpoints.sort(key=lambda x: x[0])

                        if len(args.best_mae_checkpoints) > 3:
                            _, worst_mae_path = args.best_mae_checkpoints.pop()
                            if os.path.exists(worst_mae_path):
                                os.remove(worst_mae_path)

                if args.use_wandb:
                    if phase == 'train':
                        wandb.log({"epoch": epoch,
                            # "lr": lr,
                            "train_loss": total_loss_all/float(count), 
                            "train_loss1": total_loss1/float(count), 
                            "train_loss2": total_loss2/float(count), 
                            "train_loss3": total_loss3/float(count), 
                            "train_obz": off_by_zero/count,
                            "train_obo": off_by_one/count,
                            "train_rmse": np.sqrt(mse/count),
                            "train_mae": mae/count
                        })
                    
                    if phase == 'val':
                        if not os.path.isdir(args.save_path):
                            os.makedirs(args.save_path)
                        wandb.log({"epoch": epoch, 
                            "val_loss": total_loss_all/float(count), 
                            "val_loss1": total_loss1/float(count), 
                            "val_loss2": total_loss2/float(count), 
                            "val_loss3": total_loss3/float(count), 
                            "val_obz": off_by_zero/count, 
                            "val_obo": off_by_one/count,
                            "val_mae": mae/count, 
                            "val_rmse": np.sqrt(mse/count)
                        })

                        ### Savind checkpoints
                        if total_loss_all/float(count) < best_loss:
                            best_loss = total_loss_all/float(count)
                            best_filename = f'best_1_obo{off_by_one/count:.3f}_mae{mae/count:.3f}.pyth'
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                }, os.path.join(args.save_path, best_filename))
                        torch.save({
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                }, os.path.join(args.save_path, 'epoch_{}.pyth'.format(str(epoch).zfill(3))))

    # 在训练结束后关闭writer（通常在main函数末尾）
    if args.use_tensorboard:
        writer.close()

    if args.use_wandb:                                   
        wandb_run.finish()


if __name__=='__main__':
    main()