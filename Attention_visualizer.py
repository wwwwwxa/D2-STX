import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from matplotlib import rc


class AttentionVisualizer:
    """注意力权重可视化工具"""

    def __init__(self, save_dir='attention_visualizations'):
        """
        Args:
            save_dir: 保存可视化结果的目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def visualize_spatial_video_to_pose(self, attention_weights, video_filename,
                                        frame_indices=None, aggregate_time=True):
        """
        可视化空间注意力: Video → Pose
        证明每个像素位置找到的关节点权重分布

        Args:
            attention_weights: [B, T, H*W, J] 或 [T, H*W, J] 或 [H*W, J]
            video_filename: 视频文件名
            frame_indices: 要可视化的帧索引列表（如果为None，则每隔10帧保存一次，最多5帧）
            aggregate_time: 是否聚合整个时间维度
        """
        # 处理输入维度
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # 取第一个batch [T, H*W, J]

        T, HW, J = attention_weights.shape

        # 创建保存目录
        save_subdir = os.path.join(self.save_dir, 'spatial_video_to_pose',
                                   video_filename.replace('.avi', '').replace('.mp4', ''))
        os.makedirs(save_subdir, exist_ok=True)

        # 1. 聚合整个时间长度的权重
        if aggregate_time:
            aggregated_weights = attention_weights.mean(dim=0)  # [H*W, J]

            plt.figure(figsize=(14, max(8, HW * 0.3)))
            sns.heatmap(aggregated_weights.cpu().numpy(),
                        cmap='YlOrRd',
                        cbar_kws={'label': 'Attention Weight'},
                        xticklabels=[f'J{i}' for i in range(J)],
                        yticklabels=[f'P{i}' for i in range(HW)])
            plt.xlabel('Joint Index', fontsize=12, fontweight='bold')
            plt.ylabel('Pixel Position Index', fontsize=12, fontweight='bold')
            plt.title(f'Spatial Video→Pose Attention (Aggregated over {T} frames)\n{video_filename}',
                      fontsize=14, fontweight='bold')
            plt.tight_layout()

            save_path = os.path.join(save_subdir, 'aggregated_time_heatmap.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved aggregated heatmap: {save_path}")

            # 为每个像素位置绘制关节权重分布的柱状图（选取前9个像素）
            num_pixels_to_plot = min(9, HW)
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes = axes.flatten()

            for idx in range(num_pixels_to_plot):
                weights = aggregated_weights[idx].cpu().numpy()
                axes[idx].bar(range(J), weights, color='steelblue', alpha=0.7)
                axes[idx].set_xlabel('Joint Index', fontsize=10)
                axes[idx].set_ylabel('Attention Weight', fontsize=10)
                axes[idx].set_title(f'Pixel {idx}', fontsize=11, fontweight='bold')
                axes[idx].grid(axis='y', alpha=0.3)

                # 标注最大权重的关节
                max_idx = np.argmax(weights)
                axes[idx].axvline(max_idx, color='red', linestyle='--', linewidth=2, alpha=0.5)
                axes[idx].text(max_idx, weights[max_idx], f'J{max_idx}',
                               ha='center', va='bottom', fontweight='bold', color='red')

            plt.suptitle(f'Pixel-wise Joint Attention Distribution (Aggregated)\n{video_filename}',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()

            save_path = os.path.join(save_subdir, 'pixel_joint_distribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved pixel-joint distribution: {save_path}")

        # 2. 每隔10帧保存一次（最多5帧）
        if frame_indices is None:
            frame_indices = list(range(0, T, max(1, T // 5)))[:5]

        for frame_idx in frame_indices:
            if frame_idx >= T:
                continue

            frame_weights = attention_weights[frame_idx]  # [H*W, J]

            # 热力图
            plt.figure(figsize=(14, max(8, HW * 0.3)))
            sns.heatmap(frame_weights.cpu().numpy(),
                        cmap='YlOrRd',
                        cbar_kws={'label': 'Attention Weight'},
                        xticklabels=[f'J{i}' for i in range(J)],
                        yticklabels=[f'P{i}' for i in range(HW)])
            plt.xlabel('Joint Index', fontsize=12, fontweight='bold')
            plt.ylabel('Pixel Position Index', fontsize=12, fontweight='bold')
            plt.title(f'Spatial Video→Pose Attention at Frame {frame_idx}\n{video_filename}',
                      fontsize=14, fontweight='bold')
            plt.tight_layout()

            save_path = os.path.join(save_subdir, f'frame_{frame_idx:03d}_heatmap.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved frame {frame_idx} heatmap: {save_path}")

            # 柱状图（选取前9个像素）
            num_pixels_to_plot = min(9, HW)
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes = axes.flatten()

            for idx in range(num_pixels_to_plot):
                weights = frame_weights[idx].cpu().numpy()
                axes[idx].bar(range(J), weights, color='steelblue', alpha=0.7)
                axes[idx].set_xlabel('Joint Index', fontsize=10)
                axes[idx].set_ylabel('Attention Weight', fontsize=10)
                axes[idx].set_title(f'Pixel {idx}', fontsize=11, fontweight='bold')
                axes[idx].grid(axis='y', alpha=0.3)

                max_idx = np.argmax(weights)
                axes[idx].axvline(max_idx, color='red', linestyle='--', linewidth=2, alpha=0.5)
                axes[idx].text(max_idx, weights[max_idx], f'J{max_idx}',
                               ha='center', va='bottom', fontweight='bold', color='red')

            plt.suptitle(f'Pixel-wise Joint Attention at Frame {frame_idx}\n{video_filename}',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()

            save_path = os.path.join(save_subdir, f'frame_{frame_idx:03d}_distribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved frame {frame_idx} distribution: {save_path}")

    def visualize_spatial_pose_to_video(self, attention_weights, video_filename,
                                        frame_indices=None, aggregate_time=True, attn_outputs=None):
        """
        可视化空间注意力: Pose → Video
        证明每个关节点找到的像素区域权重分布

        Args:
            attention_weights: [B, T, J, H*W] 或 [T, J, H*W] 或 [J, H*W]
            video_filename: 视频文件名
            frame_indices: 要可视化的帧索引列表
            aggregate_time: 是否聚合整个时间维度
            attn_outputs: [B, T, J, C] 注意力输出特征（可选）
        """
        # 处理输入维度
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # [T, J, H*W]

        T, J, HW = attention_weights.shape
        H = W = int(np.sqrt(HW))  # 假设是正方形

        # 创建保存目录
        save_subdir = os.path.join(self.save_dir, 'spatial_pose_to_video',
                                   video_filename.replace('.avi', '').replace('.mp4', ''))
        os.makedirs(save_subdir, exist_ok=True)

        # 定义身体部位分组
        body_parts = {
            'Right Arm': [5, 7, 9, 11],
            'Left Arm': [6, 8, 10, 12],
            'Right Leg': [13, 15, 17, 19],
            'Left Leg': [14, 16, 18, 20],
            'Shoulder & Elbow': [1, 2, 3, 4],
            'Waist & Torso': [24, 23, 22, 0]
        }

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        linestyles = ['-', '--', '-.', ':', '-', '--']

        # ========== 新增可视化1: attn_output 的激活强度（通道平均） ==========
        if attn_outputs is not None:
            # 处理维度
            if attn_outputs.dim() == 4:
                attn_outputs = attn_outputs[0]  # [T, J, C]

            T_out, J_out, C = attn_outputs.shape

            fig, ax = plt.subplots(figsize=(20, 6))

            for idx, (part_name, joint_list) in enumerate(body_parts.items()):
                part_activations = []

                for joint_idx in joint_list:
                    if joint_idx >= J_out:
                        continue

                    # 计算每个关节的通道平均值（能量强度）
                    joint_features = attn_outputs[:, joint_idx, :].cpu().numpy()  # [T, C]
                    joint_activation = np.mean(np.abs(joint_features), axis=1)  # [T] 取绝对值的平均
                    # 或者用L2范数：joint_activation = np.linalg.norm(joint_features, axis=1)
                    part_activations.append(joint_activation)

                if len(part_activations) > 0:
                    part_avg_activation = np.mean(part_activations, axis=0)  # [T]

                    ax.plot(range(T_out), part_avg_activation,
                            color=colors[idx],
                            linestyle=linestyles[idx],
                            linewidth=2.5,
                            label=part_name,
                            alpha=0.85)

            ax.set_xlabel('Time Position (Frame)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Activation Strength (Channel Mean)', fontsize=14, fontweight='bold')
            ax.set_title(f'Spatial Attention Output: Body Parts Activation over Time\n{video_filename}',
                         fontsize=16, fontweight='bold', pad=20)
            ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                      ncol=6, fontsize=12, frameon=True, shadow=True)

            if T_out > 20:
                step = max(1, T_out // 20)
                ax.set_xticks(range(0, T_out, step))

            plt.tight_layout()
            save_path = os.path.join(save_subdir, 'body_parts_spatial_activation.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved body parts spatial activation: {save_path}")

        # ========== 新增可视化2: attn_weights 在时间长度上的变化 ==========
        fig, ax = plt.subplots(figsize=(20, 6))

        for idx, (part_name, joint_list) in enumerate(body_parts.items()):
            part_weights_over_time = []

            for joint_idx in joint_list:
                if joint_idx >= J:
                    continue

                # 对每个时间点，计算该关节的平均注意力权重
                joint_weights = attention_weights[:, joint_idx, :].cpu().numpy()  # [T, H*W]
                joint_avg_weights = joint_weights.mean(axis=1)  # [T] 对空间维度求平均
                part_weights_over_time.append(joint_avg_weights)

            if len(part_weights_over_time) > 0:
                part_avg_weights = np.mean(part_weights_over_time, axis=0)  # [T]

                ax.plot(range(T), part_avg_weights,
                        color=colors[idx],
                        linestyle=linestyles[idx],
                        linewidth=2.5,
                        label=part_name,
                        alpha=0.85)

        ax.set_xlabel('Time Position (Frame)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Attention Weight', fontsize=14, fontweight='bold')
        ax.set_title(f'Spatial Attention Weights: Body Parts over Time\n{video_filename}',
                     fontsize=16, fontweight='bold', pad=20)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=6, fontsize=12, frameon=True, shadow=True)

        if T > 20:
            step = max(1, T // 20)
            ax.set_xticks(range(0, T, step))

        plt.tight_layout()
        save_path = os.path.join(save_subdir, 'body_parts_spatial_weights.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved body parts spatial weights: {save_path}")

        # # 1. 聚合整个时间长度的权重
        # if aggregate_time:
        #     aggregated_weights = attention_weights.mean(dim=0)  # [J, H*W]
        #
        #     plt.figure(figsize=(14, max(8, J * 0.5)))
        #     sns.heatmap(aggregated_weights.cpu().numpy(),
        #                 cmap='YlOrRd',
        #                 cbar_kws={'label': 'Attention Weight'},
        #                 xticklabels=[f'P{i}' for i in range(HW)],
        #                 yticklabels=[f'J{i}' for i in range(J)])
        #     plt.xlabel('Pixel Position Index', fontsize=12, fontweight='bold')
        #     plt.ylabel('Joint Index', fontsize=12, fontweight='bold')
        #     plt.title(f'Spatial Pose→Video Attention (Aggregated over {T} frames)\n{video_filename}',
        #               fontsize=14, fontweight='bold')
        #     plt.tight_layout()
        #
        #     save_path = os.path.join(save_subdir, 'aggregated_time_heatmap.png')
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #     plt.close()
        #     print(f"  ✓ Saved aggregated heatmap: {save_path}")
        #
        #     # 为每个关节绘制空间热力图（前25个关节）
        #     num_joints_to_plot = min(25, J)
        #     cols = 5
        #     rows = (num_joints_to_plot + cols - 1) // cols
        #
        #     fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        #     axes = axes.flatten() if num_joints_to_plot > 1 else [axes]
        #
        #     for joint_idx in range(num_joints_to_plot):
        #         weights_2d = aggregated_weights[joint_idx].reshape(H, W).cpu().numpy()
        #
        #         im = axes[joint_idx].imshow(weights_2d, cmap='hot', interpolation='nearest')
        #         axes[joint_idx].set_title(f'Joint {joint_idx}', fontsize=10, fontweight='bold')
        #         axes[joint_idx].axis('off')
        #         plt.colorbar(im, ax=axes[joint_idx], fraction=0.046)
        #
        #     # 隐藏多余的子图
        #     for idx in range(num_joints_to_plot, len(axes)):
        #         axes[idx].axis('off')
        #
        #     plt.suptitle(f'Joint-wise Spatial Attention Maps (Aggregated)\n{video_filename}',
        #                  fontsize=14, fontweight='bold')
        #     plt.tight_layout()
        #
        #     save_path = os.path.join(save_subdir, 'joint_spatial_maps.png')
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #     plt.close()
        #     print(f"  ✓ Saved joint spatial maps: {save_path}")
        #
        # # 2. 每隔10帧保存一次（最多5帧）
        # if frame_indices is None:
        #     frame_indices = list(range(0, T, max(1, T // 5)))[:5]
        #
        # for frame_idx in frame_indices:
        #     if frame_idx >= T:
        #         continue
        #
        #     frame_weights = attention_weights[frame_idx]  # [J, H*W]
        #
        #     # 热力图
        #     plt.figure(figsize=(14, max(8, J * 0.5)))
        #     sns.heatmap(frame_weights.cpu().numpy(),
        #                 cmap='YlOrRd',
        #                 cbar_kws={'label': 'Attention Weight'},
        #                 xticklabels=[f'P{i}' for i in range(HW)],
        #                 yticklabels=[f'J{i}' for i in range(J)])
        #     plt.xlabel('Pixel Position Index', fontsize=12, fontweight='bold')
        #     plt.ylabel('Joint Index', fontsize=12, fontweight='bold')
        #     plt.title(f'Spatial Pose→Video Attention at Frame {frame_idx}\n{video_filename}',
        #               fontsize=14, fontweight='bold')
        #     plt.tight_layout()
        #
        #     save_path = os.path.join(save_subdir, f'frame_{frame_idx:03d}_heatmap.png')
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #     plt.close()
        #     print(f"  ✓ Saved frame {frame_idx} heatmap: {save_path}")
        #
        #     # 空间热力图（前25个关节）
        #     num_joints_to_plot = min(25, J)
        #     cols = 5
        #     rows = (num_joints_to_plot + cols - 1) // cols
        #
        #     fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        #     axes = axes.flatten() if num_joints_to_plot > 1 else [axes]
        #
        #     for joint_idx in range(num_joints_to_plot):
        #         weights_2d = frame_weights[joint_idx].reshape(H, W).cpu().numpy()
        #
        #         im = axes[joint_idx].imshow(weights_2d, cmap='hot', interpolation='nearest')
        #         axes[joint_idx].set_title(f'Joint {joint_idx}', fontsize=10, fontweight='bold')
        #         axes[joint_idx].axis('off')
        #         plt.colorbar(im, ax=axes[joint_idx], fraction=0.046)
        #
        #     for idx in range(num_joints_to_plot, len(axes)):
        #         axes[idx].axis('off')
        #
        #     plt.suptitle(f'Joint-wise Spatial Attention at Frame {frame_idx}\n{video_filename}',
        #                  fontsize=14, fontweight='bold')
        #     plt.tight_layout()
        #
        #     save_path = os.path.join(save_subdir, f'frame_{frame_idx:03d}_spatial_maps.png')
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #     plt.close()
        #     print(f"  ✓ Saved frame {frame_idx} spatial maps: {save_path}")

    def visualize_temporal_video_to_pose(self, attention_weights, video_filename,
                                         pixel_indices=None):
        """
        可视化时间注意力: Video → Pose
        证明video的时间点找到pose的哪些时刻

        Args:
            attention_weights: [B, T, H*W, T] 或 [T, H*W, T] 或 [H*W, T, T]
            video_filename: 视频文件名
            pixel_indices: 要可视化的像素索引（如果为None，则选择前9个）
        """
        # 处理输入维度
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # [T, H*W, T]

        if attention_weights.shape[0] == attention_weights.shape[2]:
            # [T, H*W, T]
            T_query, HW, T_key = attention_weights.shape
        else:
            # [H*W, T, T]
            HW, T_query, T_key = attention_weights.shape

        # 创建保存目录
        save_subdir = os.path.join(self.save_dir, 'temporal_video_to_pose',
                                   video_filename.replace('.avi', '').replace('.mp4', ''))
        os.makedirs(save_subdir, exist_ok=True)

        # 选择要可视化的像素
        if pixel_indices is None:
            pixel_indices = list(range(min(9, HW)))

        # 为每个像素绘制时间注意力热力图
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()

        for idx, pixel_idx in enumerate(pixel_indices):
            if idx >= 9:
                break

            if attention_weights.shape[0] == T_query:
                # [T, H*W, T]
                weights = attention_weights[:, pixel_idx, :].cpu().numpy()  # [T_query, T_key]
            else:
                # [H*W, T, T]
                weights = attention_weights[pixel_idx].cpu().numpy()  # [T_query, T_key]

            im = axes[idx].imshow(weights, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            axes[idx].set_xlabel('Pose Time (Key)', fontsize=10)
            axes[idx].set_ylabel('Video Time (Query)', fontsize=10)
            axes[idx].set_title(f'Pixel {pixel_idx}', fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=axes[idx], fraction=0.046)

            # 绘制对角线（期望权重集中在对角线附近）
            axes[idx].plot([0, min(T_query, T_key) - 1], [0, min(T_query, T_key) - 1],
                           'w--', linewidth=2, alpha=0.7, label='Diagonal')
            axes[idx].legend(loc='upper right', fontsize=8)

        # 隐藏多余的子图
        for idx in range(len(pixel_indices), 9):
            axes[idx].axis('off')

        plt.suptitle(f'Temporal Video→Pose Attention\n{video_filename}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_subdir, 'temporal_attention_heatmaps.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved temporal attention heatmaps: {save_path}")

        # 为每个像素绘制时间权重分布的折线图
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, pixel_idx in enumerate(pixel_indices):
            if idx >= 9:
                break

            if attention_weights.shape[0] == T_query:
                weights = attention_weights[:, pixel_idx, :].cpu().numpy()
            else:
                weights = attention_weights[pixel_idx].cpu().numpy()

            # 对每个query时间点绘制其对所有key时间点的权重
            for t_query in range(0, T_query, max(1, T_query // 5)):  # 每隔若干帧画一条线
                axes[idx].plot(range(T_key), weights[t_query],
                               label=f't={t_query}', alpha=0.7, linewidth=2)

            axes[idx].set_xlabel('Pose Time (Key)', fontsize=10)
            axes[idx].set_ylabel('Attention Weight', fontsize=10)
            axes[idx].set_title(f'Pixel {pixel_idx}', fontsize=11, fontweight='bold')
            axes[idx].grid(alpha=0.3)
            axes[idx].legend(fontsize=8, ncol=2)

        for idx in range(len(pixel_indices), 9):
            axes[idx].axis('off')

        plt.suptitle(f'Temporal Video→Pose Weight Distribution\n{video_filename}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_subdir, 'temporal_weight_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved temporal weight distribution: {save_path}")

    def visualize_temporal_pose_to_video(self, attention_weights, video_filename,
                                         joint_indices=None, attn_outputs=None):
        """
        可视化时间注意力: Pose → Video
        证明pose的时间点找到video的哪些时刻

        Args:
            attention_weights: [B, T, J, T] 或 [T, J, T] 或 [J, T, T]
            video_filename: 视频文件名
            joint_indices: 要可视化的关节索引（如果为None，则选择前25个）
            attn_outputs: [B, T, J, C] 注意力输出特征（可选）
        """
        # 处理输入维度
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # [T, J, T]

        if attention_weights.shape[0] == attention_weights.shape[2]:
            # [T, J, T]
            T_query, J, T_key = attention_weights.shape
        else:
            # [J, T, T]
            J, T_query, T_key = attention_weights.shape

        # 创建保存目录
        save_subdir = os.path.join(self.save_dir, 'temporal_pose_to_video',
                                   video_filename.replace('.avi', '').replace('.mp4', ''))
        os.makedirs(save_subdir, exist_ok=True)

        # 选择要可视化的关节
        if joint_indices is None:
            joint_indices = list(range(min(25, J)))

        # 定义身体部位分组（所有可视化共用）
        body_parts = {
            'Right Arm': [5, 7, 9, 11],
            'Left Arm': [6, 8, 10, 12],
            'Right Leg': [13, 15, 17, 19],
            'Left Leg': [14, 16, 18, 20],
            'Shoulder & Elbow': [1, 2, 3, 4],
            'Waist & Torso': [24, 23, 22, 0]
        }

        # 定义颜色和线型
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        linestyles = ['-', '--', '-.', ':', '-', '--']

        # ========== 新增：使用 attn_outputs 可视化身体部位的激活强度 ==========
        if attn_outputs is not None:
            # 处理 attn_outputs 维度
            if attn_outputs.dim() == 4:
                attn_outputs_processed = attn_outputs[0]  # [T, J, C]
            else:
                attn_outputs_processed = attn_outputs

            T_out, J_out, C = attn_outputs_processed.shape

            # 创建图形
            fig, ax = plt.subplots(figsize=(20, 6))

            # 为每个身体部位计算激活强度并绘制
            for idx, (part_name, joint_list) in enumerate(body_parts.items()):
                # 获取这个部位所有关节的激活强度
                part_activations = []

                for joint_idx in joint_list:
                    if joint_idx >= J_out:
                        continue

                    # 计算该关节在每个时间点的L2范数（激活强度）
                    joint_features = attn_outputs_processed[:, joint_idx, :].cpu().numpy()  # [T, C]
                    joint_activation = np.linalg.norm(joint_features, axis=1)  # [T]
                    part_activations.append(joint_activation)

                # 计算该部位的平均激活强度
                if len(part_activations) > 0:
                    part_avg_activation = np.mean(part_activations, axis=0)  # [T]

                    # 绘制折线
                    ax.plot(range(T_out), part_avg_activation,
                            color=colors[idx],
                            linestyle=linestyles[idx],
                            linewidth=2.5,
                            label=part_name,
                            alpha=0.85)

            # 设置图形属性
            ax.set_xlabel('Time Position (Frame)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Activation Strength (L2 Norm)', fontsize=14, fontweight='bold')
            ax.set_title(f'Temporal Activation Strength: Body Parts over Time\n{video_filename}',
                         fontsize=16, fontweight='bold', pad=20)
            ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                      ncol=6, fontsize=12, frameon=True, shadow=True)

            # 设置x轴刻度
            if T_out > 20:
                step = max(1, T_out // 20)
                ax.set_xticks(range(0, T_out, step))

            plt.tight_layout()

            # 保存图形
            save_path = os.path.join(save_subdir, 'body_parts_temporal_activation.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved body parts temporal activation: {save_path}")

        # ========== 6个身体部位的注意力权重可视化 ==========
        # 创建图形
        fig, ax = plt.subplots(figsize=(20, 6))

        # 为每个身体部位计算平均注意力权重并绘制
        for idx, (part_name, joint_list) in enumerate(body_parts.items()):
            # 获取这个部位所有关节的权重
            part_weights = []
            for joint_idx in joint_list:
                if joint_idx >= J:
                    continue
                if attention_weights.shape[0] == T_query:
                    # [T, J, T] -> 选择某个时间点的query
                    # 这里我们对所有query时间点取平均
                    weights = attention_weights[:, joint_idx, :].cpu().numpy()  # [T_query, T_key]
                    avg_weights = weights.mean(axis=0)  # 对query时间维度取平均 -> [T_key]
                else:
                    # [J, T, T]
                    weights = attention_weights[joint_idx].cpu().numpy()  # [T_query, T_key]
                    avg_weights = weights.mean(axis=0)  # 对query时间维度取平均 -> [T_key]
                part_weights.append(avg_weights)

            # 计算该部位的平均权重
            if len(part_weights) > 0:
                part_avg_weights = np.mean(part_weights, axis=0)  # [T_key]

                # 绘制折线
                ax.plot(range(T_key), part_avg_weights,
                        color=colors[idx],
                        linestyle=linestyles[idx],
                        linewidth=2.5,
                        label=part_name,
                        alpha=0.85)

        # 设置图形属性
        ax.set_xlabel('Video Time Position (Frame)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Attention Weight Value', fontsize=14, fontweight='bold')
        ax.set_title(f'Temporal Attention: Body Parts → Video\n{video_filename}',
                     fontsize=16, fontweight='bold', pad=20)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=6, fontsize=12, frameon=True, shadow=True)

        # 设置x轴刻度
        if T_key > 20:
            step = max(1, T_key // 20)
            ax.set_xticks(range(0, T_key, step))

        plt.tight_layout()

        # 保存图形
        save_path = os.path.join(save_subdir, 'body_parts_temporal_attention.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved body parts temporal attention: {save_path}")

        # ========== 原有的可视化代码保持不变 ==========
        # 为每个关节绘制时间注意力热力图（5x5网格）
        num_joints_to_plot = min(25, len(joint_indices))
        fig, axes = plt.subplots(5, 5, figsize=(20, 20))
        axes = axes.flatten()

        for idx, joint_idx in enumerate(joint_indices[:num_joints_to_plot]):
            if attention_weights.shape[0] == T_query:
                # [T, J, T]
                weights = attention_weights[:, joint_idx, :].cpu().numpy()  # [T_query, T_key]
            else:
                # [J, T, T]
                weights = attention_weights[joint_idx].cpu().numpy()  # [T_query, T_key]

            im = axes[idx].imshow(weights, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            axes[idx].set_xlabel('Video Time (Key)', fontsize=9)
            axes[idx].set_ylabel('Pose Time (Query)', fontsize=9)
            axes[idx].set_title(f'Joint {joint_idx}', fontsize=10, fontweight='bold')
            plt.colorbar(im, ax=axes[idx], fraction=0.046)

            # 绘制对角线
            # axes[idx].plot([0, min(T_query, T_key) - 1], [0, min(T_query, T_key) - 1],
            #                'w--', linewidth=1.5, alpha=0.7)

        # 隐藏多余的子图
        for idx in range(num_joints_to_plot, 25):
            axes[idx].axis('off')

        plt.suptitle(f'Temporal Pose→Video Attention\n{video_filename}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_subdir, 'temporal_attention_heatmaps.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved temporal attention heatmaps: {save_path}")

        # 为选定的关节绘制时间权重分布（3x3网格）
        selected_joints = joint_indices[:9]
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, joint_idx in enumerate(selected_joints):
            if attention_weights.shape[0] == T_query:
                weights = attention_weights[:, joint_idx, :].cpu().numpy()
            else:
                weights = attention_weights[joint_idx].cpu().numpy()

            # 对每个query时间点绘制其对所有key时间点的权重
            for t_query in range(0, T_query, max(1, T_query // 5)):
                axes[idx].plot(range(T_key), weights[t_query],
                               label=f't={t_query}', alpha=0.7, linewidth=2)

            axes[idx].set_xlabel('Video Time (Key)', fontsize=10)
            axes[idx].set_ylabel('Attention Weight', fontsize=10)
            axes[idx].set_title(f'Joint {joint_idx}', fontsize=11, fontweight='bold')
            axes[idx].grid(alpha=0.3)
            axes[idx].legend(fontsize=8, ncol=2)

        for idx in range(len(selected_joints), 9):
            axes[idx].axis('off')

        plt.suptitle(f'Temporal Pose→Video Weight Distribution\n{video_filename}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_subdir, 'temporal_weight_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved temporal weight distribution: {save_path}")

    def visualize_body_parts_activation_timeline(self, attn_outputs, video_filename,
                                                 source='temporal'):
        """
        可视化不同身体部位在时间轴上的激活程度
        类似于神经元激活值的时间序列图

        Args:
            attn_outputs: [B, T, J, C] 注意力输出特征
            video_filename: 视频文件名
            source: 'temporal' 或 'spatial' 表示数据来源
        """
        # 处理维度
        if attn_outputs.dim() == 4:
            attn_outputs = attn_outputs[0]  # [T, J, C]

        T, J, C = attn_outputs.shape

        # 定义身体部位和对应的关节（按照BlazePose 25点模型）
        body_parts_mapping = {
            'Right Arm': [5, 7, 9, 11],
            'Left Arm': [6, 8, 10, 12],
            'Right Leg': [13, 15, 17, 19],
            'Left Leg': [14, 16, 18, 20],
            'Shoulder & Elbow': [1, 2, 3, 4],
            'Waist & Torso': [24, 23, 22, 0]
        }

        # ========== 配置颜色（仅用于区分，不表示重要性）==========
        colors = {
            'Waist & Torso': '#DC143C',  # 深红色
            'Right Arm': '#FF6B6B',  # 红色
            'Left Arm': '#FFA500',  # 橙色
            'Shoulder & Elbow': '#9370DB',  # 紫色
            'Right Leg': '#2E8B57',  # 绿色
            'Left Leg': '#4169E1',  # 蓝色
        }

        # ========== 配置线型（用于区分不同部位）==========
        linestyles = {
            'Waist & Torso': '-',  # 实线
            'Right Arm': '-',  # 实线
            'Left Arm': '-',  # 实线
            'Shoulder & Elbow': '-.',  # 点划线
            'Right Leg': '--',  # 虚线
            'Left Leg': ':',  # 点线
        }

        # ========== 统一线宽（所有部位平等）==========
        uniform_linewidth = 2.5

        # 计算每个身体部位的激活值
        activations = {}

        for part_name, joint_indices in body_parts_mapping.items():
            part_activation = []

            for joint_idx in joint_indices:
                if joint_idx >= J:
                    continue

                # 计算该关节在每个时间点的激活强度
                # 方法1: L2范数（推荐）
                joint_features = attn_outputs[:, joint_idx, :].cpu().numpy()  # [T, C]
                joint_activation = np.linalg.norm(joint_features, axis=1)  # [T]

                # 方法2: 绝对值均值（可选）
                # joint_activation = np.mean(np.abs(joint_features), axis=1)

                part_activation.append(joint_activation)

            if len(part_activation) > 0:
                # 对该部位的所有关节取平均
                activations[part_name] = np.mean(part_activation, axis=0)  # [T]

        # 创建图形
        fig, ax = plt.subplots(figsize=(22, 6))

        # 绘制每个身体部位的激活曲线
        for part_name, activation in activations.items():
            ax.plot(range(T), activation,
                    color=colors[part_name],
                    linestyle=linestyles[part_name],
                    linewidth=uniform_linewidth,
                    label=part_name,
                    alpha=0.85,
                    marker='o' if part_name in ['Right Hand', 'Left Hand'] else None,
                    markersize=4 if part_name in ['Right Hand', 'Left Hand'] else 0,
                    markevery=max(1, T // 20))  # 每隔几个点标记一个marker

        # ========== 自动检测活跃区域 ==========
        # 计算上肢（手）的平均激活
        upper_limb_activation = (activations.get('Right Hand', np.zeros(T)) +
                                 activations.get('Left Hand', np.zeros(T))) / 2

        # 计算动态阈值（中位数 + 标准差）
        threshold = np.median(upper_limb_activation) + 0.5 * np.std(upper_limb_activation)

        # 找出活跃区域（连续的高激活区域）
        active_regions = []
        in_active_region = False
        start_idx = 0

        for t in range(T):
            if upper_limb_activation[t] > threshold:
                if not in_active_region:
                    start_idx = t
                    in_active_region = True
            else:
                if in_active_region:
                    if t - start_idx > 3:  # 至少持续3帧
                        active_regions.append((start_idx, t - 1))
                    in_active_region = False

        # 处理最后一个区域
        if in_active_region and T - start_idx > 3:
            active_regions.append((start_idx, T - 1))

        # 标注活跃区域
        y_max = ax.get_ylim()[1]
        y_min = ax.get_ylim()[0]

        for idx, (start, end) in enumerate(active_regions):
            # 绘制背景矩形
            ax.axvspan(start, end, alpha=0.15, color='red', zorder=0)

            # 添加文本标注
            mid_point = (start + end) / 2
            ax.text(mid_point, y_max * 0.95,
                    'Active\nPositions',
                    ha='center', va='top',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5',
                              facecolor='white',
                              edgecolor='red',
                              linewidth=2))

        # 标注相对不活跃区域（如果有的话）
        if len(active_regions) > 1:
            # 在两个活跃区域之间找一个最长的不活跃区域
            max_gap = 0
            max_gap_region = None

            for i in range(len(active_regions) - 1):
                gap_start = active_regions[i][1]
                gap_end = active_regions[i + 1][0]
                gap_length = gap_end - gap_start

                if gap_length > max_gap and gap_length > 5:
                    max_gap = gap_length
                    max_gap_region = (gap_start, gap_end)

            if max_gap_region:
                start, end = max_gap_region
                ax.axvspan(start, end, alpha=0.1, color='blue', zorder=0)

                mid_point = (start + end) / 2
                ax.text(mid_point, y_max * 0.95,
                        'Relatively\nInactive\nPositions',
                        ha='center', va='top',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5',
                                  facecolor='white',
                                  edgecolor='blue',
                                  linewidth=2,
                                  linestyle='--'))

        # 设置坐标轴
        ax.set_xlabel('Joint Weight in Temporal Position', fontsize=14, fontweight='bold')
        ax.set_ylabel('Neuron Activation Value', fontsize=14, fontweight='bold')

        # 设置标题
        # action_name = "Sit-up" if "stu" in video_filename.lower() else "Action"
        # ax.set_title(f'Body Parts Activation Timeline - {action_name}\n{video_filename}',
        #              fontsize=16, fontweight='bold', pad=20)
        ax.set_title(f'Body Parts Activation Timeline\n{video_filename}',
                     fontsize=16, fontweight='bold', pad=20)
        # 设置x轴刻度
        if T > 30:
            step = max(1, T // 15)
            ax.set_xticks(range(0, T, step))
        else:
            ax.set_xticks(range(T))

        # 网格
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

        # 图例
        ax.legend(loc='upper left', fontsize=12,
                  frameon=True, shadow=True, ncol=6)

        # 设置y轴范围留出空间给标注
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], current_ylim[1] * 1.1)

        plt.tight_layout()

        # 保存
        save_dir = os.path.join(self.save_dir, f'{source}_activation_timeline')
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f'{video_filename}_body_parts_timeline.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved body parts activation timeline: {save_path}")

        # ========== 额外：绘制归一化版本 ==========
        fig, ax = plt.subplots(figsize=(22, 6))

        # 归一化到 [0, 100] 范围
        for part_name, activation in activations.items():
            activation_norm = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8) * 100

            ax.plot(range(T), activation_norm,
                    color=colors[part_name],
                    linestyle=linestyles[part_name],
                    linewidth=uniform_linewidth,
                    label=part_name,
                    alpha=0.85,
                    marker='o' if part_name in ['Right Hand', 'Left Hand'] else None,
                    markersize=4 if part_name in ['Right Hand', 'Left Hand'] else 0,
                    markevery=max(1, T // 20))

        ax.set_xlabel('Joint Weight in Temporal Position', fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Activation Value (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'Body Parts Activation Timeline (Normalized) - {video_filename}',
                     fontsize=16, fontweight='bold', pad=20)

        if T > 30:
            step = max(1, T // 15)
            ax.set_xticks(range(0, T, step))

        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='upper left', fontsize=12, frameon=True, shadow=True, ncol=6)

        plt.tight_layout()

        save_path_norm = os.path.join(save_dir, f'{video_filename}_body_parts_timeline_normalized.png')
        plt.savefig(save_path_norm, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved normalized timeline: {save_path_norm}")

    def save_attention_weights_raw(self, attention_dict, video_filename):
        """
        保存原始注意力权重数据（.npz格式）

        Args:
            attention_dict: 包含所有注意力权重的字典
            video_filename: 视频文件名
        """
        save_subdir = os.path.join(self.save_dir, 'raw_weights',
                                   video_filename.replace('.avi', '').replace('.mp4', ''))
        os.makedirs(save_subdir, exist_ok=True)

        save_dict = {}
        for key, value in attention_dict.items():
            if isinstance(value, torch.Tensor):
                save_dict[key] = value.cpu().numpy()

        save_path = os.path.join(save_subdir, 'attention_weights.npz')
        np.savez_compressed(save_path, **save_dict)
        print(f"  ✓ Saved raw attention weights: {save_path}")
