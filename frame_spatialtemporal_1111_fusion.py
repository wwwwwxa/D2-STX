import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import math





class SpatioTemporalCrossModalFusion(nn.Module):
    """
    时空跨模态融合模块（带注意力权重可视化版本）
    包含两阶段注意力机制:
    1. 空间注意力 (Spatial Attention): 帧内跨模态交互
    2. 时间注意力 (Temporal Attention): 跨帧时序互补

    支持灵活的融合模式控制:
    - enable_spatial_fusion: 控制是否启用空间融合
    - enable_temporal_fusion: 控制是否启用时间融合

    支持三种融合策略:
    1. Pose as Query (pose增强)
    2. Video as Query (video增强)
    3. Bidirectional (双向融合，输出层加权)
    """

    def _build_2d_sincos_pos_embed(self, H, W, embed_dim):
        """
        构建 2D 正弦余弦位置编码

        编码结构：
        - 前 embed_dim//2 维：H 方向的 sin/cos
        - 后 embed_dim//2 维：W 方向的 sin/cos

        Returns:
            pos_embed: [1, H*W, embed_dim]
        """
        assert embed_dim % 4 == 0, "embed_dim 必须是4的倍数"
        dim_each = embed_dim // 4  # 每个方向各用 embed_dim//4 维做 sin，//4 维做 cos

        # 生成 H、W 方向的网格坐标
        grid_h = torch.arange(H, dtype=torch.float32)  # [H]
        grid_w = torch.arange(W, dtype=torch.float32)  # [W]

        # meshgrid 生成二维坐标网格
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
        # grid_h, grid_w: [H, W]

        # 频率项（与时间位置编码一致）
        div_term = torch.exp(
            torch.arange(0, dim_each, dtype=torch.float32)
            * -(math.log(10000.0) / dim_each)
        )  # [dim_each]

        # H 方向位置编码
        pos_h = grid_h.flatten().unsqueeze(1) * div_term.unsqueeze(0)
        # [H*W, dim_each]

        # W 方向位置编码
        pos_w = grid_w.flatten().unsqueeze(1) * div_term.unsqueeze(0)
        # [H*W, dim_each]

        # 拼接：[sin_h, cos_h, sin_w, cos_w]
        pos_embed = torch.cat([
            torch.sin(pos_h),  # [H*W, dim_each]
            torch.cos(pos_h),  # [H*W, dim_each]
            torch.sin(pos_w),  # [H*W, dim_each]
            torch.cos(pos_w),  # [H*W, dim_each]
        ], dim=-1)  # [H*W, embed_dim]

        return pos_embed.unsqueeze(0)  # [1, H*W, embed_dim]


    def __init__(self, embed_dim=512, num_heads=8, fusion_mode='bidirectional_gating',
                 enable_spatial_fusion=True, enable_temporal_fusion=True,
                 spatial_fusion_weight=0.5, temporal_fusion_weight=0.5,
                 save_attention_weights=False, video_spatial_size=(3, 3)):
        """
        Args:
            embed_dim: 特征维度
            num_heads: 多头注意力的头数
            fusion_mode: 融合模式
            enable_spatial_fusion: 是否启用空间注意力
            enable_temporal_fusion: 是否启用时间注意力
            spatial_fusion_weight: 空间融合的输出层权重 (默认0.5)
            temporal_fusion_weight: 时间融合的输出层权重 (默认0.5)
            save_attention_weights: 是否保存注意力权重（用于可视化）
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.fusion_mode = fusion_mode
        self.enable_spatial_fusion = enable_spatial_fusion
        self.enable_temporal_fusion = enable_temporal_fusion
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.save_attention_weights = save_attention_weights
        # 将pose关节特征投影为空间查询向量
        self.joint_spatial_query_proj = nn.Linear(embed_dim, embed_dim)

        # 可选：使用可学习的位置编码来编码关节的空间位置先验
        # 例如：手臂关节应该更关注video上半部分
        self.joint_position_embedding = nn.Parameter(
            torch.randn(25, embed_dim) * 0.02  # 25个关节的位置先验
        )

        # 空间注意力的归一化层
        self.spatial_selection_norm = nn.LayerNorm(embed_dim)

        # ==================== 空间注意力层 (Frame-level) ====================
        if enable_spatial_fusion:
            self.spatial_pose_to_video_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            )

            self.spatial_video_to_pose_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            )
            H, W = video_spatial_size
            self.video_spatial_H = H
            self.video_spatial_W = W
            video_pos_embed = self._build_2d_sincos_pos_embed(H, W, embed_dim)
            self.register_buffer('video_pos_embed', video_pos_embed)
            # 空间注意力的LayerNorm
            self.spatial_pose_norm = nn.LayerNorm(embed_dim)
            self.spatial_video_norm = nn.LayerNorm(embed_dim)

        # ==================== 时间注意力层 (Temporal) ====================
        if enable_temporal_fusion:
            self.temporal_pose_to_video_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            )

            self.temporal_video_to_pose_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            )
            # ⭐ 不预定义时间位置编码，改为存储生成参数
            # 缓存最近一次生成的位置编码，避免重复计算
            self.register_buffer('_cached_temporal_pos', None)
            self.register_buffer('_cached_T', torch.tensor(0))

            # 时间注意力的LayerNorm
            self.temporal_pose_norm = nn.LayerNorm(embed_dim)
            self.temporal_video_norm = nn.LayerNorm(embed_dim)

        # ==================== 输出层融合权重（替代门控机制）====================
        if fusion_mode == 'bidirectional_gating':
            # 空间融合的输出层权重
            if enable_spatial_fusion:
                self.spatial_fusion_weight = nn.Parameter(torch.tensor(spatial_fusion_weight))

            # 时间融合的输出层权重
            if enable_temporal_fusion:
                self.temporal_fusion_weight = nn.Parameter(torch.tensor(temporal_fusion_weight))

        # ==================== 输入预处理层 ====================
        self.input_pose_norm = nn.LayerNorm(embed_dim)
        self.input_video_norm = nn.LayerNorm(embed_dim)

        # ==================== 特征投影层 ====================
        self.pose_proj = nn.Linear(embed_dim, embed_dim)
        self.video_proj = nn.Linear(embed_dim, embed_dim)

        # ==================== 输出映射层 ====================
        self.pose_output = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        self.video_output = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    # ==================== 空间注意力方法 ====================

    def spatial_pose_as_query_fusion(self, pose_feat, video_feat):
        """
        空间注意力: Pose作为Query，从Video中提取信息（帧级处理）
        """
        B, T, J, _, C = pose_feat.shape
        _, _, H, W, _ = video_feat.shape

        fused_pose_list = []
        attention_weights_list = [] if self.save_attention_weights else None
        attn_output_list = [] if self.save_attention_weights else None

        joint_pos = self.joint_position_embedding[:J, :]  # [J, C]，防止J<25时越界
        joint_pos = joint_pos.unsqueeze(0)  # [1, J, C]

        # 逐帧处理
        for t in range(T):

            pose_t = pose_feat[:, t, :, :, :].squeeze(2)  # [B, J, C]
            video_t = video_feat[:, t, :, :, :].reshape(B, H * W, C)  # [B, H*W, C]
            pose_t_with_pos = pose_t + joint_pos  # [B, J, C]
            video_t_with_pos = video_t + self.video_pos_embed  # [B, H*W, C]
            # 交叉注意力
            attn_output, attn_weights = self.spatial_pose_to_video_attn(
                query=pose_t_with_pos,
                key=video_t_with_pos,
                value=video_t
            )  # [B, J, C], [B, J, H*W]

            # 残差连接 + LayerNorm
            fused_t = self.spatial_pose_norm(pose_t + attn_output)
            fused_pose_list.append(fused_t)

            if self.save_attention_weights:
                attention_weights_list.append(attn_weights.detach())
                attn_output_list.append(attn_output.detach())  # ✅ 新增这行
        fused_pose = torch.stack(fused_pose_list, dim=1)  # [B, T, J, C]

        if self.save_attention_weights:
            attention_weights = torch.stack(attention_weights_list, dim=1)  # [B, T, J, H*W]
            attn_outputs = torch.stack(attn_output_list, dim=1)  # ✅ 新增这行: [B, T, J, C]
            return fused_pose, attention_weights, attn_outputs


        return fused_pose, None, None

    def spatial_video_as_query_fusion(self, pose_feat, video_feat):
        """
        空间注意力: Video作为Query，从Pose中提取信息（帧级处理）
        """
        B, T, J, _, C = pose_feat.shape
        _, _, H, W, _ = video_feat.shape

        fused_video_list = []
        attention_weights_list = [] if self.save_attention_weights else None
        attn_output_list = [] if self.save_attention_weights else None  # ✅ 新增这行
        joint_pos = self.joint_position_embedding[:J, :].unsqueeze(0)  # [1, J, C]

        for t in range(T):
            pose_t = pose_feat[:, t, :, :, :].squeeze(2)  # [B, J, C]
            video_t = video_feat[:, t, :, :, :].reshape(B, H * W, C)  # [B, H*W, C]
            pose_t_with_pos = pose_t + joint_pos  # [B, J, C]  ✅
            video_t_with_pos = video_t + self.video_pos_embed  # ✅ 为video添加位置编码 [B, H*W, C]

            # 交叉注意力：video(加位置编码)作为query，pose(不加位置编码)作为key/value
            attn_output, attn_weights = self.spatial_video_to_pose_attn(
                query=video_t_with_pos,  # ✅ 修改：使用带位置编码的video
                key=pose_t_with_pos,
                value=pose_t
            )  # [B, H*W, C], [B, H*W, J]

            # 残差连接 + LayerNorm
            fused_t_flat = self.spatial_video_norm(video_t + attn_output)  # ✅ 注意：残差用原始video_t
            fused_t = fused_t_flat.reshape(B, H, W, C)
            fused_video_list.append(fused_t)

            if self.save_attention_weights:
                attention_weights_list.append(attn_weights.detach())
                attn_output_list.append(attn_output.detach())  # ✅ 新增这行

        fused_video = torch.stack(fused_video_list, dim=1)  # [B, T, H, W, C]

        if self.save_attention_weights:
            attention_weights = torch.stack(attention_weights_list, dim=1)  # [B, T, H*W, J]
            attn_outputs = torch.stack(attn_output_list, dim=1)  # ✅ 新增这行: [B, T, H*W, C]
            return fused_video, attention_weights, attn_outputs

        return fused_video, None, None  # ✅ 修改返回值

    def spatial_bidirectional_gating_fusion(self, pose_feat, video_feat):
        """
        空间注意力: 双向融合（无门控机制，直接返回增强特征）
        """
        B, T, J, _, C = pose_feat.shape
        _, _, H, W, _ = video_feat.shape

        spatial_pose_enhanced_list = []
        spatial_video_enhanced_list = []
        pose_to_video_attn_list = [] if self.save_attention_weights else None
        video_to_pose_attn_list = [] if self.save_attention_weights else None
        joint_pos = self.joint_position_embedding[:J, :].unsqueeze(0)  # [1, J, C]

        for t in range(T):
            pose_t = pose_feat[:, t, :, :, :].squeeze(2)  # [B, J25, C]
            video_t = video_feat[:, t, :, :, :].reshape(B, H * W, C)  # [B, H*W, C]

            pose_t_with_pos = pose_t + joint_pos  # ✅ [B, J, C]
            video_t_with_pos = video_t + self.video_pos_embed  # ✅ [B, H*W, C]

            # 方向1: Pose -> Video
            pose_enhanced, pose_to_video_weights = self.spatial_pose_to_video_attn(
                query=pose_t_with_pos,
                key=video_t_with_pos,
                value=video_t
            )  # [B, J, C], [B, J, H*W]
            pose_enhanced = self.spatial_pose_norm(pose_t + pose_enhanced)
            spatial_pose_enhanced_list.append(pose_enhanced)

            # 方向2: Video -> Pose
            video_enhanced, video_to_pose_weights = self.spatial_video_to_pose_attn(
                query=video_t_with_pos,
                key=pose_t_with_pos,
                value=pose_t
            )  # [B, H*W, C], [B, H*W, J]
            video_enhanced = self.spatial_video_norm(video_t + video_enhanced)
            video_enhanced = video_enhanced.reshape(B, H, W, C)
            spatial_video_enhanced_list.append(video_enhanced)

            if self.save_attention_weights:
                pose_to_video_attn_list.append(pose_to_video_weights.detach())
                video_to_pose_attn_list.append(video_to_pose_weights.detach())

        spatial_pose_enhanced = torch.stack(spatial_pose_enhanced_list, dim=1)  # [B, T, J, C]
        spatial_video_enhanced = torch.stack(spatial_video_enhanced_list, dim=1)  # [B, T, H, W, C]

        attention_dict = {}
        if self.save_attention_weights:
            attention_dict['spatial_pose_to_video'] = torch.stack(pose_to_video_attn_list, dim=1)  # [B, T, J, H*W]
            attention_dict['spatial_video_to_pose'] = torch.stack(video_to_pose_attn_list, dim=1)  # [B, T, H*W, J]

        return spatial_pose_enhanced, spatial_video_enhanced, attention_dict

    # ==================== 时间注意力方法 ====================
    def get_temporal_pos_embed(self, T, device):
        """
        动态生成正弦时间位置编码

        Args:
            T: 时间长度
            device: 设备

        Returns:
            pos_embed: [1, T, embed_dim]
        """
        # ⭐ 检查缓存，避免重复计算
        if self._cached_temporal_pos is not None and self._cached_T == T:
            return self._cached_temporal_pos

        # 生成位置索引 [0, 1, 2, ..., T-1]
        position = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)  # [T, 1]

        # 生成频率项
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32, device=device)
            * -(math.log(10000.0) / self.embed_dim)
        )  # [embed_dim/2]

        # 初始化位置编码矩阵
        pos_embed = torch.zeros(T, self.embed_dim, dtype=torch.float32, device=device)

        # 偶数维度使用 sin
        pos_embed[:, 0::2] = torch.sin(position * div_term)

        # 奇数维度使用 cos
        pos_embed[:, 1::2] = torch.cos(position * div_term)

        pos_embed = pos_embed.unsqueeze(0)  # [1, T, embed_dim]

        # ⭐ 缓存结果
        self._cached_temporal_pos = pos_embed
        self._cached_T = torch.tensor(T)

        return pos_embed


    def temporal_pose_as_query_fusion(self, pose_feat, video_feat, spatial_attn_weights=None):
        """
        改进的时间注意力: 每个关节有自己的video时间序列

        关键改进：不使用全局池化，而是为每个关节做空间选择
        """
        B, T, J, C = pose_feat.shape
        _, _, H, W, _ = video_feat.shape

        # ========== 复用空间注意力的权重 ==========
        if spatial_attn_weights is not None:
            # 已经有空间选择的结果，直接使用
            video_spatial_temporal = einops.rearrange(video_feat, 'b t h w c -> b t (h w) c')

            # 为每个关节聚合其空间特定的video时间序列
            # [B, T, J, H*W] x [B, T, H*W, C] -> [B, T, J, C]
            joint_specific_videos = torch.einsum('btjw,btwc->btjc',
                                                 spatial_attn_weights,
                                                 video_spatial_temporal)
        else:
            # fallback: 如果没有提供，还是用全局池化或重新计算
            joint_specific_videos = video_feat.mean(dim=[2, 3])  # [B, T, C]
            joint_specific_videos = joint_specific_videos.unsqueeze(2).expand(-1, -1, J, -1)
        # [B, T, H*W, C] 而不是 [B, T, C]

        temporal_fused_pose_list = []
        attention_weights_list = [] if self.save_attention_weights else None
        attn_output_list = [] if self.save_attention_weights else None
        # ⭐ 动态生成时间位置编码（根据实际 T）
        temp_pos = self.get_temporal_pos_embed(T, pose_feat.device)  # [1, T, C]

        # ========== 逐关节处理 ==========
        for j in range(J):
            pose_joint = pose_feat[:, :, j, :]  # [B, T, C]
            joint_video = joint_specific_videos[:, :, j, :]  # [B, T, C]

            # ⭐ 加入动态生成的时间位置编码
            pose_joint_with_pos = pose_joint + temp_pos  # [B, T, C]
            joint_video_with_pos = joint_video + temp_pos  # [B, T, C]

            # ========== 时间注意力（使用关节特定的video）==========
            attn_output, temporal_attn = self.temporal_pose_to_video_attn(
                query=pose_joint_with_pos,
                key=joint_video_with_pos,
                value=joint_video_with_pos
            )

            fused_joint = self.temporal_pose_norm(pose_joint + attn_output)
            temporal_fused_pose_list.append(fused_joint)

            if self.save_attention_weights:
                attention_weights_list.append(temporal_attn.detach())
                attn_output_list.append(attn_output.detach())

        # ========== 聚合 ==========
        temporal_fused_pose = torch.stack(temporal_fused_pose_list, dim=2)

        if self.save_attention_weights:
            temporal_attention = torch.stack(attention_weights_list, dim=2)  # [B, T, J, T]
            attn_outputs = torch.stack(attn_output_list, dim=2)  # [B, T, J, C]

            return temporal_fused_pose, temporal_attention, attn_outputs

        return temporal_fused_pose, None, None

    def temporal_video_as_query_fusion(self, pose_feat, video_feat, spatial_attn_weights=None):
        """
        时间注意力: Video作为Query，从Pose的时序中提取信息
        """
        B, T, J, C = pose_feat.shape
        _, _, H, W, _ = video_feat.shape

        # ========== 复用空间注意力的权重 ==========
        if spatial_attn_weights is not None:
            # 已经有空间选择的结果，直接使用
            # spatial_attn_weights: [B, T, H*W, J]
            pose_spatial_temporal = einops.rearrange(pose_feat, 'b t j c -> b t j c')

            # 为每个像素块聚合其空间特定的pose时间序列
            # [B, T, H*W, J] x [B, T, J, C] -> [B, T, H*W, C]
            pixel_specific_poses = torch.einsum('btwj,btjc->btwc',
                                                spatial_attn_weights,
                                                pose_spatial_temporal)
        else:
            # fallback: 如果没有提供，使用全局平均
            pixel_specific_poses = pose_feat.mean(dim=2)  # [B, T, C]
            pixel_specific_poses = pixel_specific_poses.unsqueeze(2).expand(-1, -1, H * W, -1)

        temporal_fused_video_list = []
        attention_weights_list = [] if self.save_attention_weights else None
        attn_output_list = [] if self.save_attention_weights else None

        # ⭐ 动态生成时间位置编码（根据实际 T）
        temp_pos = self.get_temporal_pos_embed(T, video_feat.device)  # [1, T, C]

        # Video展平: [B, T, H*W, C]
        video_flat = einops.rearrange(video_feat, 'b t h w c -> b t (h w) c')

        # 对每个像素位置单独进行时间注意力
        for s in range(H * W):
            video_pixel = video_flat[:, :, s, :]  # [B, T, C]
            pose_pixel = pixel_specific_poses[:, :, s, :]  # [B, T, C] - 该像素对应的pose时序

            # ⭐ 加入动态生成的时间位置编码
            video_pixel_with_pos = video_pixel + temp_pos  # [B, T, C]
            pose_pixel_with_pos = pose_pixel + temp_pos  # [B, T, C]

            # 时间注意力: video(加位置编码)作为query，pose(加位置编码)作为key/value
            attn_output, attn_weights = self.temporal_video_to_pose_attn(
                query=video_pixel_with_pos,
                key=pose_pixel_with_pos,
                value=pose_pixel_with_pos
            )  # [B, T, C], [B, T, T]

            # 残差连接（使用原始video_pixel）
            fused_pixel = self.temporal_video_norm(video_pixel + attn_output)
            temporal_fused_video_list.append(fused_pixel)

            if self.save_attention_weights:
                attention_weights_list.append(attn_weights.detach())
                attn_output_list.append(attn_output.detach())

        # [B, T, H*W, C] -> [B, T, H, W, C]
        temporal_fused_video = torch.stack(temporal_fused_video_list, dim=2)
        temporal_fused_video = einops.rearrange(temporal_fused_video, 'b t (h w) c -> b t h w c', h=H, w=W)

        if self.save_attention_weights:
            attention_weights = torch.stack(attention_weights_list, dim=2)  # [B, T, H*W, T]
            attn_outputs = torch.stack(attn_output_list, dim=2)  # [B, T, H*W, C]
            return temporal_fused_video, attention_weights, attn_outputs

        return temporal_fused_video, None, None

    def temporal_bidirectional_gating_fusion(self, pose_feat, video_feat):
        """
        时间注意力: 双向融合（无门控机制，直接返回增强特征）
        """
        B, T, J, C = pose_feat.shape
        _, _, H, W, _ = video_feat.shape

        # 为时间注意力准备数据
        pose_temporal = einops.rearrange(pose_feat, 'b t j c -> b j t c')  # [B, J, T, C]
        video_flat = einops.rearrange(video_feat, 'b t h w c -> b t (h w) c')  # [B, T, H*W, C]

        temporal_fused_pose_list = []
        temporal_fused_video_list = []
        pose_to_video_attn_list = [] if self.save_attention_weights else None
        video_to_pose_attn_list = [] if self.save_attention_weights else None

        # 关键循环: 对每个关节点单独进行时间注意力
        for j in range(J):
            pose_joint = pose_feat[:, :, j, :]  # [B, T, C]

            # 时间注意力: pose的某个关节的时序 query video的时序
            attn_output, attn_weights = self.temporal_pose_to_video_attn(
                query=pose_joint,
                key=video_flat[:, :, 0, :],  # 简化：使用第一个像素位置的时序
                value=video_flat[:, :, 0, :]
            )  # [B, T, C], [B, T, T]

            # 残差连接
            fused_joint = self.temporal_pose_norm(pose_joint + attn_output)
            temporal_fused_pose_list.append(fused_joint)

            if self.save_attention_weights:
                pose_to_video_attn_list.append(attn_weights.detach())

        # 关键循环: 对每个像素位置单独进行时间注意力
        for s in range(H * W):
            video_pixel = video_flat[:, :, s, :]  # [B, T, C]

            # 时间注意力: video某个位置的时序 query pose的时序
            attn_output, attn_weights = self.temporal_video_to_pose_attn(
                query=video_pixel,
                key=pose_temporal[:, 0, :, :],  # 简化：使用第一个关节的时序
                value=pose_temporal[:, 0, :, :]
            )  # [B, T, C], [B, T, T]

            # 残差连接
            fused_pixel = self.temporal_video_norm(video_pixel + attn_output)
            temporal_fused_video_list.append(fused_pixel)

            if self.save_attention_weights:
                video_to_pose_attn_list.append(attn_weights.detach())

        # 重组为原始形状
        temporal_fused_pose = torch.stack(temporal_fused_pose_list, dim=2)  # [B, T, J, C]
        temporal_fused_video = torch.stack(temporal_fused_video_list, dim=2)  # [B, T, H*W, C]
        temporal_fused_video = einops.rearrange(temporal_fused_video, 'b t (h w) c -> b t h w c', h=H, w=W)

        attention_dict = {}
        if self.save_attention_weights:
            attention_dict['temporal_pose_to_video'] = torch.stack(pose_to_video_attn_list, dim=2)  # [B, T, J, T]
            attention_dict['temporal_video_to_pose'] = torch.stack(video_to_pose_attn_list, dim=2)  # [B, T, H*W, T]

        return temporal_fused_pose, temporal_fused_video, attention_dict

    def forward(self, pose_feat, video_feat):
        """
        前向传播

        Args:
            pose_feat: [B, T, J, 1, C] Pose特征
            video_feat: [B, T, H, W, C] Video特征

        Returns:
            output: [B, T] 动作密度预测
            fusion_info: 融合信息字典（包含注意力权重）
        """
        fusion_info = {
            'fusion_mode': self.fusion_mode,
            'spatial_enabled': self.enable_spatial_fusion,
            'temporal_enabled': self.enable_temporal_fusion,
        }
        # ==================== 输入预处理：LayerNorm + Projection ====================
        # 处理pose特征: [B, T, J, 1, C] -> squeeze -> [B, T, J, C]
        pose_feat_squeezed = pose_feat.squeeze(3) if pose_feat.dim() == 5 else pose_feat
        pose_feat_normalized = self.input_pose_norm(pose_feat_squeezed)
        pose_feat_projected = self.pose_proj(pose_feat_normalized)

        # 处理video特征: [B, T, H, W, C]
        video_feat_normalized = self.input_video_norm(video_feat)
        video_feat_projected = self.video_proj(video_feat_normalized)

        # 更新特征用于后续处理
        pose_feat = pose_feat_projected.unsqueeze(3)  # [B, T, J, 1, C] 恢复维度用于空间融合
        video_feat = video_feat_projected
        # 存储注意力权重
        if self.save_attention_weights:
            fusion_info['attention_weights'] = {}
            fusion_info['attn_outputs'] = {}  # 新增这行
            fusion_info['spatial_selection'] = {}
            # ==================== Pose as Query 模式 ====================
        if self.fusion_mode == 'pose_as_query':
            # 初始化特征
            current_pose_feat = pose_feat.squeeze(3) if pose_feat.dim() == 5 else pose_feat  # [B, T, J, C]
            pose_feat_no_spatio = current_pose_feat  # 用于无空间融合时的时间融合
            # ==================== 阶段1: 空间融合 ====================
            if self.enable_spatial_fusion:
                current_pose_feat, spatial_attn_weights, spatial_attn_outputs = \
                    self.spatial_pose_as_query_fusion(pose_feat, video_feat)

                if self.save_attention_weights and spatial_attn_weights is not None:
                    fusion_info['attention_weights']['spatial_pose_to_video'] = spatial_attn_weights
                    if spatial_attn_outputs is not None:
                        fusion_info['attn_outputs']['spatial_pose_to_video'] = spatial_attn_outputs

            # ==================== 阶段2: 时间融合 ====================
            if self.enable_temporal_fusion:
                current_pose_feat, temporal_attn_weights, attn_outputs_p2v= \
                    self.temporal_pose_as_query_fusion(current_pose_feat, video_feat, spatial_attn_weights)

                if self.save_attention_weights and temporal_attn_weights is not None:
                    fusion_info['attention_weights']['temporal_pose_to_video'] = temporal_attn_weights
                    # 新增：保存 attn_outputs
                    if attn_outputs_p2v is not None:
                        fusion_info['attn_outputs']['temporal_pose_to_video'] = attn_outputs_p2v



            # ==================== 输出映射 ====================
            pose_out = self.pose_output(current_pose_feat)  # [B, T, J, 1]
            pose_out = pose_out.mean(dim=2).squeeze(-1)  # [B, T]

            return pose_out, fusion_info

        # ==================== Video as Query 模式 ====================
        elif self.fusion_mode == 'video_as_query':
            # 初始化特征
            current_video_feat = video_feat
            current_pose_feat = pose_feat.squeeze(3) if pose_feat.dim() == 5 else pose_feat  # [B, T, J, C]

            # ==================== 阶段1: 空间融合 ====================
            if self.enable_spatial_fusion:
                current_video_feat, spatial_attn_weights = \
                    self.spatial_video_as_query_fusion(pose_feat, video_feat)

                if self.save_attention_weights and spatial_attn_weights is not None:
                    fusion_info['attention_weights']['spatial_video_to_pose'] = spatial_attn_weights

            # ==================== 阶段2: 时间融合 ====================
            if self.enable_temporal_fusion:
                current_video_feat, temporal_attn_weights = \
                    self.temporal_video_as_query_fusion(current_pose_feat, current_video_feat)

                if self.save_attention_weights and temporal_attn_weights is not None:
                    fusion_info['attention_weights']['temporal_video_to_pose'] = temporal_attn_weights

            # ==================== 输出映射 ====================
            video_out = self.video_output(current_video_feat)  # [B, T, H, W, 1]
            video_out = video_out.squeeze(-1)  # [B, T, H, W]
            video_out = self.pool(video_out)  # [B, T, 1, 1]
            video_out = video_out.squeeze(-1).squeeze(-1)  # [B, T]

            return video_out, fusion_info

        elif self.fusion_mode == 'bidirectional_gating':
            # 初始化特征
            current_pose_feat = pose_feat
            current_video_feat = video_feat

            # ==================== 阶段1: 空间融合 ====================
            if self.enable_spatial_fusion:
                spatial_pose_enhanced, spatial_video_enhanced, spatial_attn = \
                    self.spatial_bidirectional_gating_fusion(pose_feat, video_feat)

                current_pose_feat = spatial_pose_enhanced
                current_video_feat = spatial_video_enhanced

                if self.save_attention_weights:
                    fusion_info['attention_weights'].update(spatial_attn)
            else:
                current_pose_feat = pose_feat.squeeze(3) if pose_feat.dim() == 5 else pose_feat

            # ==================== 阶段2: 时间融合 ====================
            if self.enable_temporal_fusion:
                temporal_pose_enhanced, temporal_video_enhanced, temporal_attn = \
                    self.temporal_bidirectional_gating_fusion(current_pose_feat, current_video_feat)

                current_pose_feat = temporal_pose_enhanced
                current_video_feat = temporal_video_enhanced

                if self.save_attention_weights:
                    fusion_info['attention_weights'].update(temporal_attn)

            # ==================== 输出映射 ====================
            pose_out = self.pose_output(current_pose_feat)  # [B, T, J, 1]
            pose_out = pose_out.mean(dim=2).squeeze(-1)  # [B, T]

            video_out = self.video_output(current_video_feat)  # [B, T, H, W, 1]
            video_out = video_out.squeeze(-1)  # [B, T, H, W]
            video_out = self.pool(video_out)  # [B, T, 1, 1]
            video_out = video_out.squeeze(-1).squeeze(-1)  # [B, T]

            # ==================== 输出层加权融合 ====================
            if self.enable_spatial_fusion and self.enable_temporal_fusion:
                a = self.temporal_fusion_weight
                fusion_info['final_fusion_weight_a'] = a.item()
            elif self.enable_spatial_fusion:
                a = self.spatial_fusion_weight
                fusion_info['final_fusion_weight_a'] = a.item()
            elif self.enable_temporal_fusion:
                a = self.temporal_fusion_weight
                fusion_info['final_fusion_weight_a'] = a.item()
            else:
                a = 0.5
                fusion_info['final_fusion_weight_a'] = a

            fusion_out = a * pose_out + (1 - a) * video_out  # [B, T]

            fusion_info['pose_contribution'] = a if isinstance(a, float) else a.item()
            fusion_info['video_contribution'] = (1 - a) if isinstance(a, float) else (1 - a).item()

            return fusion_out, fusion_info

        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
