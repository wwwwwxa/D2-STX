import pathlib
from random import randint
import torch.utils.data
import os, sys, math
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from scipy import integrate
from scipy import ndimage

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import einops

class Rep_count(torch.utils.data.Dataset):
    def __init__(self,
                 split="train",
                 add_noise= False,
                 num_frames=512,
                 tokens_dir = "saved_tokens_reencoded",
                 exemplar_dir = "exemplar_tokens_reencoded",
                 pose_tokens_dir="saved_tokens_reencoded",
                 pose_exemplar_dir="exemplar_tokens_reencoded",
                 select_rand_segment=True,
                 compact=False,
                 lim_constraint=np.inf,
                 pool_tokens_factor=1.0,
                 peak_at_random_location=False,
                 get_overlapping_segments=False,
                 multishot=True,
                 density_peak_width=1.0,
                 threshold=0.0):
        
        self.num_frames=num_frames
        self.lim_constraint = lim_constraint
        self.tokens_dir = tokens_dir
        self.exemplar_dir = exemplar_dir
        # Pose数据路径
        self.pose_tokens_dir = pose_tokens_dir
        self.pose_exemplar_dir = pose_exemplar_dir
        self.compact = compact
        self.select_rand_segment = select_rand_segment
        self.pool_tokens = pool_tokens_factor
        self.split = split # set the split to load
        self.add_noise = add_noise # add noise to frames (augmentation)
        self.peak_at_random_location = peak_at_random_location
        self.get_overlapping_segments = get_overlapping_segments
        self.multishot = multishot
        self.density_peak_width = density_peak_width
        self.threshold = threshold ## cutoff to decide if we should select exemplar from other videos of same class
        # csv_path = f'datasets/repcount/{self.split}_with_fps.csv'
        csv_path = f'poselabel_datasets/{self.split}_with_fps.csv'
        self.df = pd.read_csv(csv_path)
        if self.split == 'train':
            self.df = self.df[self.df['count'].notna()]
            self.df = self.df[self.df['num_frames'] > 64]
            self.df = self.df.drop(self.df.loc[self.df['name']=='stu1_10.mp4'].index)
            self.df = self.df[self.df['count'] > 0] # remove no reps
        print(f"--- Loaded: {len(self.df)} videos for {self.split} --- " )

        
    def load_tokens(self,path,is_exemplar,data_type='pose',bounds=None, lim_constraint=np.inf, id=None, cycle_start_id=0, count=None, shot_num=1, get_overlapping_segments=False, segment_id=0):
        """
        loading video or exemplar tokens. 
        input: path -> the path for the saved video/exemplar tokens
               is_exemplar -> True/False for encoding exemplar tokens or not.
               bounds -> (st, end) to trim video given the start and end timestamps. 
               lim_constraint -> for memory issues, lim_constraint trims the video till this value. 
               shot_num = (1,2,3) how many exemplar tokens to return

        output:
               video/exemplar tokens
        """
        
        try:
            # print(f'Loading: {path}')  # 调试信息
            if data_type == 'pose':
                # Pose数据加载方式 (.npy文件)
                tokens = np.load(path)
                tokens = tokens.transpose(0, 2, 3, 4, 1)  # [S,C,T,V,M]->[57,768,16,25,1]
                # print(f'Loaded pose successfully, shape: {tokens.shape}')  # 调试信息
            else:
                # Video数据加载方式 (.npz文件)
                tokens = np.load(path)['arr_0']   # Load in format C x t x h x w
                # print(f'Loaded video successfully, shape: {tokens.shape}')  # 调试信息
        except:
            print(f'Could not load {path}')
            

            
        if is_exemplar:
            if shot_num == 0:
                shot_num = 1
            if count is not None:
                if count > 0.6:
                    N = round(count)
                else:
                    N = 1
            else:
                N = min(tokens.shape[0],20) if self.split == 'train' else tokens.shape[0]
            if self.select_rand_segment or self.split == 'train':
                shot_num = min(shot_num, N)
                
                idx = cycle_start_id + np.random.choice(np.arange(N), size=shot_num, replace=False) ### randomly selecting 'shot_num' of repetitions as our exemplars

            else:
                shot_num = min(shot_num, N)
                # id = N // 2
                # idx = np.array([id])
                idx = np.arange(shot_num)

            new_tokens = []
            for id in idx:
                new_tokens.append(tokens[id])
            tokens = np.stack(new_tokens)
            if tokens.shape[0] == 0:
                print(path)
            tokens = einops.rearrange(tokens,'S C T H W -> C (S T) H W')   ### rearranging the tokens shape 
            tokens = torch.from_numpy(tokens)

        else:   ### for video tokens
            if bounds is not None:
                if data_type == 'pose':
                    # Pose数据使用8作为分母
                    low_bound = bounds[0] // 8
                    up_bound = min(math.ceil(bounds[1] / 8), lim_constraint)
                else:
                    # Video数据使用8作为分母
                    low_bound = bounds[0] // 8
                    up_bound = min(math.ceil(bounds[1] / 8), lim_constraint)
            if get_overlapping_segments:
                if self.split != 'test':   
                    tokens1 = tokens[segment_id::4]   ### concatenating tokens for non-overlapping windows
                    tokens1 = einops.rearrange(tokens1,'S C T H W -> C (S T) H W')
                    tokens1 = tokens1[:, max(low_bound-(2*segment_id), 0):max(up_bound-(2*segment_id), 0)]
                    tokens1 = torch.from_numpy(tokens1)
                    tokens2 = None
                else:
                    tokens1 = tokens[0::4]
                    tokens2 = tokens[2::4]
                
                    tokens1 = einops.rearrange(tokens1,'S C T H W -> C (S T) H W')
                    tokens2 = einops.rearrange(tokens2,'S C T H W -> C (S T) H W')
                    tokens1 = tokens1[:, low_bound:up_bound]
                    tokens2 = tokens2[:, max(low_bound-4, 0) : max(up_bound-4, 0)]
                    tokens1 = torch.from_numpy(tokens1)
                    tokens2 = torch.from_numpy(tokens2)
                if self.pool_tokens < 1.0 and not is_exemplar and data_type == 'video':
                    factor = math.ceil(tokens.shape[-1] * self.pool_tokens)
                    tokens1 = torch.nn.functional.adaptive_avg_pool3d(tokens1, (tokens1.shape[-3], factor, factor)) ### spatial average pooling to fit on the gpus. set pool_tokens_factor to 1 to stop any downsampling
                    if tokens2 is not None:
                        tokens2 = torch.nn.functional.adaptive_avg_pool3d(tokens2, (tokens2.shape[-3], factor, factor))   ###
                if self.split != 'test':
                    tokens = tokens1
                else:
                    tokens = (tokens1, tokens2)
            else:
                tokens = tokens[0::4] # non overlapping segments
                tokens = einops.rearrange(tokens,'S C T H W -> C (S T) H W')
                tokens = tokens[:, low_bound:up_bound]               
            
                tokens = torch.from_numpy(tokens)
                if self.pool_tokens < 1.0 and data_type == 'video':
                    factor = math.ceil(tokens.shape[-1] * self.pool_tokens)
                    tokens = torch.nn.functional.adaptive_avg_pool3d(tokens, (tokens.shape[-3], factor, factor))
        if is_exemplar:
            return tokens, shot_num
        else:
            return tokens


    def __getitem__(self, index):
        video_name = self.df.iloc[index]['name'].replace('.mp4', '.npz')
        pose_name = self.df.iloc[index]['name'].replace('.mp4', '.npy')
        action_type = self.df.iloc[index]['type']

        
        row = self.df.iloc[index]
        if self.get_overlapping_segments and self.split=='train':
            segment_id = np.random.randint(4)
        else:
            segment_id = 0
        cycle = [int(float(row[key])) for key in row.keys() if 'L' in key and not math.isnan(row[key])]   ### repetition start-end timestamps
        try:
            cycle_start_id = row['cycle_start_id']
        except:
            cycle_start_id = 0
        # print(row['count'])
        if self.split == 'train':
            lim_constraint = 150  ### maybe have this constraint to fit into gpus
        else:
            lim_constraint = np.inf

        if self.multishot:
            if self.split == 'train':
                # shot_num_ = 0
                shot_num_ = np.random.randint(0,3)  ### number of examples. randomly sample between 0 to 3
            else:
                shot_num_ = 0  #### during inference use 0-shot
        else:
            shot_num_ = 1
        
        ### choosing examples from random videos with same class

        if np.random.rand() < self.threshold and action_type != 'other' and self.split == 'train':   #### do this with probability 0.4
            select_videos = self.df['name'][self.df['type'] == action_type].values      #### groups videos of the same action category
            select_example_video = np.random.choice(select_videos)   ### randomly select a video from the group
            exemplar_video_name = select_example_video.replace('.mp4', '.npz')   ### select exemplar from the selected video
            pose_exemplar_video_name = select_example_video.replace('.mp4', '.npy')
        else:
            exemplar_video_name = video_name
            pose_exemplar_video_name = pose_name
        
        segment_start = row['segment_start']
        segment_end = row['segment_end']  
        num_frames = row['num_frames']   
        
        ### --- Creating density maps ---
        frame_ids = np.arange(num_frames)
        low = ((segment_start // 8) + (segment_id * 2)) * 8
        up = (min(math.ceil(segment_end / 8 ), lim_constraint))* 8
        select_frame_ids = frame_ids[low:up][0::8]
        density_map_alt = np.zeros(len(select_frame_ids))
        actual_counts = 0
        for i in range(0,len(cycle),2):
            if cycle[i] == cycle[i+1]:
                continue
            actual_counts += 1
            st, end = (cycle[i]//8) * 8, min(np.ceil(cycle[i+1]/8) * 8, select_frame_ids[-1])
            if st in select_frame_ids and end in select_frame_ids:
                start_id = np.where(select_frame_ids == st)[0][0]
                end_id = np.where(select_frame_ids == end)[0][0]
                mid = (start_id + end_id)//2      ### get the middle of the repetitions
                density_map_alt[mid] = 1    ### assign 1 to the middle of repetitions
        gt_density = ndimage.gaussian_filter1d(density_map_alt, sigma=self.density_peak_width, order=0) ### gaussian smoothing
        count = gt_density.sum()
        starts = np.array(cycle[0::2])
        ends = np.array(cycle[1::2])
        durations = ends - starts
        durations = durations.astype(np.float32)
        durations[durations == 0] = np.inf
        # select_exemplar = durations.argmin()

        ### Load exemplar tokens
        pose_exemplar_path = f"{self.pose_exemplar_dir}/{pose_exemplar_video_name}"
        if self.split == 'train':
            pose_example_rep, pose_shot_num = self.load_tokens(
                pose_exemplar_path, True, 'pose', cycle_start_id=cycle_start_id, shot_num=shot_num_)
        else:
            pose_example_rep, pose_shot_num = self.load_tokens(
                pose_exemplar_path, True, 'pose', id=0, shot_num=shot_num_)

        examplar_path = f"{self.exemplar_dir}/{exemplar_video_name}"
        if self.split == 'train':
            video_example_rep, video_shot_num = self.load_tokens(examplar_path,True, 'video', cycle_start_id=cycle_start_id, shot_num=shot_num_)   ###load the exemplar tokens
        else:
            video_example_rep, video_shot_num = self.load_tokens(examplar_path,True, 'video', id = 0, shot_num=shot_num_)     #### changing the id can give you specific exemplars in the video. by default id=0, returns the first repetition

        if shot_num_ == 0:
            pose_shot_num = 0
            video_shot_num = 0
        if video_example_rep.shape[1] == 0:
            print(row)

        # 加载pose视频tokens
        pose_video_path = f"{self.pose_tokens_dir}/{pose_name}"
        pose_vid_tokens = self.load_tokens(
            pose_video_path, False, 'pose', (segment_start, segment_end),
            lim_constraint=lim_constraint, segment_id=segment_id,
            get_overlapping_segments=self.get_overlapping_segments)
        ### Load video tokens
        video_path = f"{self.tokens_dir}/{video_name}"
        vid_tokens = self.load_tokens(video_path,False, 'video',(segment_start,segment_end), lim_constraint=lim_constraint, segment_id=segment_id, get_overlapping_segments=self.get_overlapping_segments) ###load the video tokens. lim_constraint for memory issues
        

        if not self.select_rand_segment:
            vid_tokens = vid_tokens
            gt_density = torch.from_numpy(gt_density).half()
            return {
                'pose_data': pose_vid_tokens,
                'pose_example': pose_example_rep,
                'video_data': vid_tokens,
                'video_example': video_example_rep,
                'density_map': gt_density,
                'actual_counts': gt_density.sum(),
                'filename': row['name'][:-4],
                'pose_thw': list(pose_vid_tokens[0].shape[-3:]) if pose_vid_tokens is not None else None,
                'video_thw': list(vid_tokens[0].shape[-3:]) if vid_tokens is not None else None,
                'pose_shot_num': pose_shot_num,
                'video_shot_num': video_shot_num
            }

            # 处理随机片段选择的情况
        T = row['num_frames']
        if T <= self.num_frames:
            start, end = 0, T
        else:
            # 这里需要根据实际情况调整步长
            start = random.choice(np.arange(0, T - self.num_frames, 128))  # pose用128
            end = start + self.num_frames

        # 对pose数据进行采样
        pose_sampled_segments = pose_vid_tokens[(start // 128):(end // 128)]
        pose_thw = pose_sampled_segments.shape[-3:]
        pose_sampled_segments = einops.rearrange(pose_sampled_segments, 'C t h w -> (t h w) C')
        pose_gt = gt_density[(start // 8):(end // 8)]

        # 对video数据进行采样
        video_start = random.choice(np.arange(0, T - self.num_frames, 64))  # video用64
        video_end = video_start + self.num_frames
        video_sampled_segments = vid_tokens[(video_start // 64):(video_end // 64)]
        video_thw = video_sampled_segments.shape[-3:]
        video_sampled_segments = einops.rearrange(video_sampled_segments, 'C t h w -> (t h w) C')
        video_gt = gt_density[(video_start // 4):(video_end // 4)]

        return {
            'pose_data': pose_sampled_segments,
            'pose_example': pose_example_rep,
            'video_data': video_sampled_segments,
            'video_example': video_example_rep,
            'density_map': pose_gt,  # 使用pose的gt作为主要标签
            'actual_counts': pose_gt.sum(),
            'filename': row['name'][:-4],
            'pose_thw': pose_thw,
            'video_thw': video_thw,
            'pose_shot_num': pose_shot_num,
            'video_shot_num': video_shot_num
        }
        

    def __len__(self):
        return len(self.df)

    def collate_fn(self, batch):
        from torch.nn.utils.rnn import pad_sequence

        # 新的字典格式处理
        if isinstance(batch[0], dict):
            # 分别处理pose数据和video数据
            pose_data_list = [x['pose_data'] for x in batch if x['pose_data'] is not None]
            pose_example_list = [x['pose_example'] for x in batch if x['pose_example'] is not None]
            video_data_list = [x['video_data'] for x in batch if x['video_data'] is not None]
            video_example_list = [x['video_example'] for x in batch if x['video_example'] is not None]

            # 处理pose数据
            if len(pose_data_list) > 0:
                pose_vids = pad_sequence([einops.rearrange(data, 'C T H W -> T C H W') for data in pose_data_list])
                pose_examples = torch.stack(pose_example_list).squeeze(1)

                if self.compact:
                    pose_vids = einops.rearrange(pose_vids, 'T B C H W -> B (T H W) C')
                    pose_examples = einops.rearrange(pose_examples, 'B C T H W -> B (T H W) C')
                else:
                    pose_vids = einops.rearrange(pose_vids, 'T B C H W -> B C T H W')
            else:
                pose_vids = None
                pose_examples = None

            # 处理video数据
            if len(video_data_list) > 0:
                video_vids = pad_sequence([einops.rearrange(data, 'C T H W -> T C H W') for data in video_data_list])
                video_examples = torch.stack(video_example_list).squeeze(1)

                if self.compact:
                    video_vids = einops.rearrange(video_vids, 'T B C H W -> B (T H W) C')
                    video_examples = einops.rearrange(video_examples, 'B C T H W -> B (T H W) C')
                else:
                    video_vids = einops.rearrange(video_vids, 'T B C H W -> B C T H W')
            else:
                video_vids = None
                video_examples = None

            # 处理其他数据
            gt_density = einops.rearrange(pad_sequence([x['density_map'] for x in batch]), 'S B -> B S')
            gt_density_sum = torch.tensor([x['actual_counts'] for x in batch], dtype=torch.float)
            names = [x['filename'] for x in batch]
            pose_thw = [x['pose_thw'] for x in batch]
            video_thw = [x['video_thw'] for x in batch]
            pose_shot_num = [x['pose_shot_num'] for x in batch]
            video_shot_num = [x['video_shot_num'] for x in batch]

            # 直接返回字典格式
            return {
                'pose_data': pose_vids,
                'pose_example': pose_examples,
                'video_data': video_vids,
                'video_example': video_examples,
                'density_map': gt_density,
                'actual_counts': gt_density_sum,
                'filename': names,
                'pose_thw': pose_thw,
                'video_thw': video_thw,
                'pose_shot_num': pose_shot_num,
                'video_shot_num': video_shot_num
            }

        else:
            # 保持原来的处理逻辑（向后兼容）
            if len(batch[0][0]) == 2:
                vids = pad_sequence([einops.rearrange(x[0][0], 'C T H W -> T C H W') for x in batch])
                vids1 = pad_sequence([einops.rearrange(x[0][1], 'C T H W -> T C H W') for x in batch])
                if self.compact:
                    vids = einops.rearrange(vids, 'T B C H W -> B (T H W) C')
                    vids1 = einops.rearrange(vids1, 'T B C H W -> B (T H W) C')
                else:
                    vids = einops.rearrange(vids, 'T B C H W -> B C T H W')
                    vids1 = einops.rearrange(vids1, 'T B C H W -> B C T H W')
                vids = (vids, vids1)
            else:
                vids = pad_sequence([einops.rearrange(x[0], 'C T H W -> T C H W') for x in batch])
                if self.compact:
                    vids = einops.rearrange(vids, 'T B C H W -> B (T H W) C')
                else:
                    vids = einops.rearrange(vids, 'T B C H W -> B C T H W')

            exemplars = torch.stack([x[1] for x in batch]).squeeze(1)
            if self.compact:
                exemplars = einops.rearrange(exemplars, 'B C T H W -> B (T H W) C')
            gt_density = einops.rearrange(pad_sequence([x[2] for x in batch]), 'S B -> B S')
            gt_density_sum = torch.tensor([x[3] for x in batch], dtype=torch.float)
            names = [x[4] for x in batch]
            thw = [x[5] for x in batch]
            shot_num = [x[6] for x in batch]

        return vids, exemplars, gt_density, gt_density_sum, names, thw, shot_num


## testing
if __name__=='__main__':
    from tqdm import tqdm
    dat = Rep_count(select_rand_segment=False, compact=False, pool_tokens_factor=0.5, get_overlapping_segments=False)
    print('--- dataset created ---')
    device = torch.device("cpu")
    print(f'Device: {device}')
    dataloader = torch.utils.data.DataLoader(dat,
                                             batch_size=1,
                                             num_workers=1,
                                             shuffle=False,
                                             pin_memory=False,
                                             drop_last=True,
                                             collate_fn=dat.collate_fn)
    
    
    sum_clip_dur = []
    sum_tot_dur = []
    sum_clip_counts = []
    sum_tot_counts = []
    
    density_maps_sum = {}
    counts = {}
    density_map_sum = []
    
    fps = []
    
    for i, item in enumerate(tqdm(dataloader)):
        print(f"It. {i} \n vid tokens: {item[0][0].shape} \n exem tokens: {item[1].shape} \n density map: {item[2].shape}:{item[3]} \n \n")
        density_map_sum.append(item[3][0].item())