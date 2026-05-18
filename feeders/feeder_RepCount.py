#!/usr/bin/env python
# -*- coding:utf-8 -*-  
__author__ = 'Lambert'
__time__ = '2023-10-20 18:23'

import os

import numpy as np
import pickle
import json
import random
import math
import csv
import torch

from torch.utils.data import Dataset
from .label_norm import normalize_label


class Feeder(Dataset):
    def __init__(self, data_path, label_path, repeat=1, random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):

        if 'val' in label_path:
            self.train_val = 'val'
            self.data_dict_path = 'test_poses_SVAC_33-25_new'
        else:
            self.train_val = 'train'
            self.data_dict_path = 'train_poses_SVAC_33-25_new'

        # self.nw_RepCount_root = '/data/cshwang/SVAC/SVAC_1/RepCount_pose'
        self.nw_RepCount_root = '/data/xawang/SP4L-main/RepCount_pose'
        self.time_steps = 64

        # self.bone = [(2, 1), (1, 1), (3, 2), (4, 3), (8, 4), (5, 1), (6, 5), (7, 6), (9, 7), (11, 1), (10, 1),
        #              (12, 10), (14, 12), (16, 14), (24, 12), (26, 24), (28, 26), (30, 28), (32, 30), (18, 20),
        #              (20, 22), (22, 16), (13, 11), (15, 13), (17, 15), (25, 13), (27, 25), (29, 27), (31, 29),
        #              (33, 31), (19, 21), (21, 23), (23, 17)]

        self.bone = [(23, 0), (0, 0), (21, 1), (21, 2), (21, 23), (1, 3), (5, 3), (5, 7), (9, 7), (5, 11),
                   (2, 4), (4, 6), (6, 8), (6, 12), (8, 10), (21, 24), (22, 24), (13, 22), (14, 22), (15, 13),
                   (15, 17), (17, 19), (14, 16), (16, 18), (18, 20)]

        self.label = []
        self.data = []
        self.video_org_len = []
        self.time_points = []
        self.path_skeleton = os.path.join(self.nw_RepCount_root, self.data_dict_path)
        self.dir_skeleton = os.listdir(self.path_skeleton)
        self.filenames = []

        for index in range(len(self.dir_skeleton)):
            file_skeleton = self.dir_skeleton[index]

            # 会导致模型的预测值为nan
            if file_skeleton == 'test534.npy':
                continue

            if file_skeleton == 'train143.npy':
                continue

            # train dataset
            if file_skeleton == 'stu6_44.npy':
                continue

            file_skeleton_path = os.path.join(self.path_skeleton, file_skeleton)
            info = np.load(file_skeleton_path, allow_pickle=True).item()
            self.label.append(info['Count'])
            self.data.append(info['pose'])
            self.video_org_len.append(info['Video_Original_Length'])
            self.time_points.append(info['Time_points'])
            self.filenames.append(file_skeleton)
            # **检查 filenames 和 video_org_len 是否匹配**
            # print("Checking filenames and video_org_len:")
            # for i in range(len(self.filenames)):
            #     print(f"Index: {i}, Filename: {self.filenames[i]}, Video Original Length: {self.video_org_len[i]}")

        self.data_all = self.obtain_landmark_label()

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.repeat = repeat
        # self.load_data()
        if normalization:
            self.get_mean_map()

    def normalize_landmarks(self, all_landmarks):
        x_max = np.expand_dims(np.max(all_landmarks[:, :, 0], axis=1), 1)
        x_min = np.expand_dims(np.min(all_landmarks[:, :, 0], axis=1), 1)

        y_max = np.expand_dims(np.max(all_landmarks[:, :, 1], axis=1), 1)
        y_min = np.expand_dims(np.min(all_landmarks[:, :, 1], axis=1), 1)

        z_max = np.expand_dims(np.max(all_landmarks[:, :, 2], axis=1), 1)
        z_min = np.expand_dims(np.min(all_landmarks[:, :, 2], axis=1), 1)

        all_landmarks[:, :, 0] = (all_landmarks[:, :, 0] - x_min) / (x_max - x_min)
        all_landmarks[:, :, 1] = (all_landmarks[:, :, 1] - y_min) / (y_max - y_min)
        all_landmarks[:, :, 2] = (all_landmarks[:, :, 2] - z_min) / (z_max - z_min)

        # all_landmarks = all_landmarks.reshape(len(all_landmarks), 99)
        return all_landmarks

    def obtain_landmark_label(self):
        all_landmarks = []
        file_separator = ','
        n_landmarks = 25
        n_dimensions = 3
        for index in range(len(self.data)):
            skeleton_row = self.data[index]
            landmarks = np.expand_dims(skeleton_row, axis=skeleton_row.ndim).reshape([skeleton_row.shape[0], n_landmarks, n_dimensions])
            # landmarks = self.normalize_landmarks(landmarks)
            all_landmarks.append(landmarks)

        all_landmarks = np.array(all_landmarks)
        return all_landmarks

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.filenames) * self.repeat

    def __iter__(self):
        return self

    def rand_view_transform(self, X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1, 0, 0], [0, math.cos(agx), math.sin(agx)], [0, -math.sin(agx), math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0, 1, 0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])
        X0 = np.dot(np.reshape(X, (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def preprocess(self, video_frame_length, time_points, num_frames=64):
        """
        process label(.csv) to density map label
        Args:
            video_frame_length: video total frame number, i.e 1024frames
            time_points: label point example [1, 23, 23, 40,45,70,.....] or [0]
            num_frames: 64
        Returns: for example list [0.1,0.8,0.1, .....]
        """
        # new_crop = []
        # for i in range(len(time_points)):  # frame_length -> 64
        #     item = min(math.ceil((float((time_points[i])) / float(video_frame_length)) * num_frames), num_frames - 1)
        #     new_crop.append(item)
        # new_crop = np.sort(new_crop)
        # label = normalize_label(new_crop, num_frames)
        #
        # return label
        # for i in range(len(time_points)):  # frame_length -> 64
        #     item = min(math.ceil((float((time_points[i])) / float(video_frame_length)) * num_frames), num_frames - 1)
        #     new_crop.append(item)
        # new_crop = np.sort(new_crop)
        # label, index_pos = normalize_label(new_crop, num_frames)
        label, index_pos = normalize_label(time_points, video_frame_length)

        index_neg = []
        if time_points[0] > 0:
            index_neg.append(0)
            index_neg.append(time_points[0])
        for i in range(1, len(time_points), 2):
            if i == len(time_points) - 1:
                if time_points[i] < video_frame_length:
                    index_neg.append(time_points[i])
                    index_neg.append(video_frame_length)
            else:
                x_a = time_points[i]
                x_b = time_points[i + 1]
                num = x_b - x_a
                if num > 0:
                    index_neg.append(x_a)
                    index_neg.append(x_b)

        # assert len(index_neg) <= num_frames, "len(index_neg) > num_frames"
        # index_neg.extend([-1 for i in range(num_frames - len(index_neg))])
        return label, index_pos, index_neg

    def __getitem__(self, index):
        # print(f"Fetching index: {index}")  # 确保代码执行到这里
        count = self.label[index % len(self.filenames)]
        value = self.data_all[index % len(self.filenames)]
        video_len_org = self.video_org_len[index % len(self.filenames)]
        time_points = self.time_points[index % len(self.filenames)]
        # Print video_len_org and time_points
        # print(f"video_len_org: {video_len_org}")
        # print(f"time_points: {time_points}")
        # 确保 video_frame_length 计算正确
        try:
            video_frame_length = self.video_org_len[index]
        except IndexError as e:
            print(f"IndexError: index {index} out of range for video_org_len (len={len(self.video_org_len)})")
            raise e
        # video_frame_length = self.video_org_len[index]
        label, index_pos, index_neg = self.preprocess(video_frame_length, time_points, num_frames=video_frame_length)
        # print(f"label (before tensor conversion): {label}")
        # print(f"Index: {index}")

        # Create classification labels
        classification_labels = torch.zeros(video_frame_length)
        # 将index_pos位置设置为1
        # valid_pos_indices = [idx for idx in index_pos if idx >= 0 and idx < 64]
        # classification_labels[valid_pos_indices] = 1
        # 遍历每对 [start, end]
        for i in range(0, len(index_pos), 2):
            start = max(0, index_pos[i])
            if i + 1 < len(index_pos):
                end = min(63, index_pos[i + 1])  # 保证不越界
                classification_labels[start:end + 1] = 1  # 包括 end 帧

        # 创建周期长度标签
        length_labels = torch.zeros(video_frame_length)

        # 确保index_pos的点是成对的
        if len(index_pos) >= 2:
            for i in range(0, len(index_pos), 2):  # 每次跳过2个点，确保取的是独立的周期对
                if i + 1 < len(index_pos):  # 确保有结束点
                    start_idx = index_pos[i]
                    end_idx = index_pos[i + 1]

                    # 检查索引是否有效
                    if 0 <= start_idx < video_frame_length and 0 <= end_idx < video_frame_length:
                        # 计算周期长度
                        cycle_length = end_idx - start_idx + 1

                        # 为周期内的所有帧设置长度标签
                        for frame_idx in range(start_idx, end_idx + 1):
                            length_labels[frame_idx] = cycle_length

        label = torch.tensor(label)
        index_pos = torch.tensor(index_pos)
        index_neg = torch.tensor(index_neg)
        # print(f"index_pos  (before tensor conversion): {index_pos}, Shape: {index_pos.shape}")
        # print(f"index_neg  (before tensor conversion): {index_neg}, Shape: {index_neg.shape}")
        label_interval, _, _ = self.preprocess(video_len_org, time_points, num_frames=video_frame_length)
        # print(f"label_interval (before tensor conversion): {label_interval}")
        # print(f"label_interval shape: {np.array(label_interval).shape}")  # 先转换成 NumPy 数组查看形状

        label_interval = torch.tensor(label_interval)

        # label, index_pos, index_neg = self.preprocess(video_len_org, time_points, num_frames=self.num_frame)
        # label = torch.tensor(label)
        # index_pos = torch.tensor(index_pos)
        # index_neg = torch.tensor(index_neg)

        # label_interval = torch.tensor(label)  # 这里改成正确的 label

        filename = self.filenames[index % len(self.filenames)]

        if self.train_val == 'train':
            random.random()

            # agx = random.randint(-60, 60)
            # agy = random.randint(-60, 60)

            agx = random.randint(-99, 99)
            agy = random.randint(-99, 99)

            s = random.uniform(0.5, 1.5)

            center = value[0, 1, :]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0))
            scalerValue = scalerValue * 2 - 1
            scalerValue = np.reshape(scalerValue, (-1, 25, 3))

            # data = np.zeros((video_frame_length, 25, 3))
            data = scalerValue  # [T, 25, 3]

            # value = scalerValue[:, :, :]
            # length = value.shape[0]
            #
            # random_idx = random.sample(list(np.arange(length)) * 100, self.time_steps)
            # random_idx.sort()
            # data[:, :, :] = value[random_idx, :, :]
            # data[:, :, :] = value[random_idx, :, :]

        else:
            random.random()
            agx = 0
            agy = 0
            s = 1.0

            center = value[0, 1, :]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0))
            scalerValue = scalerValue * 2 - 1

            scalerValue = np.reshape(scalerValue, (-1, 25, 3))

            # data = np.zeros((video_frame_length, 25, 3))
            data = scalerValue # [T, 25, 3]

            # value = scalerValue[:, :, :]
            # length = value.shape[0]
            #
            # idx = np.linspace(0, length - 1, self.time_steps).astype(np.int)
            # data[:, :, :] = value[idx, :, :]  # T,V,C

        if 'bone' in self.data_path:
            data_bone = np.zeros_like(data)
            for bone_idx in range(25):
                data_bone[:, self.bone[bone_idx][0] - 1, :] = data[:, self.bone[bone_idx][0] - 1, :] - data[:, self.bone[bone_idx][1] - 1, :]
            data = data_bone

        if 'motion' in self.data_path:
            data_motion = np.zeros_like(data)
            data_motion[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]
            data = data_motion
        data = np.transpose(data, (2, 0, 1))
        C, T, V = data.shape
        data = np.reshape(data, (C, T, V, 1))

        # return data, label, label_interval, filename, index, index_pos, index_neg
        return data, label, count, filename, index, index_pos, index_neg, classification_labels, length_labels

    def top_k(self, score, top_k):
        rank = score.argsort()

        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
