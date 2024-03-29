## Dataset
# 
#  Defines `PushTImageDataset` and helper functions
#  Inherits from torch.utils.data.Dataset.
# 
#  The dataset class
#  - Load data ((image, agent_pos), action) from a zarr storage
#  - Normalizes each dimension of agent_pos and action to [-1,1]
#  - Returns
#   - All possible segments with length `pred_horizon`
#   - Pads the beginning and the end of each episode with repetition
#   - key `image`: shape (obs_hoirzon, 3, 96, 96)
#   - key `agent_pos`: shape (obs_hoirzon, 2)
#   - key `action`: shape (pred_horizon, 2)

# 数据集定义
# 定义了 PushTImageDataset 和辅助函数
# 数据集类
# - 从zarr存储加载数据 ((image, agent_pos), action)
# - 将agent_pos和action的每个维度标准化至[-1,1]
# - 返回
# - 所有可能的长度为 pred_horizon(预测范围) 的片段
# - 通过重复来填充每个episode的开始和结束部分
# - 关键字 image：形状为 (obs_horizon(观测范围), 3, 96, 96)
# - 关键字 agent_pos：形状为 (obs_horizon, 2)
# - 关键字 action：形状为 (pred_horizon, 2)

import cv2
import numpy as np
import zarr
import torch
import gdown
import os


def create_sample_indices(
        episode_ends:np.ndarray, # 各个episode结束的索引数组
        sequence_length:int,     # 要生成的序列长度
        pad_before: int=0,      # 序列开始前的填充量
        pad_after: int=0):      # 序列结束后的填充量
    indices = list()            # 用于存储计算出的索引
    for i in range(len(episode_ends)):  # 遍历所有的episode
        start_idx = 0
        if i > 0:       # 如果不是第一个episode
            start_idx = episode_ends[i-1]   # 从上一个episode的结束索引开始
        end_idx = episode_ends[i]   # 当前episode的结束索引
        episode_length = end_idx - start_idx    # 计算episode的长度

        min_start = -pad_before     # 序列的最小起始点，考虑填充
        max_start = episode_length - sequence_length + pad_after    # 序列的最大起始点，考虑填充

        # range stops one idx before end
        # 生成索引范围
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx              # 确定实际的起始索引，考虑填充和起点限制
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx    # 确定实际的结束索引，考虑填充和终点限制
            start_offset = buffer_start_idx - (idx+start_idx)       # 计算开始的偏移量
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx   # 计算结束的偏移量
            sample_start_idx = 0 + start_offset     # 确定样本的起始索引
            sample_end_idx = sequence_length - end_offset   # 确定样本的结束索引
            indices.append([
                buffer_start_idx, buffer_end_idx,       # 缓冲区的起始和结束索引
                sample_start_idx, sample_end_idx])      # 样本的起始和结束索引
    indices = np.array(indices)                     # 转换为 NumPy 数组
    return indices

# 当采样的数据长度小于 sequence_length 时，函数将使用原始数据序列的第一个元素填充输出序列的开始部分，使用最后一个元素填充输出序列的结束部分。
def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    # train_data：一个字典，其键为数据的不同特征名称，值为对应的numpy数组。
    # sequence_length：期望输出的序列长度。
    # buffer_start_idx 和 buffer_end_idx：采样序列在原始数据中的开始和结束索引。
    # sample_start_idx 和 sample_end_idx：采样数据在期望输出序列中的开始和结束索引。
    result = dict() # 初始化结果字典
    # 遍历训练数据中的每个键值对
    for key, input_arr in train_data.items():
        # 从训练数据中采样指定范围的数据
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        # 检查是否需要在序列前或后填充数据
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            # 创建一个形状为(sequence_length,) + input_arr.shape[1:]的全0数组
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            # 如果序列开始前有空间，则使用第一个采样点的数据填充
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            # 如果序列结束后有空间，则使用最后一个采样点的数据填充
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            # 将采样数据放到正确位置
            data[sample_start_idx:sample_end_idx] = sample
        # 将处理后的数据添加到结果字典中
        result[key] = data
    return result   # 返回包含处理后序列的字典

# normalize data
# 定义了一个 get_data_stats 函数，它计算并返回给定数据集中每个特征的最小值和最大值。这通常是数据归一化过程中的第一步，您可以使用这些统计信息将数据缩放到特定的范围，如 [0, 1]
def get_data_stats(data):
    # 将数据重塑为二维数组，其中最后一个维度保持不变
    # 这是为了能够对每个特征分别进行统计计算
    data = data.reshape(-1,data.shape[-1])
    # 创建并返回一个包含最小值和最大值的字典
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    # 根据提供的统计信息，将数据归一化到 [0, 1] 范围
    # (data - min) 计算数据相对于最小值的偏移量
    # (max - min) 为特征的取值范围
    # 除以取值范围将偏移量转换为 [0, 1] 范围内的数值
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    # 将 [0, 1] 范围的数据映射到 [-1, 1] 范围
    # ndata * 2 将 [0, 1] 映射到 [0, 2]
    # 减去 1 将 [0, 2] 映射到 [-1, 1]
    ndata = ndata * 2 - 1
    return ndata
# 这个 unnormalize_data 函数执行了与 normalize_data 函数相反的操作：它将归一化的数据恢复到原始的数值范围。如果原始数据被归一化到了 [-1, 1] 的范围，该函数将其恢复到归一化之前的范围。
def unnormalize_data(ndata, stats):
    # 将 [-1, 1] 范围的数据映射回 [0, 1] 范围
    # (ndata + 1) / 2 是将 [-1, 1] 范围内的数值转换为 [0, 2]，再除以 2 得到 [0, 1]
    ndata = (ndata + 1) / 2
    # 恢复数据到原始的数值范围
    # ndata * (stats['max'] - stats['min']) 是将 [0, 1] 范围内的数值放缩回原来的范围
    # 加上 stats['min'] 是将这个范围从 [0, range] 移动回 [min, max]
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    # 返回恢复后的数据
    return data

# dataset
class PushTImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):

        # read from zarr dataset
        # 从 Zarr 数据集中读取数据
        dataset_root = zarr.open(dataset_path, 'r')

        # float32, [0,1], (N,96,96,3)
        # 将图像数据从 NHWC 格式转换为 PyTorch 常用的 NCHW 格式
        # 原始图像数据格式为 float32, 归一化到 [0,1] 区间, 形状为 (N, 96, 96, 3)
        train_image_data = dataset_root['data']['img'][:]
        train_image_data = np.moveaxis(train_image_data, -1,1)
        # (N,3,96,96)

        # (N, D)
        # 提取训练数据的状态和动作
        # 状态向量的前两个维度代表代理（例如夹爪）的位置
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'agent_pos': dataset_root['data']['state'][:,:2],    # 提取代理位置
            'action': dataset_root['data']['action'][:]         # 提取动作
        }
        # 从数据集元数据中获取每个 episode 结束的索引
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        # 计算每个状态-动作序列的开始和结束索引，同时处理填充
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        # 计算统计数据并将数据归一化到 [-1,1]
        stats = dict()  # 存储每种数据的统计信息（最小值和最大值）
        normalized_train_data = dict()  # 存储归一化后的训练数据
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)   # 计算统计信息
            normalized_train_data[key] = normalize_data(data, stats[key])   # 归一化数据

        # images are already normalized
        # 图像数据已经被归一化
        normalized_train_data['image'] = train_image_data
        # 保存计算出的索引和统计信息，以及归一化后的训练数据
        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    # 对于 PushTImageDataset 类，样本总数由 self.indices 的长度决定，self.indices 包含了根据预定义的预测范围（pred_horizon）、观测范围（obs_horizon）和行动范围（action_horizon）计算得到的所有样本序列的索引。每个元素包含一个样本序列的起始和结束位置，以及可能的填充信息。
    def __len__(self):
        # 返回数据集中的样本总数
        # 这个总数由 self.indices 的长度决定
        # self.indices 包含了所有样本序列的索引信息
        return len(self.indices)
    # 从 self.indices 中获取当前数据点对应的开始和结束索引。
    # 使用 sample_sequence 函数根据这些索引从归一化的训练数据中提取序列。
    # 根据观测范围 obs_horizon，从提取的序列中截取所需的部分。
    # 返回处理后的样本，包括图像数据和代理位置等信息。
    # __getitem__ 方法使得 PushTImageDataset 实例可以像列表一样按索引访问，每次访问都返回一个经过预处理的数据样本，适用于模型训练或评估。
    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        # 根据索引获取这个数据点的开始和结束索引
        # 这些索引定义了需要从数据集中提取的序列的范围
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]
        # 使用这些索引从归一化的训练数据中获取相应的序列
        # sample_sequence 函数将根据这些索引从 normalized_train_data 中提取相应的序列
        # 并可能应用填充，以确保序列的长度等于 pred_horizon
        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        # 丢弃未使用的观测
        # 根据观测范围 (obs_horizon)，可能需要从序列的开始处截取一部分数据
        # 这是因为实际使用的观测数据量可能少于序列的总长度 (pred_horizon)
        nsample['image'] = nsample['image'][:self.obs_horizon,:]
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon,:]
        # 返回处理后的样本
        return nsample

## Dataset Demo
if __name__ == "__main__":    
    # download demonstration data from Google Drive
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    if not os.path.isfile(dataset_path):
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)

    # parameters
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    # create dataset from file
    dataset = PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=False,  #为了可视化效果，取消了True
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # visualize data in batch
    for batch in dataloader:
        result_image = np.zeros((2 * 96, 96, 3), dtype=np.uint8)
        image_array = batch['image'].numpy()
        cv2.namedWindow('Visualized Images', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Visualized Images', 192*4, 192*4)
        for i in range(64):
            for j in range(2):
                current_image = image_array[i, j]
                current_image = np.transpose(current_image, (1, 2, 0))
                current_image = (current_image * 255).astype(np.uint8)
                y_start = j * 96
                y_end = (j + 1) * 96
                result_image[y_start:y_end, :, :] = current_image
            cv2.imshow('Visualized Images', result_image)
        print("batch['action']：{batch['action']}")
        print("batch['action'].shape", batch['action'].shape)        # [64, 16, 2]
        print("batch['image'].shape:", batch['image'].shape)         # [64, 2, 3, 96, 96]
        print("batch['agent_pos'].shape:", batch['agent_pos'].shape) # [64, 2, 2]
        cv2.waitKey(0)
    cv2.destroyAllWindows()