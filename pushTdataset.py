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

import cv2
import numpy as np
import zarr
import torch
import gdown
import os


def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    '''
    episode_ends: 一个 NumPy 数组，包含每个episode的结束索引
    sequence_length: 一个整数，指定要从每个episode中提取的序列的长度
    pad_before: 一个整数，可选参数，默认值为0，表示在序列开始前添加的填充数量
    pad_after: 一个整数，可选参数，默认值为0，表示在序列结束后添加的填充数量
    '''
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    '''
    if 4 idx input parameters is [n, n+15, 1, 16], that is "sample_start_idx > 0":
        data的16个元素中,采样区间只有15个元素，将采样区第一个元素复制填充到第1个元素位置，i.e. [x,x,a,b,c,……]
    if 4 idx input parameters is [x, m, 0, y<16], that is "sample_end_idx < sequence_length":
        data的16个元素中,采样区间少于16个元素，将采样区最后的元素复制填充到后续缺少位置， i.e. [a,b,c,……,x,x,x]
    作者对此填充原因的说明：https://github.com/real-stanford/diffusion_policy/issues/30
    '''
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# dataset
class PushTImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')

        # pusht_cchi_v7_replay.zarr file directory tree
        # ├── data
        # │   ├── action (25650, 2)      float32
        # │   ├── img (25650, 96, 96, 3) float32
        # │   ├── keypoint (25650, 9, 2) float32
        # │   ├── n_contacts (25650, 1)  float32
        # │   └── state (25650, 5)       float32
        # └── meta
        #     └── episode_ends (206,)    int64

        train_image_data = dataset_root['data']['img'][:]       # [25650,96,96,3]
        train_image_data = np.moveaxis(train_image_data, -1,1)  # [25650,3,96,96]

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'agent_pos': dataset_root['data']['state'][:,:2],   # [25650,2]
            'action': dataset_root['data']['action'][:]         # [25650,2]
        }
        # print(dataset_root['data']['state'][0,:]) output below:
        #          [222., 97., 222.99382, 381.59903, 3.0079994]
        # 猜测意义    x  ,  y ,  z      , orientation, claw state
        # print(dataset_root['data']['action'][0,:]) output below:
        #          [233.  71.]

        episode_ends = dataset_root['meta']['episode_ends'][:]  # [206,]
        # print(episode_ends), ouput below:
        # [   161   279   420   579   738   895   964  1133  1213  1347  1535  1684
        #     1824  1949  ...   ...    ...   ...   ...   ...  ...   ...   ...   ...
        #     ...   ...   ...   ...    ...   ...   ...   ...  ...   ...   ...   ...
        #     25601 25650]

        # compute start and end of each state-action sequence, also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,  # 16
            pad_before=obs_horizon-1,      # 2-1
            pad_after=action_horizon-1)    # 8-1
        # print(indices), output below:
        # format: [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
        # ( [[    0,    15,     1,    16],
        #    [    0,    16,     0,    16],
        #    [    1,    17,     0,    16],
        #    [    2,    18,     0,    16],
        #    ...,
        #    [  151,   161,     0,    10],
        #    [  152,   161,     0,     9],
        #    [  161,   176,     1,    16],
        #    [  162,   178,     0,    16],
        #    ...,
        #    [25639, 25650,     0,    11],
        #    [25640, 25650,     0,    10],
        #    [25641, 25650,     0,     9]])
        # print(indices.shape) output is [24208, 4]

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data['image'] = train_image_data
        # normalized_train_data['image'].shape    =[25650,3,96,96]
        # normalized_train_data['agent_pos'].shape=[25650,2]
        # normalized_train_data['action'].shape   =[25650,2]

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # print(len(self.indices)) output is 24208
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

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
        nsample['image'] = nsample['image'][:self.obs_horizon,:]          # 只保留前obs_horizon张image
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon,:]  # 只保留前obs_horizon个agent_pos
        # print(nsample['image'].shape)      = [2,3,96,96]
        # print(nsample['agent_pos'])        = [2,2]
        # print(nsample['action'].shape)     = [16,2]
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
        image_array   = batch['image'].numpy()                                                        # [64,2,3,96,96]
        agent_pos_arr = unnormalize_data(batch['agent_pos'], stats=dataset.stats['agent_pos']) / 5.34 # [64,2,2]
        action_arr    = unnormalize_data(batch['action'], stats=dataset.stats['action']) /5.34        # [64,16,2]        [512,512] / [96,96] = 5.34
        cv2.namedWindow('Visualized Images', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Visualized Images', 192*4, 192*4)
        for i in range(64):
            # 绘制 image
            for j in range(2):
                current_image = image_array[i, j]
                current_image = np.transpose(current_image, (1, 2, 0))
                current_image = (current_image * 255).astype(np.uint8)
                y_start = j * 96
                y_end = (j + 1) * 96
                result_image[y_start:y_end, :, :] = current_image

            # 绘制 agent_pos
            for k in range(2):
                agent_pos = agent_pos_arr[i, k]
                x_pos, y_pos = int(agent_pos[0]), int(agent_pos[1])
                result_image[y_pos-1:y_pos+1, x_pos-1:x_pos+1, :] = [0, 255, 0]    # 绿色
                result_image[y_pos+95:y_pos+97, x_pos-1:x_pos+1, :] = [0, 255, 0]  # 绿色

            # 绘制 action
            for l in range(16):
                action_pos = action_arr[i, l]
                x_pos, y_pos = int(action_pos[0]), int(action_pos[1])
                result_image[y_pos-1:y_pos+1, x_pos-1:x_pos+1, :] = [0, 0, 255]    # 红色
                result_image[y_pos+95:y_pos+97, x_pos-1:x_pos+1, :] = [0, 0, 255]  # 红色

            cv2.imshow('Visualized Images', result_image)
            cv2.waitKey(100)
    cv2.destroyAllWindows()