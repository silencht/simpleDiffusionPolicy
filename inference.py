from pushTimageEnv import PushTImageEnv
from pushTdataset import PushTImageDataset, normalize_data , unnormalize_data
from network import get_resnet, replace_bn_with_gn, ConditionalUnet1D

import numpy as np
import torch
import torch.nn as nn
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm

from skvideo.io import vwrite
from IPython.display import Video
import gdown
import os


# parameters
#|o|o|                             observations: 2 (包括image和agent_pos)
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
pred_horizon = 16                  #此为论文Fig.3中 Diffusion Policy的预测步数 T_{p} 
obs_horizon = 2                    #此为论文Fig.3中 输入Diffusion Policy的 latest T_{o}
action_horizon = 8                 #此为论文中的执行步数 T_{a}

# download demonstration data from Google Drive
dataset_path = "pusht_cchi_v7_replay.zarr.zip"
if not os.path.isfile(dataset_path):
    id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    gdown.download(id=id, output=dataset_path, quiet=False)

# create dataset from file
dataset = PushTImageDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
# save training data statistics (min, max) for each dim，记录数据集的统计信息，用于对数据样本进行反归一化
stats = dataset.stats

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

# construct ResNet18 encoder
# if you have multiple camera views, use seperate encoder weights for each view.
# IMPORTANT! replace all BatchNorm with GroupNorm to work with EMA
# performance will tank if you forget to do this!
vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)

# ResNet18 has output dim of 512, agent_pos is 2 dimensional. observation feature has 514 dims in total per step
vision_feature_dim = 512
lowdim_obs_dim = 2
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 2

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# the final arch has 2 parts
nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
})

# for this demo, we use DDPMScheduler with 100 diffusion iterations
# 对应论文 III.D. "Accelerating Inference for Real-time Control"中
# The Denoising Diffusion Implicit Models (DDIM) approach [45] decouples the number of denoising iterations in training and inference, 
# thereby allowing the algorithm to use fewer iterations for inference to speed up the process.
# 将 num_diffusion_iters 数值修改小可以加速推理过程，代价是输出状态的不连续、不稳定性（vis.mp4结果视频观察得知）
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
device = torch.device('cuda')
ema_nets = nets.to(device)

load_pretrained = False
if load_pretrained:
    ckpt_path = "pusht_vision_100ep.ckpt"
    if not os.path.isfile(ckpt_path):
        id = "1XKpfNSlwYMGaF5CncoFaLKCDTWoLAHf1&confirm=t"
        gdown.download(id=id, output=ckpt_path, quiet=False)

    state_dict = torch.load(ckpt_path, map_location='cuda')
    ema_nets.load_state_dict(state_dict)
    print('Pretrained weights loaded.')
else:
    ckpt_path = "simpledp.ckpt"
    if not os.path.isfile(ckpt_path):
        print("No this ckpt File.")
    else:
        state_dict = torch.load(ckpt_path, map_location='cuda')
        ema_nets.load_state_dict(state_dict)
        print("Skipped pretrained weight loading.")


# limit enviornment interaction to 200 steps before termination
max_steps = 200
env = PushTImageEnv()
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(100000)
# get first observation
obs, info = env.reset()
# keep a queue of last 2 steps of observations
obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)                # maxlen=obs_horizon保证了队列长度恒定，添新移旧
# save visualization and rewards
imgs = list()
# imgs = [env.render(mode='rgb_array')]
rewards = list()
done = False
step_idx = 0

with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
    while not done:
        B = 1
        # stack the last obs_horizon number of observations
        images = np.stack([x['image'] for x in obs_deque])
        agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

        # normalize observation
        nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
        # images are already normalized to [0,1]
        nimages = images

        nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)           # (2,3,96,96)
        nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32) # (2,2)

        # infer action
        with torch.no_grad():
            # get image features
            image_features = ema_nets['vision_encoder'](nimages)                      # (2,512)

            # concat with low-dim observations
            obs_features = torch.cat([image_features, nagent_poses], dim=-1)          # (2,514)

            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)                 # (1,1028)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device)
            naction = noisy_action                                                    # (1,16,2)

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = ema_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()                                  # (1,16,2), that is (B, pred_horizon, action_dim)
        naction = naction[0]                                                          # (16,2)
        action_pred = unnormalize_data(naction, stats=stats['action'])                

        # only take action_horizon number of actions
        start = obs_horizon - 1              # 从最近的一次观测对应的action开始执行
        end = start + action_horizon         # 执行(action_horizon=)8步
        action = action_pred[start:end,:]    # 提取出要执行的action序列                  # (8,2), that is (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):         # 预测一次，执行action_horizon步，一共执行max_steps步
            # stepping env
            obs, reward, done, _, info = env.step(action[i])
            # save observations
            obs_deque.append(obs)
            # and reward/vis
            rewards.append(reward)
            imgs.append(env.render(mode='rgb_array'))

            # update progress bar
            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(reward=reward)
            if step_idx > max_steps:
                done = True
            if done:
                break

# print out the maximum target coverage
print('Score: ', max(rewards))

# visualize
from IPython.display import Video
vwrite('vis.mp4', imgs)
Video('vis.mp4', embed=True, width=256, height=256)