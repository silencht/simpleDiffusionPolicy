# local file import
from pushTimageEnv import PushTImageEnv
from pushTdataset import PushTImageDataset, gdown
from network import get_resnet, replace_bn_with_gn, ConditionalUnet1D
# diffusion policy import
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
# basic library
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import os


## Create Env
# Standard Gym Env (0.21.0 API)
# 0. create env object
env = PushTImageEnv()
# 1. seed env for initial state.
# Seed 0-200 are used for the demonstration dataset.
env.seed(1000)
# 2. must reset before use
obs, info = env.reset()
# 3. 2D positional action space [0,512]
action = env.action_space.sample()
print(action)
# 4. Standard gym step method
obs, reward, terminated, truncated, info = env.step(action)


## Dataset
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
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)


## Network
# construc ResNet18 encoder
# 构建基于ResNet18的视觉编码器
vision_encoder = get_resnet('resnet18')
# IMPORTANT!
# replace all BatchNorm with GroupNorm to work with EMA, performance will tank if you forget to do this!
# 重要：将所有BatchNorm层替换为GroupNorm层，以适配EMA训练，如果忘记这一步，性能会显著下降
vision_encoder = replace_bn_with_gn(vision_encoder)
# ResNet18 has output dim of 512
# ResNet18的输出维度为512
vision_feature_dim = 512
# agent_pos is 2 dimensional
# agent_pos是二维的
lowdim_obs_dim = 2
# observation feature has 514 dims in total per step
# 观测特征总共有514维，每个时间步
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 2
# create network object
# 创建网络对象，ConditionalUnet1D用于预测噪声
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)
# the final arch has 2 parts
# 最终的架构有两个部分：视觉编码器和噪声预测网络
nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
})
# DDPM Scheduler with 100 diffusion interations
# 使用DDPM调度器，设置扩散迭代次数为100
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    # beta调度策略对性能影响很大，发现squared cosine效果最好
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    # 将输出剪切到[-1,1]以提高稳定性
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    # 我们的网络预测的是噪声（而不是去噪后的动作）
    prediction_type='epsilon'
)
# device transfer
# 转移网络到cuda设备
device = torch.device('cuda')
_ = nets.to(device)


## Training
# 设置训练轮次为100
num_epochs = 100
# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
# 指数移动平均（EMA）加速训练并提高稳定性，它保持了模型权重的一个副本
ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75)
# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
# 标准的ADAM优化器，注意EMA参数不参与优化
optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=1e-4, weight_decay=1e-6)
# Cosine LR schedule with linear warmup
# 余弦学习率调度，包含线性预热
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)
# 训练轮次循环
with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    # 数据批次循环
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                # 数据在数据集中已归一化，进行设备转移
                nimage = nbatch['image'][:,:obs_horizon].to(device)            # [64, 2, 3, 96, 96], there [:,:obs_horizon] maybe do nothing.
                nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)    # [64, 2, 2]
                # print(f'nagent_pos: {nagent_pos}')
                naction = nbatch['action'].to(device)  
                # print(f'naction: {naction}')                        # [64, 16, 2]
                B = nagent_pos.shape[0]                                        # 64

                # encoder vision features, input nimage.flatten(end_dim=1).shape is [128,3,96,96]
                # 编码视觉特征，输入nimage展平后的形状是[128,3,96,96]
                image_features = nets['vision_encoder'](nimage.flatten(end_dim=1)) # [128,512]
                # reshape input nimage.shape[:2] is [64,2]
                # 调整形状，输入nimage的形状[:2]是[64,2]
                image_features = image_features.reshape(*nimage.shape[:2],-1)      # [64,2,512]
                # (B,obs_horizon,D)
                # (B,obs_horizon,D)

                # concatenate vision feature and low-dim obs
                # 将视觉特征和低维观测数据（agent_pos）拼接，形成完整的观测条件
                obs_features = torch.cat([image_features, nagent_pos], dim=-1)     # [64,2,514]
                # 将观测数据扁平化，用作条件U-Net的全局条件，形状变为 [64, obs_horizon * obs_dim]
                obs_cond = obs_features.flatten(start_dim=1)                       # [64,2*514]
                # (B, obs_horizon * obs_dim)
                # sample noise to add to actions
                # 为动作序列样本添加噪声，这是扩散过程的一部分
                noise = torch.randn(naction.shape, device=device)                  # [64, 16, 2]

                # sample a diffusion iteration for each data point
                 # 为每个数据点随机选择一个扩散时间步
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()                                                           # [64]

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                # 根据每个扩散迭代的噪声强度，将噪声添加到干净的动作序列中
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)                                     # [64, 16, 2]

                # predict the noise residual (with condion obs_cond)
                # 使用噪声预测网络预测噪声残差，作为模型的输出
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)                # [64, 16, 2]

                # L2 loss
                # 使用L2损失函数（均方误差损失）计算预测噪声与真实噪声之间的误差
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                # 反向传播优化
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                # 更新指数移动平均（EMA）模型的参数
                ema.step(nets.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))

# Weights of the EMA model
# is used for inference
# 训练结束后，使用EMA模型的参数进行推理
ema_nets = nets
ema.copy_to(ema_nets.parameters())
print("Train End.")

# 保存参数模型到本地检查点文件
torch.save(ema_nets.state_dict(), "simpledp.ckpt")
print("Model parameters saved to 'simpledp.ckpt'.")