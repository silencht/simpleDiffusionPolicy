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
# 4. Standard gym step method
obs, reward, terminated, truncated, info = env.step(action)


## Dataset
# download demonstration data from Google Drive
dataset_path = "pusht_cchi_v7_replay.zarr.zip"
if not os.path.isfile(dataset_path):
    id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    gdown.download(id=id, output=dataset_path, quiet=False)
# parameters
#|o|o|                             observations: 2 (包括image和agent_pos)
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
pred_horizon = 16                  #此为论文Fig.3中 Diffusion Policy的预测步数 T_{p} 
obs_horizon = 2                    #此为论文Fig.3中 输入Diffusion Policy的 latest T_{o}
action_horizon = 8                 #此为论文中的执行步数 T_{a}
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
vision_encoder = get_resnet('resnet18')
# IMPORTANT!
# replace all BatchNorm with GroupNorm to work with EMA, performance will tank if you forget to do this!
vision_encoder = replace_bn_with_gn(vision_encoder)
# ResNet18 has output dim of 512
vision_feature_dim = 512
# agent_pos is 2 dimensional
lowdim_obs_dim = 2
# observation feature has 514 dims in total per step
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
# DDPM Scheduler with 100 diffusion interations
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
_ = nets.to(device)


## Training
num_epochs = 100
# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75)
# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=1e-4, weight_decay=1e-6)
# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset, device transfer
                # 注: [:,:obs_horizon] 实际想做 pushTdataset.py 中 nsample['image'] = nsample['image'][:self.obs_horizon,:]做的事情，所以此处作用重复
                nimage = nbatch['image'][:,:obs_horizon].to(device)            # [64, 2, 3, 96, 96]
                nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)    # [64, 2, 2]
                naction = nbatch['action'].to(device)                          # [64, 16, 2]
                B = nagent_pos.shape[0]                                        # 64

                # encoder vision features, input var 'nimage.flatten(end_dim=1).shape' is [128,3,96,96]
                image_features = nets['vision_encoder'](nimage.flatten(end_dim=1)) # [128,512]
                # reshape input var 'nimage.shape[:2]' is [64,2]
                image_features = image_features.reshape(*nimage.shape[:2],-1)      # [64,2,512]
                # (B,obs_horizon,D)

                # concatenate vision feature and low-dim obs
                obs_features = torch.cat([image_features, nagent_pos], dim=-1)     # [64,2,514]
                obs_cond = obs_features.flatten(start_dim=1)                       # [64,2*514]
                # (B, obs_horizon * obs_dim)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)                  # [64, 16, 2]

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()                                                           # [64]

                # add noise to the clean images (not actions??) according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)                                     # [64, 16, 2]

                # predict the noise residual (with condion obs_cond)
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)                # [64, 16, 2]

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(nets.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))

# Weights of the EMA model
# is used for inference
ema_nets = nets
ema.copy_to(ema_nets.parameters())
print("Train End.")

# 保存参数模型到本地检查点文件
torch.save(ema_nets.state_dict(), "simpledp.ckpt")
print("Model parameters saved to 'simpledp.ckpt'.")