## Network
# 
#  Defines a 1D UNet architecture `ConditionalUnet1D`
#  as the noies prediction network
#  
#  Components
#  - `Downsample1d` Strided convolution to reduce temporal resolution
#  - `Upsample1d` Transposed convolution to increase temporal resolution
#  - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
#  - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`.
#  `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
#  `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.

from pos_Encoder import SinusoidalPosEmb
from typing import Union
import torch
import torch.nn as nn



class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)   # [2, 256, 512, 1024]
        start_dim = down_dims[0]                   # 256

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim          # 1284 = 256 + 1028

        in_out = list(zip(all_dims[:-1], all_dims[1:])) # [(2, 256), (256, 512), (512, 1024)]
        mid_dim = all_dims[-1]                     # 1024
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out): # for [(2, 256), (256, 512), (512, 1024)]
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])): # for [ (512, 1024), (256, 512)]
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        sample: (B,T,input_dim = action_dim) .  此为论文Fig.3中 输入Diffusion Policy的 Action Sequence A, [64, 16, 2]      
        timestep: (B,) or int, diffusion step . 此为论文Fig.3中 输入Diffusion Policy的 step k的位置编码特征 
        global_cond: (B,global_cond_dim) .      此为论文Fig.3中 输入Diffusion Policy的 Observation O_{t}
        output: (B,T,input_dim), this is noise prediction .   输出预测的噪声 \epsilon_{k}
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML (确保维度一致)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)                # [64,256]
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1) # [64,1284] ,there (256+1028=1284)

        x = sample   # [64, 2, 16]
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
        # [64,256,16]*2 -> [64,256,8] -> [64,512,8]*2 -> [64,512,4] -> [64,1024,4]*3

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        # [64,1024,4]*2

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        # [64,2048,4] -> [64,512,4]*2 -> [64,512,8] -> [64,1024,8] -> [64,256,8]*2 -> [64,256,16]

        x = self.final_conv(x) # [64,2,16]

        # (B,C,T)
        x = x.moveaxis(-1,-2)  # [64,16,2]
        # (B,T,C)
        return x


## Network Demo
if __name__ == "__main__":
    
    
    batch = 64
    vision_feature_dim = 512
    # agent_pos is 3 dimensional (x,y,z)
    agent_pos_dim = 2
    pred_horizon = 16
    obs_horizon = 2
    action_dim = 2
    # observation feature has 514 dims in total per step
    obs_dim = vision_feature_dim + agent_pos_dim

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # demo
    with torch.no_grad():
        image_features = torch.zeros((batch, obs_horizon, vision_feature_dim))      
        print(image_features.shape)                                                  # (64,2,514)
        agent_pos = torch.zeros((batch, obs_horizon, agent_pos_dim))
        print(agent_pos.shape)                                                       # (64,2,2)
        obs = torch.cat([image_features, agent_pos],dim=-1)     
        print(obs.shape)                                                             # (64,2,514)
        noised_action = torch.randn((batch, pred_horizon, action_dim))
        print(noised_action.shape)                                                   # (64,16,2)
        diffusion_iter = torch.arange(64)
        print(diffusion_iter)                                                        
        print(diffusion_iter.shape)                                                  # (64,)

        # the noise prediction network
        # takes noisy action, diffusion iteration and observation as input
        # predicts the noise added to action
        noise = noise_pred_net(
            sample=noised_action,
            timestep=diffusion_iter,
            global_cond=obs.flatten(start_dim=1))
        print(noise.shape)                                                           # (64,16,2)