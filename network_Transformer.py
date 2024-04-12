from typing import Union, Optional, Tuple
import torch
import torch.nn as nn
from pos_Encoder import SinusoidalPosEmb
    
class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            pred_horizon: int,        # 预测 T_{p} steps
            obs_horizon: int = None,  # 观测 O_{t} steps -> T_{o}
            cond_dim: int = 0,        # 观测 O_{t} dims
            n_layer: int = 8,         # TransformerDecoderLayer层数
            n_head: int = 4,          # the number of heads in the multiheadattention models
            n_emb: int = 256,         # 中间层宽度
            p_drop_emb: float = 0.0,  
            p_drop_attn: float = 0.01, 
            causal_attn: bool=True,   # 因果注意力掩码
            time_as_cond: bool=True,  # timestep是否作为条件
            obs_as_cond: bool=False,  # 观测是否作为条件
            n_cond_layers: int = 0    # TransformerEncoderLayer层数
        ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if obs_horizon is None:
            obs_horizon = pred_horizon
        
        T = pred_horizon            
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += obs_horizon

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # global_cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4*n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            # when used without it, the model "cheats" by looking ahead into future end-effector poses, which is almost identical 
            # to the action of the current timestep. (https://github.com/real-stanford/diffusion_policy/issues/12)
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            
            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s-1) # add one dimension since time is the first token in global_cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.pred_horizon = pred_horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
        print("number of TransformerForDiffusion parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
                        SinusoidalPosEmb, 
                        nn.TransformerEncoderLayer, 
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-1,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        global_cond: Optional[torch.Tensor]=None, **kwargs):
        """
        sample: (B,T,input_dim = action_dim) .  此为论文Fig.3中 输入Diffusion Policy的 Action Sequence A   
        timestep: (B,) or int, diffusion step . 此为论文Fig.3中 输入Diffusion Policy的 step k的位置编码特征 
        global_cond: (B,To,cond_dim)            此为论文Fig.3中 输入Diffusion Policy的 Observation O_{t}
        output: (B,T,input_dim), this is noise prediction .   输出预测的噪声 \epsilon_{k}
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)                   # (256,1,256)

        # process input
        input_emb = self.input_emb(sample)                                 # (256,16,256)

        if self.encoder_only:
            # BERT
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T+1,n_emb)
            x = self.encoder(src=x, mask=self.mask)
            # (B,T+1,n_emb)
            x = x[:,1:,:]
            # (B,T,n_emb)
        else:
            # encoder
            cond_embeddings = time_emb                                     # (256,1,256)
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(global_cond)              # (B,To,n_emb) (256,2,514) -> (256,2,256)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1) # (256,3,256)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[:, :tc, :]  # each position maps to a (learnable) vector
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x   # (B,T_cond,n_emb), T_cond = To + 1(此为timestep嵌入维度)
            
            # decoder
            token_embeddings = input_emb                                   # (256,16,256)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask
            )
            # (B,T,n_emb)
        
        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x



if __name__ == "__main__":

    # GPT with time embedding
    # transformer = TransformerForDiffusion(
    #     input_dim=16,
    #     output_dim=18,
    #     pred_horizon=8,
    #     obs_horizon=3,
    #     # cond_dim=10,
    #     causal_attn=True,
    #     # time_as_cond=False,
    #     # n_cond_layers=4
    # )
    # opt = transformer.configure_optimizers()

    # timestep = torch.tensor([0,1,2])          # 观测序列对应的step
    # sample = torch.zeros((3,8,16))            # 观测序列3个，预测8个动作，动作维度16
    # out = transformer(sample, timestep)
    # # print(timestep.shape)
    # # print(sample.shape)
    # # print(out.shape)

    

    # GPT with time embedding and obs global_cond
    transformer = TransformerForDiffusion(
        input_dim=2,
        output_dim=2,
        pred_horizon=16,
        obs_horizon=2,
        cond_dim=1028,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.randint(0,1,(256,))
    sample = torch.zeros((256,16,2))
    global_cond = torch.zeros((256,2,1028))
    out = transformer(sample, timestep, global_cond)

    # # GPT with time embedding and obs global_cond and encoder
    # transformer = TransformerForDiffusion(
    #     input_dim=16,
    #     output_dim=16,
    #     pred_horizon=8,
    #     obs_horizon=4,
    #     cond_dim=10,
    #     causal_attn=True,
    #     # time_as_cond=False,
    #     n_cond_layers=4
    # )
    # opt = transformer.configure_optimizers()
    
    # timestep = torch.tensor(0)
    # sample = torch.zeros((4,8,16))
    # global_cond = torch.zeros((4,4,10))
    # out = transformer(sample, timestep, global_cond)

    # # BERT with time embedding token
    # transformer = TransformerForDiffusion(
    #     input_dim=16,
    #     output_dim=16,
    #     pred_horizon=8,
    #     obs_horizon=4,
    #     # cond_dim=10,
    #     # causal_attn=True,
    #     time_as_cond=False,
    #     # n_cond_layers=4
    # )
    # opt = transformer.configure_optimizers()

    # timestep = torch.tensor(0)
    # sample = torch.zeros((4,8,16))
    # out = transformer(sample, timestep)