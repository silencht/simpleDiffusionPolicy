import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
from self_attention import PositionalEncoding
from transformers import AutoTokenizer, AutoModel


class image_command(nn.Module):
    """
    构造函数初始化模型参数。
    context_size: 用于上下文的观察数量。
    obs_encoder: 观察编码器的类型，例如"efficientnet-b0"。
    obs_encoding_size: 观察编码的维度大小。
    mha_num_attention_heads: 多头注意力机制中的头数。
    mha_num_attention_layers: 变压器中的层数。
    mha_ff_dim_factor: 前馈网络的维度因子。
    """
    def __init__(
        self,
        text_encode='',
        context_size: int = 2,
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:

        super().__init__()
        self.obs_encoding_size = obs_encoding_size  # 观察编码的维度。
        self.context_size = context_size # 上下文大小。

        # Initialize the observation encoder
        # 初始化观察编码器。
        self.obs_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3) # context
        # 替换EfficientNet中的批归一化（Batch Normalization）为组归一化（Group Normalization）。
        self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
        self.num_obs_features = self.obs_encoder._fc.in_features

        #初始化text编码器
        # 假定text_encode_model_path为你的模型路径
        text_encode_model_path = '/home/unitree/newHardDisk/bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(text_encode_model_path)
        self.model = AutoModel.from_pretrained(text_encode_model_path)

        self.text_compress_layer = nn.Linear(self.model.config.hidden_size, obs_encoding_size)

        # Initialize compression layers if necessary
        # 如果需要，初始化压缩层。
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        # Initialize positional encoding and self-attention layers
        # 初始化位置编码和自注意力层。
        # self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + 2)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size*2, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)


        # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
        # 定义目标掩码（约定：0 = 无掩码，1 = 掩码）。
        self.goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True # Mask out the goal  # 掩盖目标。
        self.no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool) 
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        # 定义平均池化掩码。
        self.avg_pool_mask = torch.cat([1 - self.no_mask.float(), (1 - self.goal_mask.float()) * ((self.context_size + 2)/(self.context_size + 1))], dim=0)
    """
    前向传播函数。
    obs_img: 观察图像。
    text_input: 指令
    在这个函数中，模型首先处理输入的观察图像和指令，提取特征，然后将这些特征进行融合。
    """

    def forward(self, obs_img: torch.tensor,text_inputs: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the observation encoding
        # 获取观察编码。
        batch_size, seq_len, c, h, w = obs_img.size()
        obs_img = obs_img.view(batch_size * seq_len, c, h, w)  # 调整形状
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape((-1, self.context_size, self.obs_encoding_size))

    # 如果提供了文本输入，处理文本以获取文本编码
        text_encodings=[]
        text_weight = torch.ones((batch_size, 1), device=obs_img.device)
        for idx, text_input in enumerate(text_inputs):
            if text_input:  # 如果文本输入存在
                inputs = self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=512).to(obs_img.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    text_feature = outputs.last_hidden_state[:, 0, :]  # 使用CLS token的输出作为特征
                text_encoding = self.text_compress_layer(text_feature)
            else:  # 如果文本输入为空
                text_encoding = torch.zeros((1, self.obs_encoding_size), device=obs_img.device)
                text_weight[idx] = 0
            text_encodings.append(text_encoding)
        text_features = torch.cat(text_encodings, dim=0).unsqueeze(1).expand(-1, self.context_size, -1)
        
        # 使用文本权重调整文本特征的贡献
        text_features = text_features * text_weight.unsqueeze(-1)
        combined_encoding = torch.cat([obs_encoding, text_features], dim=-1)
        
        # 应用自注意力编码器。
        obs_encoding_tokens = self.sa_encoder(combined_encoding)

        return obs_encoding_tokens



# Utils for Group Norm
'''
将神经网络中的所有批归一化（BatchNorm）层替换为组归一化（GroupNorm）层。
'''
"""
将所有批归一化（BatchNorm）层替换为组归一化（GroupNorm）层。
root_module: 需要修改的根模块。
features_per_group: 每个组的特征数，默认为16。
"""
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    # 调用 replace_submodules 函数来替换所有符合条件的子模块。
    replace_submodules(
        root_module=root_module,  # 要修改的根模块。
        predicate=lambda x: isinstance(x, nn.BatchNorm2d), # 判断子模块是否为 BatchNorm2d 类型。
        func=lambda x: nn.GroupNorm(        # 替换为 GroupNorm 的函数。
            num_groups=x.num_features//features_per_group, # 计算组的数量。
            num_channels=x.num_features) # 设置组归一化的通道数。
    )
    return root_module # 返回修改后的根模块。

"""
替换所有由predicate选择的子模块为func的输出。
root_module: 根模块。
predicate: 如果该函数返回True，则替换对应的模块。
func: 返回新的模块来替代。
replace_submodules 函数通过遍历神经网络中的所有子模块，并根据 predicate 函数指定的条件判断哪些子模块需要被替换。对于需要替换的子模块，它使用 func 函数提供的新模块进行替换。这种方法在需要批量修改模型结构时非常有用，例如在将BatchNorm层替换为GroupNorm层的场景中。
"""
def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    # 如果根模块符合predicate条件，直接替换。
    if predicate(root_module):
        return func(root_module)
    # 获取所有符合predicate条件的子模块。
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    # 遍历所有符合条件的子模块。
    for *parent, k in bn_list:
        parent_module = root_module
        # 获取子模块的父模块。
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
         # 获取源模块。
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        # 用func处理源模块，得到目标模块。
        tgt_module = func(src_module)
        # 替换原模块为目标模块。
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    # 验证所有模块是否已经被替换。
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0 # 确保没有剩余未替换的模块。
    return root_module # 返回修改后的根模块。




if __name__ == "__main__" :
    encoder = image_command()
    encoder.eval()
    device = 'cuda'
    encoder.to(device)

    import time

    time.sleep(3)
    batch_image = torch.randn([4,2, 3, 96, 96], device=device)
    text=["Passing through the left channel","Passing through the right channel","","Passing through the mid channel"]
    total_time = 0
    # print("显存使用概要（推理前）:")
    # print(torch.cuda.memory_summary(device=device, abbreviated=True))
    with torch.no_grad():
        for i in range(0, 100) :
            time.sleep(2)

            t1 = time.time()
            print(f"===ing==={i}")
            xx = encoder(batch_image,text)
            del xx
            torch.cuda.empty_cache()  # 清空 CUDA 缓存，释放未使用的显存
            t2 = time.time()

            total_time += (t2 - t1)
            # print("\n当前循环的显存使用概要：")
            # print(torch.cuda.memory_summary(device=device, abbreviated=True))
            print("------")
            time.sleep(1)
    print(f"t: {total_time / 100}")