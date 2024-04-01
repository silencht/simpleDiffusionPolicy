import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
from self_attention import PositionalEncoding
from transformers import AutoTokenizer, AutoModel

from network import get_resnet, replace_bn_with_gn, ConditionalUnet1D
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
        self.vision_encoder = get_resnet('resnet18')
        self.vision_encoder = replace_bn_with_gn(self.vision_encoder)

        #初始化text编码器
        # 假定text_encode_model_path为你的模型路径
        text_encode_model_path = '/home/unitree/newHardDisk/bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(text_encode_model_path)
        self.model = AutoModel.from_pretrained(text_encode_model_path)
        self.text_compress_layer = nn.Linear(self.model.config.hidden_size, obs_encoding_size)


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
        obs_img = obs_img.flatten(end_dim=1)                  #由[4,2,3,96,96]-->[8,3,96,96]
        image_features =self.vision_encoder(obs_img) # [128,512]
        # reshape input nimage.shape[:2] is [64,2]
        # 调整形状，输入nimage的形状[:2]是[64,2]
        obs_encoding = image_features.reshape((-1, self.context_size, self.obs_encoding_size))
    # 如果提供了文本输入，处理文本以获取文本编码
        text_weight = torch.zeros((batch_size, 1), device=obs_img.device)
        text_encodings = torch.zeros((batch_size, self.obs_encoding_size), device=obs_img.device)
        for idx, text_input in enumerate(text_inputs):
            if text_input:  # 如果文本输入存在
                inputs = self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=512).to(obs_img.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    text_feature = outputs.last_hidden_state[:, 0, :]  # 使用CLS token的输出作为特征
                text_encodings[idx] = self.text_compress_layer(text_feature)
                text_weight[idx] = 1
        text_features = text_encodings.unsqueeze(1).expand(-1, self.context_size, -1)
        
        # 使用文本权重调整文本特征的贡献
        text_features = text_features * text_weight.unsqueeze(-1)
        combined_encoding = torch.cat([obs_encoding, text_features], dim=-1)
        
        # 应用自注意力编码器。
        obs_encoding_tokens = self.sa_encoder(combined_encoding)

        return obs_encoding_tokens




if __name__ == "__main__" :
    encoder = image_command()
    encoder.eval()
    device = 'cuda'
    encoder.to(device)

    import time

    time.sleep(3)
    batch_image = torch.randn([4,2, 3, 96, 96], device=device)
    text=[]
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