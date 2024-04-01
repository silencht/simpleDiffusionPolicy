import torch
import torch.nn as nn
import torch.nn.functional as F
import math
'''
这个 PositionalEncoding 类是一个 PyTorch 神经网络模块，用于向序列模型（如 Transformer）添加位置编码信息。位置编码对于序列模型来说至关重要，因为它们提供了序列中各个元素的位置信息，这对于模型正确理解序列中元素的顺序和相对位置非常重要。

在构造函数 __init__ 中，首先计算位置编码。这个计算基于正弦和余弦函数，其中每个位置的编码值取决于它在序列中的位置。
pos_enc 是一个二维张量，其形状为 max_seq_len（序列的最大长度）乘以 d_model（模型的维度）。
pos 是一个一维张量，包含了从 0 到 max_seq_len - 1 的位置索引。
div_term 用于调整正弦和余弦函数的频率，它是基于模型维度的函数。
位置编码的奇数列使用正弦函数计算，偶数列使用余弦函数计算。
使用 self.register_buffer 将 pos_enc 注册为模型的一个缓存，这意味着它不会被视为模型参数。
在 forward 方法中，位置编码被加到输入 x 上。这个加法操作使每个元素的表示包含了其位置信息。
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        # 一次性计算位置编码
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        # 将位置编码注册为一个缓存，以避免它被当作模型参数保存
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        # 将位置编码加到输入上
        x = x + self.pos_enc[:, :x.size(1), :]
        return x
'''
这个 MultiLayerDecoder 类是一个 PyTorch 神经网络模块，用于构建一个多层解码器，其中包含位置编码、自注意力（Self-Attention）层和一系列全连接层。该解码器适用于处理序列数据，特别是在需要对序列特征进行复杂转换的场景中。

在构造函数 __init__ 中，首先创建一个 PositionalEncoding 对象，用于向输入添加位置信息。
然后，构建一个自注意力编码器层 self.sa_layer，该层使用了 Transformer 架构中的自注意力机制。
接下来，使用这个自注意力层创建一个自注意力编码器 self.sa_decoder，它包含多个这样的层。
之后，初始化一系列全连接层 self.output_layers，用于将自注意力编码器的输出进一步处理成所需维度的输出。
在 forward 方法中，首先应用位置编码，然后通过自注意力编码器处理数据，随后将数据通过一系列全连接层，每层之后应用ReLU激活函数。
'''
class MultiLayerDecoder(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder, self).__init__()
        # 添加位置编码
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        # 创建一个自注意力编码器层
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        # 基于上面的层创建一个自注意力编码器
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        # 初始化输出层，首先将序列维度展平
        self.output_layers = nn.ModuleList([nn.Linear(seq_len*embed_dim, embed_dim)])
        # 添加更多的线性层
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, x):
        # 如果存在位置编码，则首先应用位置编码
        if self.positional_encoding: x = self.positional_encoding(x)
        # 通过自注意力编码器处理输入
        x = self.sa_decoder(x)
        # currently, x is [batch_size, seq_len, embed_dim]
        # 将输入从 [batch_size, seq_len, embed_dim] 变形为 [batch_size, seq_len * embed_dim]
        x = x.reshape(x.shape[0], -1)
        # 通过一系列的线性层
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x
