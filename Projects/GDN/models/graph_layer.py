"""
定义了一个基于 图注意力机制（Graph Attention Network, GAT） 的自定义图卷积层GraphLayer，
继承自 PyTorch Geometric 的MessagePassing类。
该层结合了节点特征和节点嵌入（embedding），实现了带注意力机制的消息传递
"""

import torch
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
import time
import math

class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1,**kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs)  # 聚合方式为求和

        self.in_channels = in_channels  # 输入特征维度
        self.out_channels = out_channels  # 输出特征维度（单头）
        self.heads = heads  # 注意力头数，用于多头注意力机制
        self.concat = concat  # 是否拼接多头输出（若为False，则取平均）
        self.negative_slope = negative_slope  # LeakyReLU 中负半轴的斜率
        self.dropout = dropout  # 注意力权重的 Dropout 率

        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)  # 线性变换层

        # 注意力权重参数（用于节点特征和嵌入）
        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))  # 源节点注意力参数
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))  # 目标节点注意力参数
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))  # 源节点嵌入的注意力参数
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))  # 目标节点嵌入的注意力参数

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))  # 偏置项, 可选，根据concat参数决定维度
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        
        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias)



    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        """
            前向传播。
            关键步骤：
                线性变换：将输入特征投影到多头空间，形状为(N, heads*out_channels)。
                边索引处理：确保图包含自环（节点与自身连接），便于捕捉自身特征。
                消息传递：通过propagate方法自动调度消息传递流程，核心逻辑在message函数中。
        """
        # 1. 线性变换：将输入特征映射到多头维度
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)  # 源节点和目标节点特征相同（同构图）
        else:
            x = (self.lin(x[0]), self.lin(x[1]))   # 异构图处理

        # 2. 处理边索引：移除自环并添加自环
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))

        # 3. 消息传递：调用propagate触发message和aggregate函数
        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,
                             return_attention_weights=return_attention_weights)

        # 4. 处理多头输出
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)  # 拼接多头
        else:
            out = out.mean(dim=1)  # 平均多头

        # 5. 添加偏置
        if self.bias is not None:
            out = out + self.bias

        # 6. 返回结果和注意力权重（可选）
        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i,
                embedding,
                edges,
                return_attention_weights):

        # 1. 调整特征形状：(N, heads, out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)  # 源节点特征
        x_j = x_j.view(-1, self.heads, self.out_channels)  # 目标节点特征

        # 2. 处理节点嵌入（若存在）
        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            embedding_i = embedding_i.unsqueeze(1).repeat(1,self.heads,1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1,self.heads,1)

            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

        # 3. 计算注意力分数
        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)  # 拼接特征和嵌入的注意力参数
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)  # 点积计算注意力分数


        alpha = alpha.view(-1, self.heads, 1)

        # 4. 激活函数与归一化
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)  # 按源节点归一化

        # 5. Dropout与保存注意力权重（可选）
        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 6. 消息传递：目标节点特征乘以注意力权重
        return x_j * alpha.view(-1, self.heads, 1)



    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
