from __future__ import print_function
from __future__ import division
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GATConv
import math


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ') '

class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class EncoderP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout):
        super(EncoderP, self).__init__()
        self.gc1 = GATConv(input_dim, hidden_dim1)
        self.gc2 = GATConv(hidden_dim1, hidden_dim2)
        self.gc3 = GATConv(hidden_dim1, hidden_dim2)
        self.gc1_m = GraphConvolution(input_dim, hidden_dim1)
        self.gc2_m = GraphConvolution(hidden_dim1, hidden_dim2)
        self.gc3_m = GraphConvolution(hidden_dim1, hidden_dim2)
        self.act = nn.ReLU()
        self.dropout = dropout

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, edge_index, adj):
        x_l = self.act(self.gc1(x, edge_index))
        x_l = F.dropout(x_l, p=self.dropout, training=self.training)
        mu = self.gc2(x_l, edge_index)
        logvar = self.gc3(x_l, edge_index)
        x_h = self.reparameterize(mu, logvar)
        x_l_m = self.act(self.gc1_m(x, adj))
        x_l_m = F.dropout(x_l_m, p=self.dropout, training=self.training)
        mu_m = self.gc2_m(x_l_m, adj)
        logvar_m = self.gc3_m(x_l_m, adj)
        x_h_m = self.reparameterize(mu_m, logvar_m)
        return x_l, x_h, mu, logvar, x_l_m, x_h_m, mu_m, logvar_m


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout):
        super(Encoder, self).__init__()
        self.gc1 = GATConv(input_dim, hidden_dim1)
        self.gc2 = GATConv(hidden_dim1, hidden_dim2)
        self.gc3 = GATConv(hidden_dim1, hidden_dim2)
        self.act = nn.ReLU()
        self.dropout = dropout

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, edge_index):
        x_l = self.act(self.gc1(x, edge_index))
        x_l = F.dropout(x_l, p=self.dropout, training=self.training)
        mu = self.gc2(x_l, edge_index)
        logvar = self.gc3(x_l, edge_index)
        x_h = self.reparameterize(mu, logvar)
        return x_l, x_h, mu, logvar

class Cycler(nn.Module):
    def __init__(self, input_num, input_dim, hidden_dim, out_num, dropout):
        super(Cycler, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.trs = nn.Linear(input_num, out_num)
        self.fc2 = GATConv(hidden_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(out_num)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.adain1 = AdaIN(64, 512, 64)
        self.adain2 = AdaIN(256, 512, 64)
        self.adain3 = AdaIN(64, 512, 64)
        self.dropout = dropout

    def forward(self, x, edge_index, style1, style2):
        x = x.T
        x = self.act(self.bn2(self.trs(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.T
        x = self.adain1(x, style1, style2)
        x = self.act(self.bn1(self.fc1(x)))
        x = self.adain2(x, style1, style2)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act(self.fc2(x, edge_index))
        x = self.adain3(x, style1, style2)
        return x


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.alpha
        return grad_output, None

class GRL(nn.Module):
    def forward(self, input, alpha=1.0):
        return GradReverse.apply(input, alpha)

class DD(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(DD, self).__init__()
        self.grl = GRL()
        self.gc1 = GATConv(input_dim, hidden_dim)
        self.gc2 = GATConv(hidden_dim, 2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, alpha):
        x = self.grl(x, alpha)
        edge_index = self.grl(edge_index, alpha)
        x = self.act(self.gc1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return x


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super(EqualLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = self.activation(out)
        else:
            if self.bias is not None:
                out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
            else:
                out = F.linear(input, self.weight * self.scale, bias=None)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')

class Style(nn.Module):
    def __init__(self, style_dim, lr_mlp, dropout): # lr_mlp=0.01
        super(Style, self).__init__()
        self.fc1 = EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation=True)
        self.fc2 = EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation=True)
        self.fc3 = EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation=True)
        self.fc4 = EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation=True)
        self.pn = PixelNorm()
        self.dropout = dropout

    def forward(self, x):
        x = self.pn(x)
        x = self.fc1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc4(x)
        # x = x
        return x

class AdaIN(nn.Module):
    def __init__(self, input_dim, style_dim1, style_dim2):
        super(AdaIN, self).__init__()
        # 通过风格向量控制的两组参数（gamma和beta）
        # gamma: 用于调节标准差，beta: 用于调节均值
        self.style_fc1 = nn.Linear(style_dim1, input_dim)
        self.style_fc2 = nn.Linear(style_dim1, input_dim)
        self.style_fc3 = nn.Linear(style_dim2, input_dim)
        self.style_fc4 = nn.Linear(style_dim2, input_dim)

    def forward(self, x, style1, style2):
        # x: 输入特征向量 [batch_size, input_dim]
        # style: 风格向量 [batch_size, style_dim]

        # 计算均值和标准差
        if style1 != None and style2 !=None:
            mean = torch.mean(x, dim=1, keepdim=True)  # 对每个样本的特征进行均值计算
            std = torch.std(x, dim=1, keepdim=True)  # 对每个样本的特征进行标准差计算

            # 计算gamma和beta
            gamma1 = self.style_fc1(style1).view(-1, x.size(1))  # [batch_size, input_dim]
            beta1 = self.style_fc2(style1).view(-1, x.size(1))  # [batch_size, input_dim]
            gamma2 = self.style_fc3(style2).view(-1, x.size(1))  # [batch_size, input_dim]
            beta2 = self.style_fc4(style2).view(-1, x.size(1))  # [batch_size, input_dim]

            # 归一化
            x_normalized = (x - mean) / (std + 1e-8)

            # 应用gamma和beta调整特征向量
            out = (0.5*gamma1 + 0.5*gamma2) * x_normalized + (0.5*beta1 + 0.5*beta2)
        else:
            # mean = torch.mean(x, dim=1, keepdim=True)  # 对每个样本的特征进行均值计算
            # std = torch.std(x, dim=1, keepdim=True)  # 对每个样本的特征进行标准差计算
            # out = (x - mean) / (std + 1e-8)
            out = x
        return out


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)


    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


class EncoderS(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout):
        super(EncoderS, self).__init__()
        self.gc1_o = GATConv(input_dim, hidden_dim1)
        self.gc2_o = GATConv(hidden_dim1, hidden_dim2)
        self.gc3_o = GATConv(hidden_dim1, hidden_dim2)
        self.gc1_m = GraphConvolution(input_dim, hidden_dim1)
        self.gc2_m = GraphConvolution(hidden_dim1, hidden_dim2)
        self.gc3_m = GraphConvolution(hidden_dim1, hidden_dim2)
        self.act = nn.ReLU()
        self.dropout = dropout

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, edge_index, adj):
        x_l = self.act(self.gc1_o(x, edge_index))
        x_l = F.dropout(x_l, p=self.dropout, training=self.training)
        mu = self.gc2_o(x_l, edge_index)
        logvar = self.gc3_o(x_l, edge_index)
        x_h = self.reparameterize(mu, logvar)
        x_l_m = self.act(self.gc1_m(x, adj))
        x_l_m = F.dropout(x_l_m, p=self.dropout, training=self.training)
        mu_m = self.gc2_m(x_l_m, adj)
        logvar_m = self.gc3_m(x_l_m, adj)
        x_h_m = self.reparameterize(mu_m, logvar_m)
        return x_l, x_h, mu, logvar, x_l_m, x_h_m, mu_m, logvar_m











