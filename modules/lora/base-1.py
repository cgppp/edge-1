# encoding: utf-8
import torch
from math import sqrt
from torch import nn
from torch.nn import functional as nnFunc
# 引入一个上下文管理器，用于在不跟踪梯度的情况下执行代码块
from utils.torch.comp import torch_no_grad

class Linear(nn.Linear):
    """
    支持 LoRA 的线性层。
    继承自 torch.nn.Linear，但在前向传播时额外增加了低秩适配器的计算。
    """
    def __init__(self, in_features, out_features, bias=True, lora_features=None, lora_alpha=None, scaling=1.0, update_bias=True, **kwargs):
        # 初始化标准的 Linear 层
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        
        # 冻结原始权重，不参与梯度更新
        self.weight.requires_grad_(False)
        self.update_bias = update_bias
        
        # 如果存在偏置项，根据配置决定是否可训练
        if self.bias is not None:
            self.bias.requires_grad_(update_bias)
        
        # 设置 LoRA 的秩 (lora_features) 和缩放系数 (scaling)
        # 如果提供了 lora_alpha，则 scaling = alpha / rank，否则使用默认 scaling
        self.lora_features, self.scaling = lora_features, (scaling if lora_alpha is None else float(lora_alpha) / float(lora_features))
        
        # 初始化 LoRA 的两个低秩矩阵 A 和 B
        # lora_wa: 形状为 (in_features, lora_features)，使用均匀分布初始化
        self.lora_wa = nn.Parameter(torch.empty(in_features, lora_features, dtype=self.weight.dtype, device=self.weight.device))
        # lora_wb: 形状为 (lora_features, out_features)，初始化为零，确保初始状态等效于恒等映射（不加额外输出）
        self.lora_wb = nn.Parameter(torch.zeros(lora_features, out_features, dtype=self.weight.dtype, device=self.weight.device))
        
        # 保存重置参数的方法引用
        self.fix_init = self.reset_parameters
        # 执行参数重置（包括原始权重和 LoRA 权重）
        self.reset_parameters()

    def forward(self, x, **kwargs):
        # 1. 计算原始线性层的输出：y = x * W^T + b
        out = nnFunc.linear(x, self.weight, self.bias)
        
        # 2. 计算 LoRA 分支的输出：delta = x * A * B * scaling
        # x.view(-1, x.size(-1)): 将输入展平为二维以便矩阵乘法
        # .mm(self.lora_wa).mm(self.lora_wb): 依次乘以矩阵 A 和 B
        # .view(out.size()): 恢复形状以匹配原始输出
        # out.add_(..., alpha=self.scaling): 原地加法，加上缩放后的 LoRA 增量
        out.add_(x.view(-1, x.size(-1)).mm(self.lora_wa).mm(self.lora_wb).view(out.size()), alpha=self.scaling)
        return out

    def reset_parameters(self):
        """重置原始权重和偏置"""
        with torch_no_grad():
            # 使用 Xavier 初始化的一种变体
            _ = 1.0 / sqrt(self.weight.size(-1))
            self.weight.uniform_(-_, _)
            if self.bias is not None:
                self.bias.zero_()
        # 初始化 LoRA 参数
        self.init_lora()

    def init_lora(self):
        """专门初始化 LoRA 矩阵"""
        with torch_no_grad():
            _ = 1.0 / sqrt(self.weight.size(-1))
            # 矩阵 A 使用均匀分布初始化
            if hasattr(self, "lora_wa"):
                self.lora_wa.uniform_(-_, _)
            # 矩阵 B 初始化为 0，保证训练开始时 LoRA 不产生额外影响
            if hasattr(self, "lora_wb"):
                self.lora_wb.zero_()

    def acc_lora(self):
        """
        将 LoRA 的增量合并回原始权重中。
        公式: W_new = W_old + A * B^T
        这通常用于推理阶段，以减少计算开销。
        """
        with torch_no_grad():
            # 将 A * B 的结果加到原始 weight 上 (注意转置以匹配维度)
            self.weight.add_(self.lora_wa.mm(self.lora_wb).t())
        # 合并后重新初始化 LoRA 矩阵，防止重复累加
        self.init_lora()

    def from_std(self, m):
        """从标准的 nn.Linear 模块加载权重"""
        self.weight = m.weight
        self.weight.requires_grad_(False)
        if m.bias is None:
            if self.bias is not None:
                self.register_parameter("bias", None)
        else:
            self.bias = m.bias
            self.bias.requires_grad_(self.update_bias)
        self.to(device=m.weight.device, dtype=m.weight.dtype, non_blocking=True)

    def to_std(self):
        """
        转换为标准的 nn.Linear 模块。
        先合并 LoRA 权重，然后返回一个新的标准 Linear 层。
        """
        out_features, in_features = self.weight.size()
        rs = nn.Linear(in_features, out_features, bias=self.bias is not None)
        
        # 合并 LoRA 权重
        self.acc_lora()
        
        rs.weight = self.weight
        rs.weight.requires_grad_(True)
        if self.bias is not None:
            rs.bias = self.bias
            rs.bias.requires_grad_(True)
        rs.to(device=self.weight.device, dtype=self.weight.dtype, non_blocking=True)
        return rs

    def extra_repr(self):
        """自定义打印信息，显示 LoRA 相关参数"""
        return "in_features={}, lora_features={}, out_features={}, bias={}".format(self.in_features, self.lora_features, self.out_features, self.bias is not None)


class Embedding(nn.Embedding):
    """
    支持 LoRA 的嵌入层。
    原理与 Linear 类似，但在查找表操作后应用 LoRA 增量。
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, lora_features=None, lora_alpha=None, scaling=1.0, **kwargs):
        # 初始化标准 Embedding 层
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight)
        
        # 冻结原始嵌入权重
        self.weight.requires_grad_(False)
        
        # 设置 LoRA 参数
        self.lora_features, self.scaling = lora_features, (scaling if lora_alpha is None else float(lora_alpha) / float(lora_features))
        
        # 初始化 LoRA 矩阵 A 和 B
        # lora_wa: (num_embeddings, lora_features)
        self.lora_wa = nn.Parameter(torch.empty(num_embeddings, lora_features, dtype=self.weight.dtype, device=self.weight.device))
        # lora_wb: (lora_features, embedding_dim)
        self.lora_wb = nn.Parameter(torch.zeros(lora_features, embedding_dim, dtype=self.weight.dtype, device=self.weight.device))
        
        self.fix_init = self.reset_parameters
        if _weight is None:
            self.reset_parameters()

    def forward(self, x):
        # 1. 获取原始嵌入向量
        out = nnFunc.embedding(x, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        
        # 2. 计算 LoRA 增量
        # 对输入索引查询 lora_wa，得到中间结果，再乘以 lora_wb
        lora_delta = nnFunc.embedding(x.view(-1), self.lora_wa, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).mm(self.lora_wb)
        
        # 3. 将增量加到原始输出上
        out.add_(lora_delta.view(out.size()), alpha=self.scaling)
        return out

    def reset_parameters(self):
        """重置原始嵌入权重"""
        with torch_no_grad():
            _ = 1.0 / sqrt(self.weight.size(-1))
            self.weight.uniform_(-_, _)
            # 如果有 padding_idx，将其对应的向量置零
            if self.padding_idx is not None:
                self.weight[self.padding_idx].zero_()
        self.init_lora()

    def init_lora(self):
        """初始化 LoRA 矩阵"""
        with torch_no_grad():
            _ = 1.0 / sqrt(self.weight.size(-1))
            if hasattr(self, "lora_wa"):
                self.lora_wa.uniform_(-_, _)
            if hasattr(self, "lora_wb"):
                self.lora_wb.zero_()

    def acc_lora(self):
        """将 LoRA 增量合并回原始嵌入权重"""
        with torch_no_grad():
            # 注意：这里不需要转置，因为 Embedding 的维度逻辑与 Linear 略有不同
            self.weight.add_(self.lora_wa.mm(self.lora_wb))
        self.init_lora()

    def from_std(self, m):
        """从标准 Embedding 加载权重"""
        self.weight = m.weight
        self.weight.requires_grad_(False)
        self.to(device=m.weight.device, dtype=m.weight.dtype, non_blocking=True)

    def to_std(self):
        """转换为标准 Embedding 模块"""
        self.acc_lora()
        num_embeddings, embedding_dim = self.weight.size()
        rs = nn.Embedding(num_embeddings, embedding_dim, padding_idx=self.padding_idx, max_norm=self.max_norm, norm_type=self.norm_type, scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse, _weight=self.weight)
        rs.weight.requires_grad_(True)
        rs.to(device=self.weight.device, dtype=self.weight.dtype, non_blocking=True)
        return rs

    def extra_repr(self):
        """自定义打印信息"""
        s = "{num_embeddings}, {embedding_dim}, {lora_features}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2.0:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)