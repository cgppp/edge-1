#encoding: utf-8

from torch import nn
from torch.nn import functional as nnFunc

from modules.norm.cust import RMSNorm as custRMSNorm, rms_norm as cust_rms_norm

LayerNorm = nn.LayerNorm
RMSNorm = nn.RMSNorm if hasattr(nn, "RMSNorm") else custRMSNorm

normer_cls = (LayerNorm, RMSNorm,)

layer_norm = nnFunc.layer_norm
rms_norm = nnFunc.rms_norm if hasattr(nnFunc, "rms_norm") else cust_rms_norm
