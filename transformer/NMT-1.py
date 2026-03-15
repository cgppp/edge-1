#encoding: utf-8

import torch
from torch import nn

# 导入编码器模块
from transformer.Encoder import Encoder

# 可以选择标准解码器或平均解码器（通过切换注释）
# 使用 transformer.TA.Decoder 可实现透明解码器（Transparent Decoder），具体行为需查看对应模块
from transformer.Decoder import Decoder          # 标准解码器（常规 Transformer 解码层）
# from transformer.AvgDecoder import Decoder     # 平均解码器（可能对上下文进行平均处理，需查阅具体实现）

from utils.base import select_zero_
from utils.fmt.parser import parse_double_value_tuple
from utils.relpos.base import share_rel_pos_cache
from utils.torch.comp import all_done

from cnfg.ihyp import *                           # 内部超参数，如 cache_len_default 等
from cnfg.vocab.base import eos_id, pad_id        # 特殊符号 ID

class NMT(nn.Module):
    """
    神经机器翻译（NMT）模型，由编码器（Encoder）和解码器（Decoder）组成。
    支持共享词嵌入、束搜索解码、训练时解码等功能。
    """

    def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0,
                 attn_drop=0.0, act_drop=None, global_emb=False, num_head=8,
                 xseql=cache_len_default, ahsize=None, norm_output=True,
                 bindDecoderEmb=True, forbidden_index=None, **kwargs):
        """
        初始化 NMT 模型。

        参数：
            isize (int): 词嵌入维度。
            snwd (int): 源语言词汇表大小（Encoder 输入）。
            tnwd (int): 目标语言词汇表大小（Decoder 输出）。
            num_layer (int or tuple): 如果为 int，表示编码器和解码器层数相同；
                                      如果为 tuple，应包含 (enc_layers, dec_layers)。
            fhsize (int, optional): 前馈网络隐藏层维度，默认与 isize 有关（具体见 Encoder/Decoder 实现）。
            dropout (float): Dropout 概率。
            attn_drop (float): 注意力模块内的 Dropout 概率。
            act_drop (float, optional): 激活后的 Dropout 概率。
            global_emb (bool): 是否共享编码器和解码器的词嵌入（源和目标词汇表需相同）。
            num_head (int): 多头注意力头数。
            xseql (int): 最大序列长度（用于位置编码缓存）。
            ahsize (int, optional): 注意力隐藏层维度，默认与 isize 相同。
            norm_output (bool): 是否对输出进行层归一化。
            bindDecoderEmb (bool): 解码器是否绑定输入嵌入和输出分类器的权重（通常为 True）。
            forbidden_index (list, optional): 禁止生成的 token ID 列表（解码时会屏蔽）。
            **kwargs: 其他参数（传递给子模块）。
        """
        super(NMT, self).__init__()

        # 解析 num_layer 为 (enc_layers, dec_layers)
        enc_layer, dec_layer = parse_double_value_tuple(num_layer)

        # 初始化编码器
        self.enc = Encoder(
            isize, snwd, enc_layer,
            fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop,
            num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output
        )

        # 如果 global_emb 为 True，则编码器和解码器共享词嵌入矩阵
        emb_w = self.enc.wemb.weight if global_emb else None

        # 初始化解码器
        self.dec = Decoder(
            isize, tnwd, dec_layer,
            fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop,
            emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize,
            norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=forbidden_index
        )
        # 注：如果使用 AvgDecoder，可能参数有所不同，需根据具体实现调整

        # 如果启用了相对位置编码，则共享编码器和解码器的位置缓存（提升效率）
        if rel_pos_enabled:   # rel_pos_enabled 从 cnfg.ihyp 导入
            share_rel_pos_cache(self)

    def forward(self, inpute, inputo, mask=None, **kwargs):
        """
        前向传播（训练时使用）。

        参数：
            inpute (Tensor): 源语言序列，形状 (batch_size, src_len)。
            inputo (Tensor): 目标语言序列（解码器输入），形状 (batch_size, tgt_len)。
            mask (Tensor, optional): 源序列的 padding mask，形状 (batch_size, 1, src_len)。
                                     若未提供，自动根据 pad_id 生成。
        返回：
            Tensor: 解码器输出 logits，形状 (batch_size, tgt_len, tgt_vocab_size)。
        """
        # 生成 padding mask（如果未提供）
        _mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask

        # 编码器前向 -> 解码器前向
        return self.dec(self.enc(inpute, _mask), inputo, _mask)

    def decode(self, inpute, beam_size=1, max_len=None, length_penalty=0.0, **kwargs):
        """
        推理时解码（束搜索或贪心搜索）。

        参数：
            inpute (Tensor): 源语言序列，形状 (batch_size, src_len)。
            beam_size (int): 束宽，>1 为束搜索，=1 为贪心搜索。
            max_len (int, optional): 最大生成长度，默认根据源长度动态计算。
            length_penalty (float): 长度惩罚系数（通常 <0 鼓励长句，>0 鼓励短句）。
        返回：
            Tensor: 生成的目标序列，形状 (batch_size, gen_len)。
        """
        mask = inpute.eq(pad_id).unsqueeze(1)

        # 计算最大生成长度（若未指定）
        _max_len = (inpute.size(1) + max(64, inpute.size(1) // 4)) if max_len is None else max_len

        # 解码器推理（调用 Decoder 的 decode 方法）
        return self.dec.decode(self.enc(inpute, mask), mask, beam_size, _max_len, length_penalty)

    def load_base(self, base_nmt):
        """
        从已有的 NMT 模型加载参数（用于继续训练或微调）。
        若子模块有 load_base 方法则调用，否则直接赋值。
        """
        if hasattr(self.enc, "load_base"):
            self.enc.load_base(base_nmt.enc)
        else:
            self.enc = base_nmt.enc

        if hasattr(self.dec, "load_base"):
            self.dec.load_base(base_nmt.dec)
        else:
            self.dec = base_nmt.dec

    def update_vocab(self, src_indices=None, tgt_indices=None):
        """
        更新词汇表（例如添加新词后调整嵌入层大小）。
        参数：
            src_indices (Tensor, optional): 源语言新词索引映射（旧索引 -> 新索引）。
            tgt_indices (Tensor, optional): 目标语言新词索引映射。
        注：此方法会调用子模块的 update_vocab，可能涉及嵌入层和分类器的扩展。
        """
        _share_emb, _sembw = False, None
        _update_src, _update_tgt = src_indices is not None, tgt_indices is not None

        # 检查是否共享嵌入（编码器和解码器嵌入相同）
        if _update_src and _update_tgt and src_indices.equal(tgt_indices) \
                and hasattr(self.enc, "get_embedding_weight") \
                and hasattr(self.dec, "get_embedding_weight"):
            _share_emb = self.enc.get_embedding_weight().is_set_to(self.dec.get_embedding_weight())

        # 更新源词汇表
        if _update_src and hasattr(self.enc, "update_vocab"):
            _ = self.enc.update_vocab(src_indices)
            if _share_emb:
                _sembw = _   # 如果共享嵌入，保留更新后的嵌入权重用于解码器

        # 更新目标词汇表
        if _update_tgt and hasattr(self.dec, "update_vocab"):
            self.dec.update_vocab(tgt_indices, wemb_weight=_sembw)

    def update_classifier(self, *args, **kwargs):
        """
        更新解码器的分类器（例如在微调时调整输出层）。
        直接转发给解码器的同名方法。
        """
        if hasattr(self.dec, "update_classifier"):
            self.dec.update_classifier(*args, **kwargs)

    def train_decode(self, inpute, beam_size=1, max_len=None, length_penalty=0.0, mask=None):
        """
        训练时使用的解码函数（用于生成样本或评估）。
        根据 beam_size 选择调用束搜索或贪心搜索。
        """
        _mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask

        _max_len = (inpute.size(1) + max(64, inpute.size(1) // 4)) if max_len is None else max_len

        if beam_size > 1:
            return self.train_beam_decode(inpute, _mask, beam_size, _max_len, length_penalty)
        else:
            return self.train_greedy_decode(inpute, _mask, _max_len)

    def train_greedy_decode(self, inpute, mask=None, max_len=512):
        """
        训练时使用的贪心解码（逐词生成，用于快速验证）。
        """
        ence = self.enc(inpute, mask)          # 编码器输出
        bsize = inpute.size(0)

        # 初始化解码器输入为 <sos>（这里用 1 代表起始符，实际应根据词汇表调整）
        out = inpute.new_ones(bsize, 1)         # 形状 (batch_size, 1)

        done_trans = None

        for i in range(0, max_len):
            _out = self.dec(ence, out, mask)    # 解码器前向
            _out = _out.argmax(dim=-1)          # 取最大概率 token

            wds = _out.narrow(1, _out.size(1) - 1, 1)  # 获取最后一个时间步的输出
            out = torch.cat((out, wds), -1)      # 拼接到输入序列

            # 更新完成标志（遇到 eos_id 的样本标记为完成）
            done_trans = wds.squeeze(1).eq(eos_id) if done_trans is None \
                        else (done_trans | wds.squeeze(1).eq(eos_id))

            if all_done(done_trans, bsize):      # 所有样本都生成了 eos
                break

        # 返回时去掉起始符（第一个 token）
        return out.narrow(1, 1, out.size(1) - 1)

    def train_beam_decode(self, inpute, mask=None, beam_size=8, max_len=512,
                          length_penalty=0.0, return_all=False, clip_beam=clip_beam_with_lp):
        """
        训练时使用的束搜索解码。

        参数：
            clip_beam (bool): 是否在每一步应用长度惩罚进行剪枝（提高效率）。
            return_all (bool): 若 True，返回所有束的完整序列及分数；否则返回最佳束。
        注：clip_beam_with_lp 从 cnfg.ihyp 导入。
        """
        bsize, seql = inpute.size()
        real_bsize = bsize * beam_size

        # 复制编码器输出以匹配束数
        ence = self.enc(inpute, mask).repeat(1, beam_size, 1).view(real_bsize, seql, -1)
        mask = mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)

        # 初始化解码器输入（起始符）
        out = inpute.new_ones(real_bsize, 1)

        # 初始化长度惩罚向量（若使用）
        if length_penalty > 0.0:
            lpv = ence.new_ones(real_bsize, 1)          # 长度惩罚系数
            lpv_base = 6.0 ** length_penalty            # 基础值，用于后续归一化

        done_trans = None
        scores = None
        sum_scores = None

        beam_size2 = beam_size * beam_size
        bsizeb2 = bsize * beam_size2

        for step in range(1, max_len + 1):
            _out = self.dec(ence, out, mask)

            # 取最后一个时间步的输出，并重塑为 (bsize, beam_size, vocab_size)
            _out = _out.narrow(1, _out.size(1) - 1, 1).view(bsize, beam_size, -1)

            # 取 top-k（beam_size）个候选
            _scores, _wds = _out.topk(beam_size, dim=-1)   # 形状 (bsize, beam_size, beam_size)

            # 处理已经完成翻译的束（已生成 eos 的束后续分数应保持不变）
            if done_trans is not None:
                _done_trans_unsqueeze = done_trans.unsqueeze(2)
                # 已完成束的分数设为当前累积分数，其他设为 -inf 以保证不会被选中
                _scores = _scores.masked_fill(_done_trans_unsqueeze.expand(bsize, beam_size, beam_size), 0.0) \
                          + sum_scores.unsqueeze(2).repeat(1, 1, beam_size).masked_fill_(
                              select_zero_(_done_trans_unsqueeze.repeat(1, 1, beam_size), -1, 0),
                              -inf_default
                          )

                if length_penalty > 0.0:
                    # 更新长度惩罚系数（已完成束的惩罚不变）
                    lpv = lpv.masked_fill(1 - done_trans.view(real_bsize, 1),
                                           ((step + 5.0) ** length_penalty) / lpv_base)

            # 根据是否 clip_beam 决定如何选择 top-k 束
            if clip_beam and (length_penalty > 0.0):
                # 在每一步应用长度惩罚后选择
                scores, _inds = (_scores / lpv.expand(real_bsize, beam_size)).view(bsize, beam_size2).topk(beam_size, dim=-1)
                _tinds = (_inds + torch.arange(0, bsizeb2, beam_size2, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)

                # 更新累积分数（未惩罚的原始分数）
                sum_scores = _scores.view(bsizeb2).index_select(0, _tinds).view(bsize, beam_size)
            else:
                # 直接根据原始分数选择
                scores, _inds = _scores.view(bsize, beam_size2).topk(beam_size, dim=-1)
                _tinds = (_inds + torch.arange(0, bsizeb2, beam_size2, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
                sum_scores = scores

            # 获取对应的 token ID
            wds = _wds.view(bsizeb2).index_select(0, _tinds).view(real_bsize, 1)

            # 计算每个新束对应的父束索引
            _inds = (_inds // beam_size + torch.arange(0, real_bsize, beam_size, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)

            # 更新输出序列（保留历史，拼接新 token）
            out = torch.cat((out.index_select(0, _inds), wds), -1)

            # 更新完成标志
            done_trans = wds.view(bsize, beam_size).eq(eos_id) if done_trans is None \
                         else (done_trans.view(real_bsize).index_select(0, _inds) | wds.view(real_bsize).eq(eos_id)).view(bsize, beam_size)

            # 检查是否可以提前停止（所有束的最佳束都已生成 eos）
            _done = False
            if length_penalty > 0.0:
                lpv = lpv.index_select(0, _inds)
            elif (not return_all) and all_done(done_trans.select(1, 0), bsize):
                _done = True

            if _done or all_done(done_trans, real_bsize):
                break

        # 移除起始符
        out = out.narrow(1, 1, out.size(1) - 1)

        # 如果未在每步应用长度惩罚，则在最后应用
        if (not clip_beam) and (length_penalty > 0.0):
            scores = scores / lpv.view(bsize, beam_size)

        if return_all:
            # 返回所有束的序列和分数
            return out.view(bsize, beam_size, -1), scores
        else:
            # 返回最佳束（第一个束）
            return out.view(bsize, beam_size, -1).select(1, 0)