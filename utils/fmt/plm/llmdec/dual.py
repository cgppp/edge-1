#encoding: utf-8
"""
双文件 LLM Decoder 数据加载：从「指令 .ids」与「回答 .ids」两个文件按行对齐读取，拼成「指令+回答」序列，
并记录每条样本的 [指令长度, 总长度]，供后续写成 HDF5（src 组 + tgt 组）或直接做 batch 训练。

与 tools/plm/mkiodata_llmdec.py 的写入格式一致：每条样本 = 指令 token 序列 + 回答 token 序列，
tgt 侧 HDF5 直接存 **[lid, lgth]** 两数，语义为拼接序列上半开区间 **[lid, lgth)**（0-based）。
"""

from math import ceil

from utils.fmt.base import get_bsize, iter_to_int, list_reader as file_reader, pad_batch

from cnfg.vocab.base import pad_id


def batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=file_reader, iter_to_int=iter_to_int, **kwargs):
	"""
	从「指令 .ids」与「回答 .ids」双文件按行对齐读入，将每对 (指令行, 回答行) 拼成一条「指令+回答」序列，
	按 maxpad/maxpart/maxtoken/bsize 等约束组成 batch，逐个 yield。

	参数:
		finput: 指令侧文件路径或可迭代对象，每行为一串空格分隔的 token id。
		ftarget: 回答侧文件路径或可迭代对象，与 finput 行一一对应。
		bsize (int): 期望的 batch 内样本数（会结合长度由 get_bsize 调整）。
		maxpad (int): 允许的 padding 上限。
		maxpart (float): 用于计算 maxlen 的分母，maxlen 与 lgth/maxpart 相关。
		maxtoken (int): 每 batch 的 token 数上限，供 get_bsize 计算实际 bsize。
		minbsize (int): batch 内至少样本数，未满时可继续累积。
		get_bsize: 函数 (maxlen, maxtoken, bsize) -> 实际使用的 batch 大小。
		file_reader: 读文件为行的函数，返回可迭代的「每行 token 字符串」。
		iter_to_int: 将一行字符串转为 token id 列表的函数。
		**kwargs: 保留，未使用。

	Yields:
		rsi (list of list of int): 当前 batch 的样本列表，每条样本为「指令 token 列表 + 回答 token 列表」拼接而成。
		rsl (list of [lid, lgth]): 与 rsi 一一对应，lid=指令长度，lgth=指令+回答总长度；用于标识回答区间 [lid, lgth)。
		mlen_i (int): 当前 batch 内最长序列长度（用于后续 pad 到同一长度）。
	"""
	_f_maxpart = float(maxpart)
	rsi = []   # 当前 batch 的样本：每条为 instruction + answer 的 token id 列表
	rsl = []   # 当前 batch 的 [lid, lgth] 列表，与 rsi 一一对应
	nd = 0         # 当前 batch 已累积的样本数
	maxlen = 0     # 当前 batch 允许的最大序列长度（用于判断是否还能塞进当前样本）
	mlen_i = 0     # 当前 batch 内已出现的最长序列长度

	for i_d, td in zip(file_reader(finput, keep_empty_line=True), file_reader(ftarget, keep_empty_line=True)):
		# 将一行字符串解析为 token id 列表
		i_d, td = list(iter_to_int(i_d)), list(iter_to_int(td))
		lid = len(i_d)    # 指令长度（回答区间起点，0-based）
		ltd = len(td)     # 回答长度
		lgth = lid + ltd  # 拼接后总长度（回答区间终点，0-based 下回答为 [lid, lgth)）

		# 第一个样本或刚 yield 完：根据当前样本长度初始化 maxlen 与 _bsize
		if maxlen == 0:
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(maxlen + _maxpad, maxtoken, bsize)

		# 当前 batch 还能容纳本条样本：未满 minbsize，或长度未超且未满 _bsize
		if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
			rsi.append(i_d + td)       # 拼接：指令 + 回答
			rsl.append([lid, lgth])    # 记录区间，供 tgt 原样写 [lid, lgth]
			if lgth > mlen_i:
				mlen_i = lgth
			nd += 1
		else:
			# 当前 batch 已满或本条超长：先 yield 已有 batch，再以本条为起点开新 batch
			yield rsi, rsl, mlen_i
			rsi = [i_d + td]
			rsl = [[lid, lgth]]
			mlen_i = lgth
			_maxpad = min(maxpad, ceil(lgth / _f_maxpart))
			maxlen = lgth + _maxpad
			_bsize = get_bsize(maxlen + _maxpad, maxtoken, bsize)
			nd = 1
	# 最后不足一 batch 的样本也要 yield
	if rsi:
		yield rsi, rsl, mlen_i


def batch_padder(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, pad_batch=pad_batch, batch_loader=batch_loader, pad_id=pad_id, **kwargs):
	"""
	在 batch_loader 基础上，将每个 batch 内的序列按 mlen_i 对齐 padding，输出「已 pad 的 token 矩阵」与「区间列表」。

	参数:
		finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize: 同 batch_loader。
		pad_batch: 函数 (list of list of int, mlen_i, pad_id=...) ->  padded tensor 或矩阵，将长短不一的序列 pad 到 mlen_i。
		batch_loader: 上面定义的 batch_loader，用于产生 (rsi, rsl, mlen_i)。
		pad_id (int): padding 使用的 token id（如 Qwen 的 <|endoftext|>）。
		**kwargs: 传给 batch_loader。

	Yields:
		padded_batch: 当前 batch 经 pad_batch 对齐后的结果（形状通常为 (batch_size, mlen_i)）。
		ll (list of [lid, lgth]): 与 batch_loader 的 rsl 相同，每条样本的 [指令长度, 总长度]。
	"""
	for i_d, ll, mlen_i in batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		yield pad_batch(i_d, mlen_i, pad_id=pad_id), ll
