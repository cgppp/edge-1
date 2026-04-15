#encoding: utf-8
"""
LoRA 工具层：将标准 nn.Linear / nn.Embedding 替换为带 LoRA 的版本，或反向还原。

主要接口：
- std2lora：整棵模块树中「标准层 → LoRA 层」的批量替换，并可选保留/恢复权重共享（parameter tying）。
- lora2std：将 LoRA 层还原为标准层（合并 LoRA 权重到主权重后替换）。
"""

from torch import nn

from modules.lora.base import Embedding, Linear
from utils.base import add_parameter, get_parameter_tying
from utils.func import always_true as name_cfunc_full
from utils.module.patcher import patch_std, to_std


# ------------------------------------------------------------------------------
# 单层替换：标准层 → LoRA 层（供 patch_std 按类型调用）
# ------------------------------------------------------------------------------

def lora_get_linear(modin, lora_features=None, lora_alpha=None, scaling=1.0, update_bias=True, **kwargs):
	"""
	将单个 nn.Linear 换成 LoRA 版 Linear（或已是则直接返回）。

	参数:
		modin: 待替换的 nn.Linear 模块。
		lora_features: LoRA 秩（中间维度），即 lora_wa 的列数 / lora_wb 的行数。
		lora_alpha: LoRA 缩放中的分子，实际 scaling = lora_alpha / lora_features（若未传则用 scaling）。
		scaling: 直接指定的 LoRA 输出缩放系数；若 lora_alpha 已传则会被覆盖。
		update_bias: 是否在微调时更新 bias（True=更新，False=冻结）。
		**kwargs: 其它未使用参数，便于与 patch_std 传参兼容。

	返回:
		替换后的 modules.lora.base.Linear 实例（权重通过 from_std 从 modin 拷贝，主权重冻结）。
	"""
	if isinstance(modin, Linear):
		return modin

	out_features, in_features = modin.weight.size()
	rsm = Linear(in_features, out_features, bias=modin.bias is not None, lora_features=lora_features, lora_alpha=lora_alpha, scaling=scaling, update_bias=update_bias)
	rsm.from_std(modin)

	return rsm


def lora_get_embedding(modin, lora_features=None, lora_alpha=None, scaling=1.0, **kwargs):
	"""
	将单个 nn.Embedding 换成 LoRA 版 Embedding（或已是则直接返回）。

	参数:
		modin: 待替换的 nn.Embedding 模块。
		lora_features: LoRA 秩。
		lora_alpha: 同 Linear，用于计算 scaling。
		scaling: 直接指定的缩放；有 lora_alpha 时会被覆盖。
		**kwargs: 其它未使用参数。

	返回:
		替换后的 modules.lora.base.Embedding 实例。
	"""
	if isinstance(modin, Embedding):
		return modin

	num_embeddings, embedding_dim = modin.weight.size()
	rsm = Embedding(num_embeddings, embedding_dim, padding_idx=modin.padding_idx, max_norm=modin.max_norm, norm_type=modin.norm_type, scale_grad_by_freq=modin.scale_grad_by_freq, sparse=modin.sparse, _weight=modin.weight, lora_features=lora_features, lora_alpha=lora_alpha, scaling=scaling)
	rsm.from_std(modin)

	return rsm


# 类型 → 替换函数的映射，供 patch_std 遍历子模块时按类型调用
type_func = {nn.Linear: lora_get_linear, nn.Embedding: lora_get_embedding}
# 需要被 lora2std 识别并还原为标准层的类型
tgt_types = (Linear, Embedding,)


# ------------------------------------------------------------------------------
# 整棵树的替换：标准 → LoRA（std2lora）、LoRA → 标准（lora2std）
# ------------------------------------------------------------------------------

def std2lora(modin, lora_features=None, lora_alpha=None, scaling=1.0, update_bias=True, name_cfunc=name_cfunc_full, keep_lora_weight_tying=True, type_func=type_func, **kwargs):
	"""
	将模块树中所有 nn.Linear / nn.Embedding 替换为 LoRA 版本，并可选保持原有「权重共享」关系。

	参数:
		modin: 根模块（如整个 Decoder）；其下所有匹配类型的子模块会被替换。
		lora_features: LoRA 秩。
		lora_alpha: 用于 scaling = lora_alpha / lora_features。
		scaling: 直接指定的 LoRA 缩放；有 lora_alpha 时以 lora_alpha 为准。
		update_bias: 是否更新 Linear 的 bias。
		name_cfunc: 子模块名过滤函数，签名为 (name: str) -> bool；仅对 name_cfunc(name) 为 True 的
		            子模块做替换。默认 always_true 表示全部替换。
		keep_lora_weight_tying: 若为 True，替换前会记录原模块的 parameter tying（多参数名指向同一
		                        tensor），替换后在新 LoRA 模块上恢复：主权重与 lora_wa/lora_wb 的共享
		                        关系会按原命名规则重新绑定。
		type_func: 类型到替换函数的映射，默认 {nn.Linear: lora_get_linear, nn.Embedding: lora_get_embedding}。
		**kwargs: 会传给 type_func 中的替换函数。

	返回:
		(rsm, md): rsm 为替换后的根模块（即 modin 原地替换子模块后的引用）；md 为被替换掉的
		           子模块的 {name: 原模块} 字典，可用于后续 restore 等。
	"""
	# 若需要保持权重共享：先记录「哪些参数名共享同一 tensor」
	_tl = get_parameter_tying(modin) if keep_lora_weight_tying else []
	# 递归遍历 modin，对 nn.Linear / nn.Embedding 调用 type_func 中对应函数，替换为 LoRA 层
	rsm, md = patch_std(modin, lora_features=lora_features, lora_alpha=lora_alpha, scaling=scaling, update_bias=update_bias, type_pfunc=type_func, name_cfunc=name_cfunc, **kwargs)

	if _tl:
		# 替换后，新模块的 named_parameters(remove_duplicate=False) 可得到 (name, param)
		_mpd = dict(rsm.named_parameters(remove_duplicate=False))
		for _nl in _tl:
			# _nl：一组共享同一 tensor 的参数名，如 ["layer.0.self_attn.in_proj_weight", "layer.1.self_attn.in_proj_weight"]
			_el = [_ for _ in _nl if _ in _mpd]
			if len(_el) > 1:
				# 以第一个参数为基准，把其余同名位置的参数都「绑」到同一 tensor（恢复主权重共享）
				_p = _mpd[_el[0]]
				for _n in _el[1:]:
					if not _p.is_set_to(_mpd[_n]):
						add_parameter(rsm, _n, _p)
				# 对 LoRA 参数做同样的事：原参数名多为 "xxx.weight"，对应 lora 为 "xxx_lora_wa" / "xxx_lora_wb"
				# _el 中名去掉最后 6 个字符 ".weight" 得到前缀，再拼 "lora_wa" / "lora_wb"
				_el = [_ for _ in _nl if (("%slora_wa" % _[:-6]) in _mpd) and (("%slora_wb" % _[:-6]) in _mpd)]
				if len(_el) > 1:
					_pa, _pb = _mpd["%slora_wa" % _el[0][:-6]], _mpd["%slora_wb" % _el[0][:-6]]
					for _n in _el[1:]:
						_na, _nb = "%slora_wa" % _n[:-6], "%slora_wb" % _n[:-6]
						if not _pa.is_set_to(_mpd[_na]):
							add_parameter(rsm, _na, _pa)
						if not _pb.is_set_to(_mpd[_nb]):
							add_parameter(rsm, _nb, _pb)

	return rsm, md


def lora2std(modin, types=tgt_types):
	"""
	将模块树中所有 LoRA 层（Linear / Embedding）还原为标准 nn.Linear / nn.Embedding。
	还原时会把 LoRA 权重合并进主权重（各层的 to_std() 里做 acc_lora），然后替换为标准层。

	参数:
		modin: 根模块（通常为已 std2lora 过的模型）。
		types: 需要被还原的模块类型元组，默认 (Linear, Embedding)。

	返回:
		(modin, md): modin 为原地替换后的根模块；md 为被替换掉的 LoRA 子模块 {name: 原模块}。
	"""
	return to_std(modin, types=types)
