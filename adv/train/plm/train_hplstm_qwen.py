#encoding: utf-8
"""
Qwen3 基座训练入口（按 `cnfg.lora.hplstm_fusion` 或环境变量 `HPLSTM_FUSION` 加载 `Decoder_1`/`Decoder_2`/`Decoder_3`）。

- **默认（cnfg.lora.lora_features is None）**：冻结全部参数，**仅解冻**各层 `self.hplstm` 子模块，只训练 HPLSTM；checkpoint 用 `rgrad_filter` 只存 `requires_grad=True` 的参数。
- **LoRA（lora_features 非空）**：`std2lora` 后只训 LoRA 等；可用环境变量 LORA_RANK、LORA_ALPHA。

数据：HDF5 由 `mkiodata_llmdec.py` 生成；`PMaskDataConverter` 只对回答区间算 loss。
环境变量：DATA_ID、PRE_TRAINED_M、LORA_RANK、LORA_ALPHA、HPLSTM_FUSION（A/B/C，对应 Decoder_1/2/3）。
"""

import importlib
import os

# ----- 环境变量覆盖配置（便于 task.sh 传参，无需改 cnfg 文件） -----
_data_id = os.environ.get("DATA_ID", "").strip()
if _data_id:
	import cnfg.base as _base_cnfg
	_base_cnfg.data_id = _data_id
	_base_cnfg.train_data = _base_cnfg.cache_dir + _data_id + "/train.h5"
	_base_cnfg.dev_data = _base_cnfg.cache_dir + _data_id + "/dev.h5"
_pre_trained_m = os.environ.get("PRE_TRAINED_M", "").strip()
if _pre_trained_m:
	import cnfg.plm.qwen.v3.base as _qwen_cnfg
	_qwen_cnfg.pre_trained_m = _pre_trained_m
_lora_rank = os.environ.get("LORA_RANK", "").strip()
_lora_alpha = os.environ.get("LORA_ALPHA", "").strip()
if _lora_rank or _lora_alpha:
	import cnfg.lora as _lora_cnfg
	if _lora_rank:
		try:
			_lora_cnfg.lora_features = int(_lora_rank)
		except ValueError:
			pass
	if _lora_alpha:
		try:
			_lora_cnfg.lora_alpha = int(_lora_alpha)
		except ValueError:
			pass
_use_amp_env = os.environ.get("USE_AMP", "").strip().lower()
if _use_amp_env in ("1", "true", "yes"):
	import cnfg.base as _base_cnfg
	_base_cnfg.use_amp = True

_hplstm_fusion_env = os.environ.get("HPLSTM_FUSION", "").strip().upper()
if _hplstm_fusion_env in ("A", "B", "C"):
	import cnfg.lora as _lcnfg_fusion
	_lcnfg_fusion.hplstm_fusion = _hplstm_fusion_env

import torch
from random import shuffle
from torch.optim import Adam as Optimizer

from loss.base import LabelSmoothingLoss
from lrsch import CustLR as LRScheduler
from optm.agent import fp32_optm_agent_wrapper as mp_optm_agent_wrapper
from parallel.base import DataParallelCriterion
from parallel.optm import MultiGPUGradScaler
from parallel.parallelMT import DataParallelMT

import cnfg.lora as _lcnfg_nmt
_hplstm_fusion = str(getattr(_lcnfg_nmt, "hplstm_fusion", "B")).strip().upper()
if _hplstm_fusion not in ("A", "B", "C"):
	_hplstm_fusion = "B"
NMT = importlib.import_module("transformer.PLM.QWen.v3.Decoder_%s" % {"A": "1", "B": "2", "C": "3"}[_hplstm_fusion]).Decoder

from utils.base import free_cache, get_logger, mkdir, set_random_seed
from utils.contpara import get_model_parameters
from utils.fmt.base import iter_to_str
from utils.fmt.base4torch import load_emb, parse_cuda
from utils.h5serial import h5File
from utils.init.base import init_model_params
from utils.io import load_model_cpu, save_model, save_states
from utils.lora.base import std2lora
from utils.norm.mp.f import convert as make_mp_model
from utils.plm.inference import get_h5g_common_prefix, prepare_states_bsize
from utils.state.holder import Holder
from utils.state.pyrand import PyRandomState
from utils.state.thrand import THRandomState
from utils.torch.comp import GradScaler, torch_autocast, torch_compile, torch_inference_mode
from utils.tqdm import tqdm
from utils.train.base import freeze_module, getlr, optm_step, optm_step_zero_grad_set_none, reset_Adam, unfreeze_module
from utils.train.dss import dynamic_sample
from utils.train.ft import rgrad_filter, unfreeze_linear_bias, unfreeze_normer, unfreeze_hplstm, unfreeze_reslstm
from utils.train.llm import PMaskDataConverter

import cnfg.lora as lcnfg  # 与上方 _lcnfg_nmt 同一模块；hplstm_fusion 已在 import NMT 前由环境变量可能覆盖
import cnfg.plm.qwen.v3.base as cnfg
from cnfg.plm.qwen.v3.ihyp import *
from cnfg.vocab.plm.qwen.v3 import vocab_size

def train(td, tl, ed, nd, optm, lrsch, model, lossf, mv_device, logger, done_tokens, multi_gpu, multi_gpu_optimizer, tokens_optm=32768, nreport=None, save_every=None, chkpf=None, state_holder=None, statesf=None, num_checkpoint=1, cur_checkid=0, report_eva=True, remain_steps=None, save_loss=False, save_checkp_epoch=False, scaler=None):
	"""
	一轮训练循环：按 tl 中的 batch key 遍历 HDF5，取 src/tgt，经 PMaskDataConverter 得到只对回答区间的 (oi, pred_mask, ot)，前向+loss+反向，累积到 tokens_optm 后 step 优化器。

	参数: td=训练 HDF5 句柄, tl=本轮 batch key 列表("0","1",...), ed=验证 HDF5, nd=验证 batch 数,
	      optm=优化器, lrsch=学习率调度, model=Qwen Decoder(+LoRA), lossf=LabelSmoothingLoss,
	      mv_device=主设备, logger=日志, done_tokens=已累积 token 数(用于梯度累积),
	      multi_gpu/multi_gpu_optimizer=多卡与优化器模式, tokens_optm=每多少 token 做一次 step,
	      nreport=每多少 batch 打日志/做验证, save_every/chkpf/state_holder/statesf=checkpoint 与状态保存,
	      num_checkpoint/cur_checkid=轮转 checkpoint 数与当前 id, report_eva=是否在 report 时跑验证,
	      remain_steps=剩余训练步数(可为 None), save_loss=是否记录每 batch loss, save_checkp_epoch=是否按 epoch 存 checkpoint, scaler=AMP 梯度缩放.
	返回: (平均 loss, 本轮结束时的 done_tokens, cur_checkid, 更新后的 remain_steps, 每 batch loss 字典或 None).
	"""
	sum_loss = part_loss = 0.0
	sum_wd = part_wd = 0
	_done_tokens, _cur_checkid, _cur_rstep, _use_amp = done_tokens, cur_checkid, remain_steps, scaler is not None
	global minerr, minloss, wkdir, save_auto_clean, namin, save_model_ps_func, data_converter
	model.train()
	cur_b, _ls = 1, {} if save_loss else None   # cur_b=当前 batch 序号, _ls=每 batch 的 loss（若 save_loss）
	src_grp, tgt_grp = td["src"], td["tgt"]
	for i_d in tqdm(tl, mininterval=tqdm_mininterval):
		seq_batch = torch.from_numpy(src_grp[i_d][()])   # (batch, seq_len) 整段 token
		seq_o = torch.from_numpy(tgt_grp[i_d][()])       # (batch, 2) 每行 [lid, lgth]，与 dual / PMaskDataConverter 一致
		lo = seq_o.size(1) - 1
		if mv_device:
			seq_batch = seq_batch.to(mv_device, non_blocking=True)
		seq_batch = seq_batch.to(torch.int64, non_blocking=True)
		oi, pred_mask, ot = data_converter(seq_batch, seq_o)   # 只对回答区间算 loss 的输入/ mask/ 目标

		with torch_autocast(enabled=_use_amp):
			output = model(oi, word_prediction=True, pred_mask=pred_mask)
			loss = lossf(output, ot)
			if multi_gpu:
				loss = loss.sum()
		loss_add = loss.data.item()

		if scaler is None:
			loss.backward()
		else:
			scaler.scale(loss).backward()

		wd_add = ot.numel()   # 本 batch 参与 loss 的 token 数
		loss = output = oi = ot = seq_batch = seq_o = pred_mask = None
		sum_loss += loss_add
		if save_loss:
			_ls[i_d] = loss_add / wd_add
		sum_wd += wd_add
		_done_tokens += wd_add

		# 梯度累积到 tokens_optm 后执行一次优化器 step
		if _done_tokens >= tokens_optm:
			optm_step(optm, model=model, scaler=scaler, multi_gpu=multi_gpu, multi_gpu_optimizer=multi_gpu_optimizer, zero_grad_none=optm_step_zero_grad_set_none)
			_done_tokens = 0
			if _cur_rstep is not None:
				if save_checkp_epoch and (save_every is not None) and (_cur_rstep % save_every == 0) and (chkpf is not None) and (_cur_rstep > 0):
					if num_checkpoint > 1:
						_fend = "_%d.h5" % (_cur_checkid)
						_chkpf = chkpf[:-3] + _fend
						_cur_checkid = (_cur_checkid + 1) % num_checkpoint
					else:
						_chkpf = chkpf
					save_model(model, _chkpf, multi_gpu, print_func=logger.info, ps_func=save_model_ps_func)
					if statesf is not None:
						save_states(state_holder.state_dict(update=False, **{"remain_steps": _cur_rstep, "checkpoint_id": _cur_checkid, "training_list": tl[cur_b - 1:]}), statesf, print_func=logger.info)
				_cur_rstep -= 1
				if _cur_rstep <= 0:
					break
			lrsch.step()

		# 每 nreport 个 batch 打日志，可选跑验证并保存最佳
		if nreport is not None:
			part_loss += loss_add
			part_wd += wd_add
			if cur_b % nreport == 0:
				if report_eva:
					_leva, _eeva = eva(ed, nd, model, lossf, mv_device, multi_gpu, _use_amp)
					logger.info("Average loss over %d tokens: %.3f, valid loss/error: %.3f %.2f" % (part_wd, part_loss / part_wd, _leva, _eeva,))
					if (_eeva < minerr) or (_leva < minloss):
						save_model(model, wkdir + "eva_%.3f_%.2f.h5" % (_leva, _eeva,), multi_gpu, print_func=logger.info, mtyp="ieva" if save_auto_clean else None, ps_func=save_model_ps_func)
						if statesf is not None:
							save_states(state_holder.state_dict(update=False, **{"remain_steps": _cur_rstep, "checkpoint_id": _cur_checkid, "training_list": tl[cur_b - 1:]}), statesf, print_func=logger.info)
						logger.info("New best model saved")
						namin = 0
						if _eeva < minerr:
							minerr = _eeva
						if _leva < minloss:
							minloss = _leva
					free_cache(mv_device)
					model.train()
				else:
					logger.info("Average loss over %d tokens: %.3f" % (part_wd, part_loss / part_wd,))
				part_loss = 0.0
				part_wd = 0

		if save_checkp_epoch and (_cur_rstep is None) and (save_every is not None) and (cur_b % save_every == 0) and (chkpf is not None) and (cur_b < ntrain):
			if num_checkpoint > 1:
				_fend = "_%d.h5" % (_cur_checkid)
				_chkpf = chkpf[:-3] + _fend
				_cur_checkid = (_cur_checkid + 1) % num_checkpoint
			else:
				_chkpf = chkpf
			save_model(model, _chkpf, multi_gpu, print_func=logger.info, ps_func=save_model_ps_func)
			if statesf is not None:
				save_states(state_holder.state_dict(update=False, **{"remain_steps": _cur_rstep, "checkpoint_id": _cur_checkid, "training_list": tl[cur_b - 1:]}), statesf, print_func=logger.info)
		cur_b += 1
	if part_wd != 0.0:
		logger.info("Average loss over %d tokens: %.3f" % (part_wd, part_loss / part_wd,))
	return sum_loss / sum_wd, _done_tokens, _cur_checkid, _cur_rstep, _ls

def eva(ed, nd, model, lossf, mv_device, multi_gpu, use_amp=False):
	"""
	在验证集上评估：遍历 nd 个 batch，用 data_converter 得到 (oi, pred_mask, ot)，前向算 loss 与准确 token 数。
	参数: ed=验证 HDF5, nd=batch 数, model/lossf/mv_device/multi_gpu/use_amp 同训练。
	返回: (平均 loss, 错误率百分比 即 (w-r)/w*100)。
	"""
	r = w = 0         # r=预测正确的 token 数, w=参与 loss 的 token 总数
	sum_loss = 0.0
	model.eval()
	src_grp, tgt_grp = ed["src"], ed["tgt"]
	global data_converter, prefix_ids, prefix_len
	with torch_inference_mode():
		# 若有固定 prefix（如指令前缀），先跑一遍得到 prefix_states 供后续 batch 复用
		if prefix_ids:
			seq_batch = torch.as_tensor(prefix_ids, dtype=torch.int32).unsqueeze(0)
			if cuda_device:
				seq_batch = seq_batch.to(cuda_device, non_blocking=True)
			seq_batch = seq_batch.to(torch.int64, non_blocking=True)
			prefix_states = mymodel.build_states(seq_batch, states=None, return_last_hidden=False)
		else:
			prefix_states = None
		for i in tqdm(range(nd), mininterval=tqdm_mininterval):
			bid = str(i)
			seq_batch = torch.from_numpy(src_grp[bid][()])
			seq_o = torch.from_numpy(tgt_grp[bid][()])
			if prefix_len:
				seq_batch = seq_batch.narrow(-1, prefix_len, seq_batch.size(-1) - prefix_len)
			if mv_device:
				seq_batch = seq_batch.to(mv_device, non_blocking=True)
				# seq_o device movement is handled by data_converter in case necessary
				#seq_o = seq_o.to(mv_device, non_blocking=True)
			seq_batch = seq_batch.to(torch.int64, non_blocking=True)
			oi, pred_mask, ot = data_converter(seq_batch, seq_o, seq_o_sub_len=prefix_len)
			with torch_autocast(enabled=use_amp):
				output = model(oi, word_prediction=True, pred_mask=pred_mask, states=prepare_states_bsize(prefix_states, bsize=seq_batch.size(0)))
				loss = lossf(output, ot)
				if multi_gpu:
					loss = loss.sum()
					trans = torch.cat([outu.argmax(-1).to(mv_device, non_blocking=True) for outu in output], 0)
				else:
					trans = output.argmax(-1)
			sum_loss += loss.data.item()
			correct = trans.eq(ot).to(torch.int32, non_blocking=True)
			w += ot.numel()
			r += correct.sum().item()
			correct = pred_mask = trans = loss = output = oi = ot = seq_batch = seq_o = None
	w = float(w)
	return sum_loss / w, (w - r) / w * 100.0

def hook_lr_update(optm, flags=None):
	"""学习率/优化器重置钩子，内部调用 reset_Adam。"""
	reset_Adam(optm, flags)

def init_fixing(module):
	"""对子模块若有 fix_init 则调用（如 LoRA 的 init_lora）。"""
	if hasattr(module, "fix_init"):
		module.fix_init()

def load_fixing(module):
	"""对子模块若有 fix_load 则调用（加载权重后的修正）。"""
	if hasattr(module, "fix_load"):
		module.fix_load()


# ----- 从 cnfg 读取训练控制与路径 -----
rid = cnfg.run_id
earlystop = cnfg.earlystop
maxrun = cnfg.maxrun
tokens_optm = cnfg.tokens_optm
done_tokens = 0
batch_report = cnfg.batch_report
report_eva = cnfg.report_eva
use_ams = cnfg.use_ams
cnt_states = cnfg.train_statesf
save_auto_clean = cnfg.save_auto_clean
overwrite_eva = cnfg.overwrite_eva
save_every = cnfg.save_every
start_chkp_save = cnfg.epoch_start_checkpoint_save
epoch_save = cnfg.epoch_save
remain_steps = cnfg.training_steps

wkdir = "".join((cnfg.exp_dir, cnfg.data_id, "/", cnfg.group_id, "/", rid, "/"))
mkdir(wkdir)
chkpf = None
statesf = None
if save_every is not None:
	chkpf = wkdir + "checkpoint.h5"
if cnfg.save_train_state:
	statesf = wkdir + "train.states.t7"
logger = get_logger(wkdir + "train.log")
_hf = str(getattr(lcnfg, "hplstm_fusion", "B")).strip().upper()
if _hf not in ("A", "B", "C"):
	_hf = "B"
logger.info("HPLSTM fusion scheme: %s (Decoder_%s.py)" % (_hf, {"A": "1", "B": "2", "C": "3"}[_hf]))

# ----- CUDA / 随机种子 / 多卡 -----
use_cuda, cuda_device, cuda_devices, multi_gpu, use_amp, use_cuda_bfmp, use_cuda_fp16 = parse_cuda(cnfg.use_cuda, gpuid=cnfg.gpuid, use_amp=cnfg.use_amp, use_cuda_bfmp=cnfg.use_cuda_bfmp)
set_random_seed(cnfg.seed, use_cuda)
multi_gpu_optimizer = multi_gpu and cnfg.multi_gpu_optimizer

# ----- 数据：PMaskDataConverter + 训练/验证 HDF5 -----
data_converter = PMaskDataConverter(xseql=cache_len_default, device=cuda_device)
td = h5File(cnfg.train_data, "r", **h5_fileargs)
vd = h5File(cnfg.dev_data, "r", **h5_fileargs)
ntrain = td["ndata"][()].item()
nvalid = vd["ndata"][()].item()
tl = [str(i) for i in range(ntrain)]   # 训练 batch key 列表 "0","1",...

# ----- 可选：prefix（固定前缀 token，验证时可复用 prefix_states） -----
prefix_ids = lcnfg.prefix_ids
if prefix_ids:
	prefix_len = len(prefix_ids)
elif lcnfg.find_common_prefix:
	prefix_ids = get_h5g_common_prefix(vd["src"])
	prefix_len = len(prefix_ids) if prefix_ids else None
else:
	prefix_len = None

# ----- 构建 Qwen3 Decoder（NMT 实为单 Decoder，无 Encoder） -----
logger.info("Design models with seed: %d" % torch.initial_seed())
mymodel = NMT(cnfg.isize, vocab_size, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, emb_w=None, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindemb=cnfg.bindDecoderEmb, num_kv_head=cnfg.kv_nhead, model_name=cnfg.model_name)
fine_tune_m = cnfg.fine_tune_m

# ----- 加载基座权重：优先 pre_trained_m（.h5/.bin），否则 fine_tune_m 或随机初始化 -----
if fine_tune_m is None:
	if cnfg.pre_trained_m is None:
		mymodel = init_model_params(mymodel)
		mymodel.apply(init_fixing)
	else:
		logger.info("Load pre-trained model from: " + cnfg.pre_trained_m)
		mymodel.load_plm(cnfg.pre_trained_m)
else:
	logger.info("Load pre-trained model from: " + fine_tune_m)
	mymodel = load_model_cpu(fine_tune_m, mymodel)
	mymodel.apply(load_fixing)

#lossf = NLLLoss(ignore_index=-1, reduction="sum")
lossf = LabelSmoothingLoss(vocab_size, cnfg.label_smoothing, ignore_index=-1, reduction="sum", forbidden_index=cnfg.forbidden_indexes)

if cnfg.src_emb is not None:
	logger.info("Load source embedding from: " + cnfg.src_emb)
	load_emb(cnfg.src_emb, mymodel.enc.wemb.weight, vocab_size, cnfg.scale_down_emb, cnfg.freeze_srcemb)
if cnfg.tgt_emb is not None:
	logger.info("Load target embedding from: " + cnfg.tgt_emb)
	load_emb(cnfg.tgt_emb, mymodel.dec.wemb.weight, vocab_size, cnfg.scale_down_emb, cnfg.freeze_tgtemb)

if lcnfg.save_base:
	save_model(mymodel, wkdir + "base.h5", False, print_func=logger.info, ps_func=None, h5args=h5zipargs)

# ----- 训练模式二选一：LoRA（lora_features 非空）/ 仅 HPLSTM（本脚本默认：lora_features 为空） -----
# HPLSTM：先冻结全模型参数，再仅解冻各层 DecoderLayer 中名为 hplstm 的子模块（Decoder_1/2）；Decoder_3 用 reslstm。
# LoRA：冻结后 std2lora 替换 Linear/Embedding，仅训 LoRA 分支。
if lcnfg.lora_features is not None:
	freeze_module(mymodel)
	mymodel = std2lora(
		mymodel,
		lora_features=lcnfg.lora_features,
		lora_alpha=lcnfg.lora_alpha,
		scaling=lcnfg.scaling,
		update_bias=lcnfg.update_bias,
		name_cfunc=lcnfg.name_cfunc,
		keep_lora_weight_tying=lcnfg.keep_lora_weight_tying,
	)[0]
	if lcnfg.fine_tune_linear_bias:
		unfreeze_linear_bias(mymodel, name_cfunc=lcnfg.name_cfunc_lb)
	if lcnfg.fine_tune_normer:
		unfreeze_normer(mymodel, name_cfunc=lcnfg.name_cfunc_normer)
	if lcnfg.lora_fine_tune_m is not None:
		mymodel = load_model_cpu(lcnfg.lora_fine_tune_m, mymodel)
	save_model_ps_func = rgrad_filter
else:
	freeze_module(mymodel)
	if lcnfg.fine_tune_hplstm:
		unfreeze_hplstm(mymodel)
	if lcnfg.fine_tune_reslstm:
		unfreeze_reslstm(mymodel)
	save_model_ps_func = rgrad_filter



# ----- 设备与多卡 / AMP / 优化器 / 学习率调度 / 状态保存 Holder -----
if use_cuda_bfmp:
	make_mp_model(mymodel)
	Optimizer = mp_optm_agent_wrapper(Optimizer)
if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)
	lossf.to(cuda_device, non_blocking=True)
scaler = (MultiGPUGradScaler() if multi_gpu_optimizer else GradScaler()) if use_amp else None
if multi_gpu:
	mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)
	lossf = DataParallelCriterion(lossf, device_ids=cuda_devices, output_device=cuda_device.index, replicate_once=True)
if multi_gpu:
	optimizer = mymodel.build_optimizer(Optimizer, lr=init_lr, betas=adam_betas_default, eps=ieps_adam_default, weight_decay=cnfg.weight_decay, amsgrad=use_ams, multi_gpu_optimizer=multi_gpu_optimizer, contiguous_parameters=contiguous_parameters)
else:
	optimizer = Optimizer(get_model_parameters(mymodel, contiguous_parameters=contiguous_parameters), lr=init_lr, betas=adam_betas_default, eps=ieps_adam_default, weight_decay=cnfg.weight_decay, amsgrad=use_ams)
optimizer.zero_grad(set_to_none=optm_step_zero_grad_set_none)

lrsch = LRScheduler(optimizer)#, cnfg.warm_step, dmodel=cnfg.isize, scale=cnfg.lr_scale

mymodel = torch_compile(mymodel, *torch_compile_args, **torch_compile_kwargs)
lossf = torch_compile(lossf, *torch_compile_args, **torch_compile_kwargs)

state_holder = None if statesf is None and cnt_states is None else Holder(**{"optm": optimizer, "lrsch": lrsch, "pyrand": PyRandomState(), "thrand": THRandomState(use_cuda=use_cuda)})

num_checkpoint = cnfg.num_checkpoint
cur_checkid = 0

tminerr = inf_default

# ----- 初始验证并保存 init / 或加载训练状态续训 -----
minloss, minerr = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu, use_amp)
logger.info("Init lr: %s, Dev Loss/Error: %.3f %.2f" % (" ".join(iter_to_str(getlr(optimizer))), minloss, minerr,))
if fine_tune_m is None:
	save_model(mymodel, wkdir + "init.h5", multi_gpu, print_func=logger.info, ps_func=save_model_ps_func)
	logger.info("Initial model saved")
else:
	if cnt_states is not None:
		logger.info("Loading training states")
		_remain_states = state_holder.load_state_dict(torch.load(cnt_states))
		remain_steps, cur_checkid = _remain_states["remain_steps"], _remain_states["checkpoint_id"]
		if "training_list" in _remain_states:
			_ctl = _remain_states["training_list"]
		else:
			shuffle(tl)
			_ctl = tl
		tminerr, done_tokens, cur_checkid, remain_steps, _ = train(td, _ctl, vd, nvalid, optimizer, lrsch, mymodel, lossf, cuda_device, logger, done_tokens, multi_gpu, multi_gpu_optimizer, tokens_optm, batch_report, save_every, chkpf, state_holder, statesf, num_checkpoint, cur_checkid, report_eva, remain_steps, False, False, scaler)
		_ctl = _remain_states = None
		vloss, vprec = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu, use_amp)
		logger.info("Epoch: 0, train loss: %.3f, valid loss/error: %.3f %.2f" % (tminerr, vloss, vprec,))
		save_model(mymodel, wkdir + "train_0_%.3f_%.3f_%.2f.h5" % (tminerr, vloss, vprec,), multi_gpu, print_func=logger.info, mtyp=("eva" if overwrite_eva else "train") if save_auto_clean else None, ps_func=save_model_ps_func)
		if statesf is not None:
			save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
		logger.info("New best model saved")

# ----- 动态样本选择（DSS）相关；不用则 dss_ws=0 -----
if cnfg.dss_ws is not None and cnfg.dss_ws > 0.0 and cnfg.dss_ws < 1.0:
	dss_ws = int(cnfg.dss_ws * ntrain)
	_Dws = {}
	_prev_Dws = {}
	_crit_inc = {}
	if cnfg.dss_rm is not None and cnfg.dss_rm > 0.0 and cnfg.dss_rm < 1.0:
		dss_rm = int(cnfg.dss_rm * ntrain * (1.0 - cnfg.dss_ws))
	else:
		dss_rm = 0
else:
	dss_ws = 0
	dss_rm = 0
	_Dws = None

namin = 0   # 连续多少 epoch 未刷新最佳验证，用于 early stop

# ----- 主训练循环：每 epoch shuffle tl，train -> eva -> 按条件保存 / early stop -----
for i in range(1, maxrun + 1):
	shuffle(tl)
	free_cache(use_cuda)
	terr, done_tokens, cur_checkid, remain_steps, _Dws = train(td, tl, vd, nvalid, optimizer, lrsch, mymodel, lossf, cuda_device, logger, done_tokens, multi_gpu, multi_gpu_optimizer, tokens_optm, batch_report, save_every, chkpf, state_holder, statesf, num_checkpoint, cur_checkid, report_eva, remain_steps, dss_ws > 0, i >= start_chkp_save, scaler)
	vloss, vprec = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu, use_amp)
	logger.info("Epoch: %d, train loss: %.3f, valid loss/error: %.3f %.2f" % (i, terr, vloss, vprec,))

	if (vprec <= minerr) or (vloss <= minloss):
		save_model(mymodel, wkdir + "eva_%d_%.3f_%.3f_%.2f.h5" % (i, terr, vloss, vprec,), multi_gpu, print_func=logger.info, mtyp="eva" if save_auto_clean else None, ps_func=save_model_ps_func)
		if statesf is not None:
			save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
		logger.info("New best model saved")

		namin = 0

		if vprec < minerr:
			minerr = vprec
		if vloss < minloss:
			minloss = vloss

	else:
		if terr < tminerr:
			tminerr = terr
			save_model(mymodel, wkdir + "train_%d_%.3f_%.3f_%.2f.h5" % (i, terr, vloss, vprec,), multi_gpu, print_func=logger.info, mtyp=("eva" if overwrite_eva else "train") if save_auto_clean else None, ps_func=save_model_ps_func)
			if statesf is not None:
				save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
		elif epoch_save:
			save_model(mymodel, wkdir + "epoch_%d_%.3f_%.3f_%.2f.h5" % (i, terr, vloss, vprec,), multi_gpu, print_func=logger.info, ps_func=save_model_ps_func)
			if statesf is not None:
				save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)

		namin += 1
		if namin >= earlystop:
			if done_tokens > 0:
				optm_step(optimizer, model=mymodel, scaler=scaler, multi_gpu=multi_gpu, multi_gpu_optimizer=multi_gpu_optimizer)
				lrsch.step()
				done_tokens = 0
			logger.info("early stop")
			break

	if remain_steps is not None and remain_steps <= 0:
		logger.info("Last training step reached")
		break
	# 动态样本选择：根据本 epoch 每 batch loss 变化更新下一轮 tl
	if dss_ws > 0:
		if _prev_Dws:
			for _key, _value in _Dws.items():
				if _key in _prev_Dws:
					_ploss = _prev_Dws[_key]
					_crit_inc[_key] = (_ploss - _value) / _ploss
			tl = dynamic_sample(_crit_inc, dss_ws, dss_rm)
		_prev_Dws = _Dws

# ----- 收尾：未满 tokens_optm 的梯度 step 一次，保存 last.h5 与状态 -----
if done_tokens > 0:
	optm_step(optimizer, model=mymodel, scaler=scaler, multi_gpu=multi_gpu, multi_gpu_optimizer=multi_gpu_optimizer)
	lrsch.step()
save_model(mymodel, wkdir + "last.h5", multi_gpu, print_func=logger.info, ps_func=save_model_ps_func)
if statesf is not None:
	save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
logger.info("model saved")

td.close()
vd.close()
