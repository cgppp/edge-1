# encoding: utf-8
import torch
from random import shuffle
# from torch import nn # 注释掉的导入
from torch.optim import Adam as Optimizer
from loss.base import LabelSmoothingLoss  # 标签平滑损失函数，防止过拟合
from lrsch import GoogleLR as LRScheduler  # 学习率调度器 (类似 Transformer 的 warmup + decay)
from optm.agent import fp32_optm_agent_wrapper as mp_optm_agent_wrapper
from parallel.base import DataParallelCriterion
from parallel.optm import MultiGPUGradScaler
from parallel.parallelMT import DataParallelMT  # 自定义的多 GPU 数据并行包装器
from transformer.NMT import NMT  # NMT 模型定义
from utils.base import free_cache, get_logger, mkdir, set_random_seed
from utils.contpara import get_model_parameters
from utils.fmt.base import iter_to_str
from utils.fmt.base4torch import load_emb, parse_cuda
from utils.h5serial import h5File  # 用于读取 HDF5 格式的数据集
from utils.init.base import init_model_params
from utils.io import load_model_cpu, save_model, save_states
from utils.norm.mp.f import convert as make_mp_model
from utils.state.holder import Holder
from utils.state.pyrand import PyRandomState
from utils.state.thrand import THRandomState
from utils.torch.comp import GradScaler, torch_autocast, torch_compile, torch_inference_mode
from utils.tqdm import tqdm
from utils.train.base import getlr, optm_step, optm_step_zero_grad_set_none, reset_Adam
from utils.train.dss import dynamic_sample  # 动态采样策略
import cnfg.base as cnfg
from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

def train(td, tl, ed, nd, optm, lrsch, model, lossf, mv_device, logger, done_tokens, multi_gpu, multi_gpu_optimizer, tokens_optm=32768, nreport=None, save_every=None, chkpf=None, state_holder=None, statesf=None, num_checkpoint=1, cur_checkid=0, report_eva=True, remain_steps=None, save_loss=False, save_checkp_epoch=False, scaler=None):
    """
    核心训练函数。
    参数:
        td: 训练数据句柄
        tl: 训练数据索引列表
        ed: 验证数据句柄
        nd: 验证数据数量
        optm: 优化器
        lrsch: 学习率调度器
        model: 模型
        lossf: 损失函数
        mv_device: 移动到的设备 (GPU)
        logger: 日志记录器
        done_tokens: 已处理的 token 总数
        tokens_optm: 每次优化步骤累积的 token 数 (梯度累积)
        ...其他参数涉及检查点保存、报告频率等
    """
    sum_loss = part_loss = 0.0
    sum_wd = part_wd = 0
    # 初始化局部变量
    _done_tokens, _cur_checkid, _cur_rstep, _use_amp = done_tokens, cur_checkid, remain_steps, scaler is not None
    global minerr, minloss, wkdir, save_auto_clean, namin
    
    model.train() # 设置模型为训练模式
    cur_b, _ls = 1, {} if save_loss else None
    src_grp, tgt_grp = td["src"], td["tgt"]
    
    # 遍历训练数据索引
    for i_d in tqdm(tl, mininterval=tqdm_mininterval):
        # 从 HDF5 文件读取批次数据
        seq_batch = torch.from_numpy(src_grp[i_d][()])
        seq_o = torch.from_numpy(tgt_grp[i_d][()])
        lo = seq_o.size(1) - 1 # 目标序列长度减1 (用于 shift 操作)
        
        # 数据移至 GPU
        if mv_device:
            seq_batch = seq_batch.to(mv_device, non_blocking=True)
            seq_o = seq_o.to(mv_device, non_blocking=True)
        
        # 确保数据类型为 int64
        seq_batch, seq_o = seq_batch.to(torch.int64, non_blocking=True), seq_o.to(torch.int64, non_blocking=True)
        
        # 构造输入和目标
        # oi: 解码器输入 (去掉最后一个 token)
        # ot: 真实目标 (去掉第一个 token，即 <bos>)
        oi = seq_o.narrow(1, 0, lo)
        ot = seq_o.narrow(1, 1, lo).contiguous()
        
        # 混合精度训练上下文 (如果启用)
        with torch_autocast(enabled=_use_amp):
            output = model(seq_batch, oi)
            loss = lossf(output, ot)
            if multi_gpu:
                loss = loss.sum()
        
        loss_add = loss.data.item()
        
        # 反向传播
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
        
        # 统计非 padding 的 token 数量 (用于计算平均 loss)
        wd_add = ot.ne(pad_id).to(torch.int32, non_blocking=True).sum().item()
        
        # 释放显存
        loss = output = oi = ot = seq_batch = seq_o = None
        
        sum_loss += loss_add
        if save_loss:
            _ls[i_d] = loss_add / wd_add
        sum_wd += wd_add
        _done_tokens += wd_add
        
        # 梯度累积：当累积的 token 数达到阈值时，执行优化步骤
        if _done_tokens >= tokens_optm:
            optm_step(optm, model=model, scaler=scaler, multi_gpu=multi_gpu, multi_gpu_optimizer=multi_gpu_optimizer, zero_grad_none=optm_step_zero_grad_set_none)
            _done_tokens = 0
            
            # 处理基于步数的检查点保存
            if _cur_rstep is not None:
                if save_checkp_epoch and (save_every is not None) and (_cur_rstep % save_every == 0) and (chkpf is not None) and (_cur_rstep > 0):
                    if num_checkpoint > 1:
                        _fend = "_%d.h5" % (_cur_checkid)
                        _chkpf = chkpf[:-3] + _fend
                        _cur_checkid = (_cur_checkid + 1) % num_checkpoint
                    else:
                        _chkpf = chkpf
                    save_model(model, _chkpf, multi_gpu, print_func=logger.info)
                    if statesf is not None:
                        save_states(state_holder.state_dict(update=False, **{"remain_steps": _cur_rstep, "checkpoint_id": _cur_checkid, "training_list": tl[cur_b - 1:]}), statesf, print_func=logger.info)
                
                _cur_rstep -= 1
                if _cur_rstep <= 0:
                    break
            
            # 更新学习率
            lrsch.step()
        
        # 定期报告训练损失和验证结果
        if nreport is not None:
            part_loss += loss_add
            part_wd += wd_add
            if cur_b % nreport == 0:
                if report_eva:
                    # 执行验证
                    _leva, _eeva = eva(ed, nd, model, lossf, mv_device, multi_gpu, _use_amp)
                    logger.info("Average loss over %d tokens: %.3f, valid loss/error: %.3f %.2f" % (part_wd, part_loss / part_wd, _leva, _eeva,))
                    
                    # 如果验证效果提升，保存最佳模型
                    if (_eeva < minerr) or (_leva < minloss):
                        save_model(model, wkdir + "eva_%.3f_%.2f.h5" % (_leva, _eeva,), multi_gpu, print_func=logger.info, mtyp="ieva" if save_auto_clean else None)
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
        
        # 按 epoch 或固定步数保存检查点
        if save_checkp_epoch and (_cur_rstep is None) and (save_every is not None) and (cur_b % save_every == 0) and (chkpf is not None) and (cur_b < ntrain):
            if num_checkpoint > 1:
                _fend = "_%d.h5" % (_cur_checkid)
                _chkpf = chkpf[:-3] + _fend
                _cur_checkid = (_cur_checkid + 1) % num_checkpoint
            else:
                _chkpf = chkpf
            save_model(model, _chkpf, multi_gpu, print_func=logger.info)
            if statesf is not None:
                save_states(state_holder.state_dict(update=False, **{"remain_steps": _cur_rstep, "checkpoint_id": _cur_checkid, "training_list": tl[cur_b - 1:]}), statesf, print_func=logger.info)
        
        cur_b += 1
    
    # 处理最后剩余的梯度
    if part_wd != 0.0:
        logger.info("Average loss over %d tokens: %.3f" % (part_wd, part_loss / part_wd,))
    
    return sum_loss / sum_wd, _done_tokens, _cur_checkid, _cur_rstep, _ls

def eva(ed, nd, model, lossf, mv_device, multi_gpu, use_amp=False):
    """
    验证函数。
    计算验证集上的平均 Loss 和错误率 (Error Rate)。
    """
    r = w = 0
    sum_loss = 0.0
    model.eval() # 设置为评估模式
    src_grp, tgt_grp = ed["src"], ed["tgt"]
    
    with torch_inference_mode(): # 禁用梯度计算，节省显存
        for i in tqdm(range(nd), mininterval=tqdm_mininterval):
            bid = str(i)
            seq_batch = torch.from_numpy(src_grp[bid][()])
            seq_o = torch.from_numpy(tgt_grp[bid][()])
            lo = seq_o.size(1) - 1
            
            if mv_device:
                seq_batch = seq_batch.to(mv_device, non_blocking=True)
                seq_o = seq_o.to(mv_device, non_blocking=True)
            
            seq_batch, seq_o = seq_batch.to(torch.int64, non_blocking=True), seq_o.to(torch.int64, non_blocking=True)
            
            ot = seq_o.narrow(1, 1, lo).contiguous()
            
            with torch_autocast(enabled=use_amp):
                output = model(seq_batch, seq_o.narrow(1, 0, lo))
                loss = lossf(output, ot)
                if multi_gpu:
                    loss = loss.sum()
                    trans = torch.cat([outu.argmax(-1).to(mv_device, non_blocking=True) for outu in output], 0)
                else:
                    trans = output.argmax(-1) # 获取预测的 token ID
            
            sum_loss += loss.data.item()
            
            # 计算正确率
            data_mask = ot.ne(pad_id) # 忽略 padding
            correct = (trans.eq(ot) & data_mask).to(torch.int32, non_blocking=True)
            w += data_mask.to(torch.int32, non_blocking=True).sum().item()
            r += correct.sum().item()
            
            # 清理变量
            correct = data_mask = trans = loss = output = ot = seq_batch = seq_o = None
    
    w = float(w)
    # 返回平均 loss 和错误率百分比
    return sum_loss / w, (w - r) / w * 100.0

def hook_lr_update(optm, flags=None):
    reset_Adam(optm, flags)

def init_fixing(module):
    if hasattr(module, "fix_init"):
        module.fix_init()

def load_fixing(module):
    if hasattr(module, "fix_load"):
        module.fix_load()

# --- 主程序入口 ---

# 加载配置文件参数
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

# 创建工作目录
wkdir = "".join((cnfg.exp_dir, cnfg.data_id, "/", cnfg.group_id, "/", rid, "/"))
mkdir(wkdir)

chkpf = None
statesf = None
if save_every is not None:
    chkpf = wkdir + "checkpoint.h5"
if cnfg.save_train_state:
    statesf = wkdir + "train.states.t7"

logger = get_logger(wkdir + "train.log")

# 解析 CUDA 配置
use_cuda, cuda_device, cuda_devices, multi_gpu, use_amp, use_cuda_bfmp, use_cuda_fp16 = parse_cuda(cnfg.use_cuda, gpuid=cnfg.gpuid, use_amp=cnfg.use_amp, use_cuda_bfmp=cnfg.use_cuda_bfmp)
set_random_seed(cnfg.seed, use_cuda)

multi_gpu_optimizer = multi_gpu and cnfg.multi_gpu_optimizer

# 打开训练和验证数据文件 (HDF5)
td = h5File(cnfg.train_data, "r", **h5_fileargs)
vd = h5File(cnfg.dev_data, "r", **h5_fileargs)
ntrain = td["ndata"][()].item()
nvalid = vd["ndata"][()].item()
nword = td["nword"][()].tolist()
nwordi, nwordt = nword[0], nword[-1]
tl = [str(i) for i in range(ntrain)]

logger.info("Design models with seed: %d" % torch.initial_seed())

# 初始化 NMT 模型
mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes)

fine_tune_m = cnfg.fine_tune_m
if fine_tune_m is None:
    # 随机初始化
    mymodel = init_model_params(mymodel)
    mymodel.apply(init_fixing)
else:
    # 加载预训练模型进行微调
    logger.info("Load pre-trained model from: " + fine_tune_m)
    mymodel = load_model_cpu(fine_tune_m, mymodel)
    mymodel.apply(load_fixing)

# 初始化损失函数 (标签平滑)
lossf = LabelSmoothingLoss(nwordt, cnfg.label_smoothing, ignore_index=pad_id, reduction="sum", forbidden_index=cnfg.forbidden_indexes)

# 加载预训练的词向量 (可选)
if cnfg.src_emb is not None:
    logger.info("Load source embedding from: " + cnfg.src_emb)
    load_emb(cnfg.src_emb, mymodel.enc.wemb.weight, nwordi, cnfg.scale_down_emb, cnfg.freeze_srcemb)
if cnfg.tgt_emb is not None:
    logger.info("Load target embedding from: " + cnfg.tgt_emb)
    load_emb(cnfg.tgt_emb, mymodel.dec.wemb.weight, nwordt, cnfg.scale_down_emb, cnfg.freeze_tgtemb)

# 处理混合精度/特殊精度格式
if use_cuda_bfmp:
    make_mp_model(mymodel)
    Optimizer = mp_optm_agent_wrapper(Optimizer)

if cuda_device:
    mymodel.to(cuda_device, non_blocking=True)
    lossf.to(cuda_device, non_blocking=True)

# 初始化 GradScaler (用于 AMP)
scaler = (MultiGPUGradScaler() if multi_gpu_optimizer else GradScaler()) if use_amp else None

# 多 GPU 包装
if multi_gpu:
    mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)
    lossf = DataParallelCriterion(lossf, device_ids=cuda_devices, output_device=cuda_device.index, replicate_once=True)

# 初始化优化器
if multi_gpu:
    optimizer = mymodel.build_optimizer(Optimizer, lr=init_lr, betas=adam_betas_default, eps=ieps_adam_default, weight_decay=cnfg.weight_decay, amsgrad=use_ams, multi_gpu_optimizer=multi_gpu_optimizer, contiguous_parameters=contiguous_parameters)
else:
    optimizer = Optimizer(get_model_parameters(mymodel, contiguous_parameters=contiguous_parameters), lr=init_lr, betas=adam_betas_default, eps=ieps_adam_default, weight_decay=cnfg.weight_decay, amsgrad=use_ams)

optimizer.zero_grad(set_to_none=optm_step_zero_grad_set_none)

# 初始化学习率调度器
lrsch = LRScheduler(optimizer, cnfg.warm_step, dmodel=cnfg.isize, scale=cnfg.lr_scale)

# 编译模型 (PyTorch 2.0+)
mymodel = torch_compile(mymodel, *torch_compile_args, **torch_compile_kwargs)
lossf = torch_compile(lossf, *torch_compile_args, **torch_compile_kwargs)

# 初始化状态持有者 (用于保存/恢复训练状态)
state_holder = None if statesf is None and cnt_states is None else Holder(**{"optm": optimizer, "lrsch": lrsch, "pyrand": PyRandomState(), "thrand": THRandomState(use_cuda=use_cuda)})

num_checkpoint = cnfg.num_checkpoint
cur_checkid = 0
tminerr = inf_default

# 初始验证
minloss, minerr = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu, use_amp)
logger.info("Init lr: %s, Dev Loss/Error: %.3f %.2f" % (" ".join(iter_to_str(getlr(optimizer))), minloss, minerr,))

if fine_tune_m is None:
    save_model(mymodel, wkdir + "init.h5", multi_gpu, print_func=logger.info)
    logger.info("Initial model saved")
else:
    # 如果是微调且需要恢复状态
    if cnt_states is not None:
        logger.info("Loading training states")
        _remain_states = state_holder.load_state_dict(torch.load(cnt_states))
        remain_steps, cur_checkid = _remain_states["remain_steps"], _remain_states["checkpoint_id"]
        if "training_list" in _remain_states:
            _ctl = _remain_states["training_list"]
        else:
            shuffle(tl)
            _ctl = tl
        
        # 继续训练
        tminerr, done_tokens, cur_checkid, remain_steps, _ = train(td, _ctl, vd, nvalid, optimizer, lrsch, mymodel, lossf, cuda_device, logger, done_tokens, multi_gpu, multi_gpu_optimizer, tokens_optm, batch_report, save_every, chkpf, state_holder, statesf, num_checkpoint, cur_checkid, report_eva, remain_steps, False, False, scaler)
        _ctl = _remain_states = None
        
        vloss, vprec = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu, use_amp)
        logger.info("Epoch: 0, train loss: %.3f, valid loss/error: %.3f %.2f" % (tminerr, vloss, vprec,))
        save_model(mymodel, wkdir + "train_0_%.3f_%.3f_%.2f.h5" % (tminerr, vloss, vprec,), multi_gpu, print_func=logger.info, mtyp=("eva" if overwrite_eva else "train") if save_auto_clean else None)
        if statesf is not None:
            save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
        logger.info("New best model saved")

# 动态采样配置 (Dynamic Sample Selection)
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

namin = 0

# 主训练循环
for i in range(1, maxrun + 1):
    shuffle(tl)
    free_cache(use_cuda)
    
    # 执行一个 epoch 的训练
    terr, done_tokens, cur_checkid, remain_steps, _Dws = train(td, tl, vd, nvalid, optimizer, lrsch, mymodel, lossf, cuda_device, logger, done_tokens, multi_gpu, multi_gpu_optimizer, tokens_optm, batch_report, save_every, chkpf, state_holder, statesf, num_checkpoint, cur_checkid, report_eva, remain_steps, dss_ws > 0, i >= start_chkp_save, scaler)
    
    # 验证
    vloss, vprec = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu, use_amp)
    logger.info("Epoch: %d, train loss: %.3f, valid loss/error: %.3f %.2f" % (i, terr, vloss, vprec,))
    
    # 判断是否保存最佳模型
    if (vprec <= minerr) or (vloss <= minloss):
        save_model(mymodel, wkdir + "eva_%d_%.3f_%.3f_%.2f.h5" % (i, terr, vloss, vprec,), multi_gpu, print_func=logger.info, mtyp="eva" if save_auto_clean else None)
        if statesf is not None:
            save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
        logger.info("New best model saved")
        namin = 0
        if vprec < minerr:
            minerr = vprec
        if vloss < minloss:
            minloss = vloss
    else:
        # 如果没有提升，但训练 loss 降低了，也可以保存
        if terr < tminerr:
            tminerr = terr
            save_model(mymodel, wkdir + "train_