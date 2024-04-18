#encoding: utf-8

import torch

from utils.torch.comp import torch_any_wodim

from cnfg.vocab.gec.edit import blank_id, keep_id, mask_id as edit_mask_id
from cnfg.vocab.plm.custbert import init_normal_token_id, mask_id as plm_mask_id

def get_batch(seq_batch, seq_edt, seq_o, p=0.15, init_normal_token_id=init_normal_token_id, keep_id=keep_id, blank_id=blank_id, edit_mask_id=edit_mask_id, plm_mask_id=plm_mask_id):

	_n = seq_batch.numel()
	_m = torch.randperm(_n, dtype=torch.int32, device=seq_batch.device).view_as(seq_batch).lt(max(1, int(_n * p))) & seq_batch.ge(init_normal_token_id) & (seq_edt.eq(keep_id) | seq_edt.eq(blank_id))
	if torch_any_wodim(_m).item():
		seq_o.masked_scatter_(_m, seq_batch[_m])
		seq_batch.masked_fill_(_m, plm_mask_id)
		seq_edt.masked_fill_(_m, edit_mask_id)

	return seq_batch, seq_edt, seq_o
