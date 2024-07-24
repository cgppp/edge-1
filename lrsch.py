#encoding: utf-8

from math import cos, pi, sqrt
from torch.optim.lr_scheduler import _LRScheduler

from cnfg.ihyp import init_lr

class GoogleLR(_LRScheduler):

	def __init__(self, optimizer, warm_steps, dmodel=None, scale=1.0, last_epoch=-1, cur_step=0, **kwargs):

		self.cur_step, self.warm_steps, self.k = cur_step, warm_steps, scale / sqrt(dmodel)
		self.wk = self.k / (warm_steps ** 1.5)
		super(GoogleLR, self).__init__(optimizer, last_epoch=last_epoch)

	def get_lr(self):

		self.cur_step += 1
		cur_lr = (self.cur_step * self.wk) if self.cur_step < self.warm_steps else (self.k / sqrt(self.cur_step))

		return [cur_lr for i in range(len(self.base_lrs))]

# inverse square root with warm up, portal from: https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py, equal to GoogleLR when warm_end_lr = scale / sqrt(dmodel * warm_steps)
class WarmUpInverseSqrtLR(_LRScheduler):

	def __init__(self, optimizer, warm_steps, warm_end_lr=None, warm_init_lr=0.0, dmodel=None, scale=1.0, last_epoch=-1, cur_step=0, **kwargs):

		self.cur_step, self.warm_steps, self.warm_init_lr = cur_step, warm_steps, warm_init_lr
		if dmodel is None:
			self.lr_step, self.decay_factor = (warm_end_lr - warm_init_lr) / warm_steps, warm_end_lr * sqrt(warm_steps)
		else:
			self.decay_factor = scale / sqrt(dmodel)
			self.lr_step = (self.decay_factor / sqrt(warm_steps) - warm_init_lr) / warm_steps
		super(WarmUpInverseSqrtLR, self).__init__(optimizer, last_epoch=last_epoch)

	def get_lr(self):

		self.cur_step += 1
		cur_lr = (self.warm_init_lr + self.cur_step * self.lr_step) if self.cur_step < self.warm_steps else (self.decay_factor / sqrt(self.cur_step))

		return [cur_lr for i in range(len(self.base_lrs))]

class InverseSqrtLR(_LRScheduler):

	def __init__(self, optimizer, lr=1e-4, scalar=1.0, min_lr=None, last_epoch=-1, cur_step=0, **kwargs):

		self.cur_step, self.base_lr, self.epoch_steps, self.min_lr = cur_step, lr, scalar, (lr / 512.0) if min_lr is None else min_lr
		super(InverseSqrtLR, self).__init__(optimizer, last_epoch=last_epoch)

	def get_lr(self):

		self.cur_step += 1
		cur_lr = max(min(1.0, 1.0 / sqrt(self.cur_step / self.epoch_steps)), self.min_lr) * self.base_lr

		return [cur_lr for i in range(len(self.base_lrs))]

class WarmUpCosineLR(_LRScheduler):

	def __init__(self, optimizer, warm_steps, warm_end_lr=None, min_lr=None, max_steps=None, ratio=0.001953125, max_step_factor=16, warm_init_lr=0.0, dmodel=None, scale=1.0, last_epoch=-1, cur_step=0, **kwargs):

		self.cur_step, self.warm_steps, self.warm_init_lr, self.max_steps = cur_step, warm_steps, warm_init_lr, (warm_steps * max_step_factor) if max_steps is None else max_steps
		_warm_end_lr = warm_end_lr if dmodel is None else (scale / sqrt(dmodel * warm_steps))
		self.lr_step, self.min_lr, self.beta = (_warm_end_lr - warm_init_lr) / warm_steps, (_warm_end_lr * ratio) if min_lr is None else min_lr, pi / float(max_steps - warm_steps)
		self.alpha = (_warm_end_lr - self.min_lr) / 2.0
		super(WarmUpCosineLR, self).__init__(optimizer, last_epoch=last_epoch)

	def get_lr(self):

		self.cur_step += 1
		cur_lr = ((self.warm_init_lr + self.cur_step * self.lr_step) if self.cur_step < self.warm_steps else (self.min_lr + self.alpha * (1.0 + cos((self.cur_step - self.warm_steps) * self.beta)))) if self.cur_step < self.max_steps else self.min_lr

		return [cur_lr for i in range(len(self.base_lrs))]

class CosineLR(_LRScheduler):

	def __init__(self, optimizer, lr=1e-4, min_lr=None, max_steps=None, ratio=0.001953125, last_epoch=-1, cur_step=0, **kwargs):

		self.cur_step, self.max_steps, self.min_lr, self.beta = cur_step, max_steps, (lr * ratio) if min_lr is None else min_lr, pi / float(max_steps)
		self.alpha = (lr - self.min_lr) / 2.0
		super(CosineLR, self).__init__(optimizer, last_epoch=last_epoch)

	def get_lr(self):

		self.cur_step += 1
		cur_lr = (self.min_lr + self.alpha * (1.0 + cos(self.cur_step * self.beta))) if self.cur_step < self.max_steps else self.min_lr

		return [cur_lr for i in range(len(self.base_lrs))]

class FullCosLR(_LRScheduler):

	def __init__(self, optimizer, warm_steps, warm_end_lr=None, min_lr=None, max_steps=None, ratio=0.001953125, max_step_factor=16, warm_init_lr=0.0, dmodel=None, scale=1.0, last_epoch=-1, cur_step=0, **kwargs):

		self.cur_step, self.warm_steps, self.warm_init_lr, self.max_steps = cur_step, warm_steps, warm_init_lr, (warm_steps * max_step_factor) if max_steps is None else max_steps
		_warm_end_lr = warm_end_lr if dmodel is None else (scale / sqrt(dmodel * warm_steps))
		self.betaw, self.alphaw, self.betar, self.min_lr = pi / float(warm_steps), (_warm_end_lr - warm_init_lr) / 2.0, pi / float(max_steps - warm_steps), (_warm_end_lr * ratio) if min_lr is None else min_lr
		self.alphar = (_warm_end_lr - self.min_lr) / 2.0
		super(FullCosLR, self).__init__(optimizer, last_epoch=last_epoch)

	def get_lr(self):

		self.cur_step += 1
		cur_lr = ((self.warm_init_lr + self.alphaw * (1.0 + cos(((self.cur_step - self.warm_steps) * self.betaw)))) if self.cur_step < self.warm_steps else (self.min_lr + self.alphar * (1.0 + cos((self.cur_step - self.warm_steps) * self.betar)))) if self.cur_step < self.max_steps else self.min_lr

		return [cur_lr for i in range(len(self.base_lrs))]

class WarmUpLinearLR(_LRScheduler):

	def __init__(self, optimizer, warm_steps, warm_end_lr=None, min_lr=None, max_steps=None, ratio=0.001953125, max_step_factor=16, warm_init_lr=0.0, dmodel=None, scale=1.0, last_epoch=-1, cur_step=0, **kwargs):

		self.cur_step, self.warm_steps, self.warm_init_lr, self.max_steps = cur_step, warm_steps, warm_init_lr, (warm_steps * max_step_factor) if max_steps is None else max_steps
		self.warm_end_lr = warm_end_lr if dmodel is None else (scale / sqrt(dmodel * warm_steps))
		self.lr_stepw, self.min_lr = (self.warm_end_lr - warm_init_lr) / warm_steps, (self.warm_end_lr * ratio) if min_lr is None else min_lr
		self.lr_stepr = (self.warm_end_lr - self.min_lr) / float(max_steps - warm_steps)
		super(WarmUpLinearLR, self).__init__(optimizer, last_epoch=last_epoch)

	def get_lr(self):

		self.cur_step += 1
		cur_lr = ((self.warm_init_lr + self.cur_step * self.lr_stepw) if self.cur_step < self.warm_steps else (self.warm_end_lr - (self.cur_step - self.warm_steps) * self.lr_stepr)) if self.cur_step < self.max_steps else self.min_lr

		return [cur_lr for i in range(len(self.base_lrs))]

class CustLR(_LRScheduler):

	def __init__(self, optimizer, lr_func=lambda a, b: (init_lr, b,), ctx=None, last_epoch=-1, cur_step=0, **kwargs):

		self.cur_step, self.lr_func, self.ctx = cur_step, lr_func, ctx
		super(CustLR, self).__init__(optimizer, last_epoch=last_epoch)

	def get_lr(self):

		self.cur_step += 1
		cur_lr, self.ctx = self.lr_func(self.cur_step, self.ctx)

		return [cur_lr for i in range(len(self.base_lrs))]
