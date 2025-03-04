#encoding: utf-8

import torch
from torch.optim.optimizer import Optimizer

from optm.wrapper import wrap
from utils.torch.comp import torch_no_grad

cpu_device = torch.device("cpu")

class OptmAgentCore(Optimizer):

	def __init__(self, Optm, params, *args, cfunc=lambda x: x.to(torch.float32, non_blocking=True), filter_by_rgrad=True, **kwargs):

		self.params = [_ for _ in params if _.requires_grad] if filter_by_rgrad else params
		self.c_params = [cfunc(_).detach() for _ in self.params]
		self.optm = Optm(self.params, *args, **kwargs)
		self.c_optm = Optm(self.c_params, *args, **kwargs)
		for _ in ["state_dict", "load_state_dict", "register_state_dict_pre_hook", "register_state_dict_post_hook", "register_load_state_dict_pre_hook", "register_load_state_dict_post_hook", "register_step_pre_hook", "register_step_post_hook"]:
			if hasattr(self.c_optm, _):
				setattr(self, _, getattr(self.c_optm, _))

	def step(self, *args, **kwargs):

		for _p, _cp in zip(self.params, self.c_params):
			if _p.grad is None:
				_cp.grad = None
			else:
				if _cp.grad is None:
					_cp.grad = _p.grad.to(device=_cp.data.device, dtype=_cp.data.dtype, non_blocking=True)
				else:
					_cp.grad.copy_(_p.grad)
		self.c_optm.step(*args, **kwargs)
		with torch_no_grad():
			for _p, _cp in zip(self.params, self.c_params):
				if _p.grad is not None:
					_p.copy_(_cp)

	def zero_grad(self, *args, **kwargs):

		self.optm.zero_grad(*args, **kwargs)
		self.c_optm.zero_grad(*args, **kwargs)

	def __getattr__(self, name):

		if hasattr(self.c_optm, name):

			return getattr(self.c_optm, name)

		raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

class FP32OptmAgent(OptmAgentCore):

	def __init__(self, Optm, params, *args, **kwargs):

		super(FP32OptmAgent, self).__init__(Optm, params, *args, cfunc=lambda x: x.to(torch.float32, non_blocking=True), **kwargs)

class CPUFP32OptmAgent(OptmAgentCore):

	def __init__(self, Optm, params, *args, **kwargs):

		super(CPUFP32OptmAgent, self).__init__(Optm, params, *args, cfunc=lambda x: x.to(device=cpu_device, dtype=torch.float32, non_blocking=True), **kwargs)

fp32_optm_agent_wrapper = wrap(FP32OptmAgent)
cpu_fp32_optm_agent_wrapper = wrap(CPUFP32OptmAgent)
