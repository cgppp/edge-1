#encoding: utf-8

import torch
from os import remove
from os.path import exists as fs_check
from threading import Thread

from utils.fmt.base import dict_is_list
from utils.h5serial import h5File, h5ensure_tensor, h5save
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import h5_fileargs, h5modelwargs, hdf5_save_parameter_name, list_key_func, n_keep_best#, hdf5_load_parameter_name

def load_model_cpu_p_ohf(ohf, base_model, list_key_func=list_key_func, **kwargs):

	with torch_no_grad():
		for i, para in enumerate(base_model.parameters()):
			_key = list_key_func(i)
			if _key in ohf:
				para.copy_(h5ensure_tensor(ohf[_key]))

	return base_model

def load_model_cpu_np_ohf(ohf, base_model, **kwargs):

	with torch_no_grad():
		for _n, _p in base_model.state_dict().items():
			if _n in ohf:
				_p.copy_(h5ensure_tensor(ohf[_n]))

	return base_model

def load_model_cpu_auto_ohf(ohf, base_model, **kwargs):

		return (load_model_cpu_p_ohf if dict_is_list(set(ohf.keys()), kfunc=list_key_func) else load_model_cpu_np_ohf)(ohf, base_model, **kwargs)

def h5_reader_wrapper(h5f, func, *args, **kwargs):

	if isinstance(h5f, str):
		with h5File(h5f, "r", **h5_fileargs) as _:
			return func(_, *args, **kwargs)
	else:
		return func(h5f, *args, **kwargs)

def load_model_cpu_p(modf, base_model, **kwargs):

		return h5_reader_wrapper(modf, load_model_cpu_p_ohf, base_model, **kwargs)

def load_model_cpu_np(modf, base_model, **kwargs):

		return h5_reader_wrapper(modf, load_model_cpu_np_ohf, base_model, **kwargs)

def load_model_cpu_auto(modf, base_model, **kwargs):

		return h5_reader_wrapper(modf, load_model_cpu_auto_ohf, base_model, **kwargs)

mp_func_p = lambda m: [_t.data for _t in m.parameters()]
mp_func_np = lambda m: {_k: _t.data for _k, _t in m.named_parameters()}

load_model_cpu = load_model_cpu_auto#load_model_cpu_np if hdf5_load_parameter_name else load_model_cpu_p
mp_func = mp_func_np if hdf5_save_parameter_name else mp_func_p

class bestfkeeper:

	def __init__(self, fnames=None, k=n_keep_best, **kwargs):

		self.fnames, self.k = [] if fnames is None else fnames, k
		self.clean()

	def update(self, fname=None):

		self.fnames.append(fname)
		self.clean(last_fname=fname)

	def clean(self, last_fname=None):

		_n_files = len(self.fnames)
		_last_fname = (self.fnames[-1] if self.fnames else None) if last_fname is None else last_fname
		while _n_files > self.k:
			fname = self.fnames.pop(0)
			if (fname is not None) and (fname != _last_fname) and fs_check(fname):
				try:
					remove(fname)
				except Exception as e:
					print(e)
			_n_files -= 1

class SaveModelCleaner:

	def __init__(self, *args, **kwargs):

		self.holder = {}

	def __call__(self, fname, typename, **kwargs):

		if typename in self.holder:
			self.holder[typename].update(fname)
		else:
			self.holder[typename] = bestfkeeper(fnames=[fname])

save_model_cleaner = SaveModelCleaner()

def save_model(model, fname, sub_module=False, print_func=print, mtyp=None, h5args=h5modelwargs):

	_msave = model.module if sub_module else model
	try:
		h5save(mp_func(_msave), fname, h5args=h5args)
		if mtyp is not None:
			save_model_cleaner(fname, mtyp)
	except Exception as e:
		if print_func is not None:
			print_func(str(e))

def async_save_model(model, fname, sub_module=False, print_func=print, mtyp=None, h5args=h5modelwargs, para_lock=None, log_success=None):

	def _worker(model, fname, sub_module=False, print_func=print, mtyp=None, para_lock=None, log_success=None):

		success = True
		_msave = model.module if sub_module else model
		try:
			if para_lock is None:
				h5save(mp_func(_msave), fname, h5args=h5args)
				if mtyp is not None:
					save_model_cleaner(fname, mtyp)
			else:
				with para_lock:
					h5save(mp_func(_msave), fname, h5args=h5args)
					if mtyp is not None:
						save_model_cleaner(fname, mtyp)
		except Exception as e:
			if print_func is not None:
				print_func(str(e))
			success = False
		if success and (print_func is not None) and (log_success is not None):
			print_func(str(log_success))

	Thread(target=_worker, args=(model, fname, sub_module, print_func, mtyp, para_lock, log_success)).start()

def save_states(state_dict, fname, print_func=print, mtyp=None):

	try:
		torch.save(state_dict, fname)
		if mtyp is not None:
			save_model_cleaner(fname, mtyp)
	except Exception as e:
		if print_func is not None:
			print_func(str(e))
