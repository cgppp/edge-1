#encoding: utf-8

from multiprocessing import Manager, Value
from os.path import exists as fs_check
from shutil import rmtree
from time import sleep

from utils.h5serial import h5File

from cnfg.ihyp import h5_fileargs

class Loader:

	def __init__(self, sleep_secs=1.0, print_func=print, **kwargs):

		self.sleep_secs, self.print_func = sleep_secs, print_func
		self.manager = Manager()
		self.out = self.manager.list()
		self.todo = self.manager.list()
		self.out_lck = self.manager.Lock()
		self.todo_lck = self.manager.Lock()
		self.running = Value("B", 1, lock=True)
		self.cache_path = self.p_loader = self.iter = None

	def __call__(self, *args, **kwargs):

		if self.iter is None:
			self.iter = self.iter_func(*args, **kwargs)
		for _ in self.iter:
			yield _
		self.iter = None

	def get_todo(self):

		_cache_file = None
		while self.running.value and (_cache_file is None):
			with self.todo_lck:
				if self.todo:
					_cache_file = self.todo.pop(0)
			if self.running.value and (_cache_file is None):
				sleep(self.sleep_secs)

		return _cache_file

	def get_h5(self):

		td = _cache_file = None
		while self.running.value and (td is None):
			with self.out_lck:
				if self.out:
					_cache_file = self.out.pop(0)
			if (_cache_file is not None) and fs_check(_cache_file):
				try:
					td = h5File(_cache_file, "r", **h5_fileargs)
				except Exception as e:
					td = None
					if self.print_func is not None:
						self.print_func(e)
					with self.todo_lck:
						self.todo.append(_cache_file)
			if self.running.value and (td is None):
				sleep(self.sleep_secs)

		return td, _cache_file

	def status(self, mode=True):

		with self.running.get_lock():
			self.running.value = 1 if mode else 0

	def close(self):

		self.running.value = 0
		sleep(self.sleep_secs)
		if self.p_loader is not None:
			if self.p_loader.is_alive():
				try:
					self.p_loader.terminate()
				except Exception as e:
					if self.print_func is not None:
						self.print_func(e)
				if self.p_loader.is_alive():
					try:
						self.p_loader.kill()
					except Exception as e:
						if self.print_func is not None:
							self.print_func(e)
			if not self.p_loader.is_alive():
				self.p_loader.join(self.sleep_secs)
		if self.cache_path is not None:
			rmtree(self.cache_path, ignore_errors=True)
