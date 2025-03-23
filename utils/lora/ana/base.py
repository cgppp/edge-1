#encoding: utf-8

from json import dumps

from utils.fmt.base import sys_open

class Ana:

	def __init__(self, logf, model=None, **kwargs):

		self.logf, self.model = sys_open(logf, "wb"), model

	def __call__(self, model=None):

		_ = self.ana(model=model)
		if _:
			self.logf.write(dumps(_, ensure_ascii=False).encode("utf-8"))
			self.logf.write("\n".encode("utf-8"))

	def close(self):

		self.logf.close()
