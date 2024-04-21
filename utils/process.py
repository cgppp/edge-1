#encoding: utf-8

from multiprocessing import Process, active_children
from time import sleep

def start_process(*args, **kwargs):

	_ = Process(*args, **kwargs)
	_.start()

	return _

def process_keeper_core(t, sleep_secs, *args, **kwargs):

	if t.is_alive():
		sleep(sleep_secs)
	else:
		t.join()
		t.close()
		t = start_process(*args, **kwargs)

	return t

def process_keeper(condition, sleep_secs, *args, **kwargs):

	_t = start_process(*args, **kwargs)
	while condition.value:
		_t = process_keeper_core(_t, sleep_secs, *args, **kwargs)

def exit_all(print_func=None, **kwargs):

	for _ in active_children():
		if _.is_alive():
			try:
				_.terminate()
			except Exception as e:
				if print_func is not None:
					print_func(e)
			if _.is_alive():
				try:
					_.kill()
				except Exception as e:
					if print_func is not None:
						print_func(e)
