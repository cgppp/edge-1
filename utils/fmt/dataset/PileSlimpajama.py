#encoding: utf-8

from os import walk
from os.path import join as pjoin

from utils.fmt.fs.parquet import line_reader as pq_line_reader
from utils.fmt.ifilter.emp import ifilter

data_key = "text"
data_columns = (data_key,)

def scan_train_set(ptw):

	rs = []
	for root, dirs, files in walk(ptw):
		for f in files:
			if f.startswith("train-") and f.endswith(".parquet"):
				rs.append(pjoin(root, f))

	return f

def line_reader_core(fname, batch_size=65536, row_groups=None, columns=data_columns, data_key=data_key, **kwargs):

	for _ in pq_line_reader(fname, batch_size=batch_size, row_groups=row_groups, columns=columns):
		yield _.get(data_key, "")

def line_reader(fname, batch_size=65536, row_groups=None, columns=data_columns, **kwargs):

	return ifilter(line_reader_core(fname, batch_size=batch_size, row_groups=row_groups, columns=columns))

def get_line_readers(ptw, batch_size=65536, row_groups=None, columns=data_columns, **kwargs):

	return [line_reader(_, batch_size=batch_size, row_groups=row_groups, columns=columns) for _ in scan_train_set(ptw)]
