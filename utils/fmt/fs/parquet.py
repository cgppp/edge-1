#encoding: utf-8

from pyarrow.parquet import ParquetFile

def line_reader(fname, batch_size=65536, row_groups=None, columns=None, **kwargs):

	with ParquetFile(fname) as f:
		for _b in f.iter_batches(batch_size=batch_size, row_groups=row_groups, columns=columns):
			for _ in _b.to_pylist():
				yield _
