#include <torch/extension.h>
#include <vector>
#include "omp.h"

// fgate, igh, cell: (bsize, seql, nhead, isize)
// init_cell: (nhead, isize)
template <typename scalar_t> inline void omp_lgate_forward_(scalar_t *fgate, scalar_t *igh, scalar_t *init_cell, scalar_t *cell, int bsize, int seqlen, int nhead, int isize) {

	int _seq_shift = nhead * isize;
	#pragma omp parallel for collapse(3) schedule(static)
	for (int i_bsize = 0; i_bsize < bsize; i_bsize++) {
		for (int i_head = 0; i_head < nhead; i_head ++) {
			for (int i_isize = 0; i_isize < isize; i_isize++) {
				int _h_shift = i_head * isize;
				int _i_base = i_bsize * seqlen * _seq_shift + _h_shift + i_isize;
				scalar_t c = igh[_i_base] + init_cell[_h_shift + i_isize] * fgate[_i_base];
				cell[_i_base] = c;
				for (int i = 1; i < seqlen; i++) {
					_i_base += _seq_shift;
					cell[_i_base] = c = c * fgate[_i_base] + igh[_i_base];
				}
			}
		}
	}
}

// grad_cell, cell, fgate, grad_fgate, grad_igh: (bsize, seql, nhead, isize)
// init_cell: (nhead, isize)
// grad_prev_cell: (bsize, nhead, isize)
template <typename scalar_t> inline void omp_lgate_backward_(torch::TensorAccessor<scalar_t, 4> grad_cell, scalar_t *cell, scalar_t *fgate, scalar_t *init_cell, scalar_t *grad_fgate, scalar_t *grad_igh, scalar_t *grad_prev_cell, int bsize, int seqlen, int nhead, int isize) {

	int _seq_shift = nhead * isize;
	int last_index = seqlen - 1;
	if (last_index > 0) {
		#pragma omp parallel for collapse(3) schedule(static)
		for (int i_bsize = 0; i_bsize < bsize; i_bsize++) {
			for (int i_head = 0; i_head < nhead; i_head ++) {
				for (int i_isize = 0; i_isize < isize; i_isize++) {
					int _bhi_base = i_bsize * _seq_shift;
					int _h_base = i_head * isize;
					int _i_base = _bhi_base * seqlen + last_index * _seq_shift + _h_base + i_isize;
					scalar_t agc = grad_cell[i_bsize][last_index][i_head][i_isize];
					for (int i = last_index; i > 0; i--) {
						grad_igh[_i_base] = agc;
						int _i_base_new = _i_base - _seq_shift;
						grad_fgate[_i_base] = agc * cell[_i_base_new];
						agc = agc * fgate[_i_base] + grad_cell[i_bsize][i - 1][i_head][i_isize];
						_i_base = _i_base_new;
					}
					grad_igh[_i_base] = agc;
					grad_prev_cell[_bhi_base + _h_base + i_isize] = agc * fgate[_i_base];
					grad_fgate[_i_base] = agc * init_cell[_h_base + i_isize];
				}
			}
		}
	}
	else {
		#pragma omp parallel for collapse(3) schedule(static)
		for (int i_bsize = 0; i_bsize < bsize; i_bsize++) {
			for (int i_head = 0; i_head < nhead; i_head ++) {
				for (int i_isize = 0; i_isize < isize; i_isize++) {
					int _bhi_base = i_bsize * _seq_shift;
					int _h_base = i_head * isize;
					int _i_base = _bhi_base + _h_base + i_isize;// * seqlen equals to 1; + last_index * _seq_shift (last_index is 0)
					scalar_t gc = grad_cell[i_bsize][0][i_head][i_isize];
					grad_igh[_i_base] = gc;
					grad_prev_cell[_bhi_base + _h_base + i_isize] = gc * fgate[_i_base];
					grad_fgate[_i_base] = gc * init_cell[_h_base + i_isize];
				}
			}
		}
	}
}

template <typename scalar_t> at::Tensor lgate_forward(torch::Tensor fgate, torch::Tensor igh, torch::Tensor init_cell, torch::Tensor cell, int64_t bsize, int64_t seqlen, int64_t nhead, int64_t isize) {

	omp_lgate_forward_<scalar_t>(fgate.data_ptr<scalar_t>(), igh.data_ptr<scalar_t>(), init_cell.data_ptr<scalar_t>(), cell.data_ptr<scalar_t>(), (int)bsize, (int)seqlen, (int)nhead, (int)isize);

	return cell;
}

template <typename scalar_t> std::vector<torch::Tensor> lgate_backward(torch::Tensor grad_cell, torch::Tensor cell, torch::Tensor fgate, torch::Tensor init_cell, torch::Tensor grad_fgate, torch::Tensor grad_igh, torch::Tensor grad_prev_cell, int64_t bsize, int64_t seqlen, int64_t nhead, int64_t isize) {

	omp_lgate_backward_<scalar_t>(grad_cell.accessor<scalar_t, 4>(), cell.data_ptr<scalar_t>(), fgate.data_ptr<scalar_t>(), init_cell.data_ptr<scalar_t>(), grad_fgate.data_ptr<scalar_t>(), grad_igh.data_ptr<scalar_t>(), grad_prev_cell.data_ptr<scalar_t>(), (int)bsize, (int)seqlen, (int)nhead, (int)isize);

	return {grad_fgate, grad_igh, grad_prev_cell};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &lgate_forward<float>, "LGate forward");
	m.def("backward", &lgate_backward<float>, "LGate backward");
}
