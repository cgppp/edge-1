#include <torch/extension.h>
#include <vector>
#include "omp.h"

// fgate, igh, cell: (bsize, seql, nhead, isize)
// init_cell: (nhead, isize)
template <typename scalar_t> inline void omp_lgate_forward_(torch::TensorAccessor<scalar_t, 4> fgate, torch::TensorAccessor<scalar_t, 4> igh, torch::TensorAccessor<scalar_t, 2> init_cell, torch::TensorAccessor<scalar_t, 4> cell, int bsize, int seqlen, int nhead, int isize) {

	#pragma omp parallel for collapse(3) schedule(static)
	for (int i_bsize = 0; i_bsize < bsize; i_bsize++) {
		for (int i_head = 0; i_head < nhead; i_head ++) {
			for (int i_isize = 0; i_isize < isize; i_isize++) {
				scalar_t c = igh[i_bsize][0][i_head][i_isize] + init_cell[i_head][i_isize] * fgate[i_bsize][0][i_head][i_isize];
				cell[i_bsize][0][i_head][i_isize] = c;
				for (int i = 1; i < seqlen; i++) {
					cell[i_bsize][i][i_head][i_isize] = c = c * fgate[i_bsize][i][i_head][i_isize] + igh[i_bsize][i][i_head][i_isize];
				}
			}
		}
	}
}

// grad_cell, cell, fgate, grad_fgate, grad_igh: (bsize, seql, nhead, isize)
// init_cell: (nhead, isize)
// grad_prev_cell: (bsize, nhead, isize)
template <typename scalar_t> inline void omp_lgate_backward_(torch::TensorAccessor<scalar_t, 4> grad_cell, torch::TensorAccessor<scalar_t, 4> cell, torch::TensorAccessor<scalar_t, 4> fgate, torch::TensorAccessor<scalar_t, 2> init_cell, torch::TensorAccessor<scalar_t, 4> grad_fgate, torch::TensorAccessor<scalar_t, 4> grad_igh, torch::TensorAccessor<scalar_t, 3> grad_prev_cell, int bsize, int seqlen, int nhead, int isize) {

	int last_index = seqlen - 1;
	if (last_index > 0) {
		#pragma omp parallel for collapse(3) schedule(static)
		for (int i_bsize = 0; i_bsize < bsize; i_bsize++) {
			for (int i_head = 0; i_head < nhead; i_head ++) {
				for (int i_isize = 0; i_isize < isize; i_isize++) {
					int _prev_i;
					scalar_t agc = grad_cell[i_bsize][last_index][i_head][i_isize];
					for (int i = last_index; i > 0; ) {
						grad_igh[i_bsize][i][i_head][i_isize] = agc;
						_prev_i = i - 1;
						grad_fgate[i_bsize][i][i_head][i_isize] = agc * cell[i_bsize][_prev_i][i_head][i_isize];
						agc = agc * fgate[i_bsize][i][i_head][i_isize] + grad_cell[i_bsize][_prev_i][i_head][i_isize];
						i = _prev_i;
					}
					grad_igh[i_bsize][0][i_head][i_isize] = agc;
					grad_prev_cell[i_bsize][i_head][i_isize] = agc * fgate[i_bsize][0][i_head][i_isize];
					grad_fgate[i_bsize][0][i_head][i_isize] = agc * init_cell[i_head][i_isize];
				}
			}
		}
	}
	else {
		#pragma omp parallel for collapse(3) schedule(static)
		for (int i_bsize = 0; i_bsize < bsize; i_bsize++) {
			for (int i_head = 0; i_head < nhead; i_head ++) {
				for (int i_isize = 0; i_isize < isize; i_isize++) {
					scalar_t gc = grad_cell[i_bsize][0][i_head][i_isize];
					grad_igh[i_bsize][0][i_head][i_isize] = gc;
					grad_prev_cell[i_bsize][i_head][i_isize] = gc * fgate[i_bsize][0][i_head][i_isize];
					grad_fgate[i_bsize][0][i_head][i_isize] = gc * init_cell[i_head][i_isize];
				}
			}
		}
	}
}

template <typename scalar_t> at::Tensor lgate_forward(torch::Tensor fgate, torch::Tensor igh, torch::Tensor init_cell, torch::Tensor cell, int64_t bsize, int64_t seqlen, int64_t nhead, int64_t isize) {

	omp_lgate_forward_<scalar_t>(fgate.accessor<scalar_t, 4>(), igh.accessor<scalar_t, 4>(), init_cell.accessor<scalar_t, 2>(), cell.accessor<scalar_t, 4>(), (int)bsize, (int)seqlen, (int)nhead, (int)isize);

	return cell;
}

template <typename scalar_t> std::vector<torch::Tensor> lgate_backward(torch::Tensor grad_cell, torch::Tensor cell, torch::Tensor fgate, torch::Tensor init_cell, torch::Tensor grad_fgate, torch::Tensor grad_igh, torch::Tensor grad_prev_cell, int64_t bsize, int64_t seqlen, int64_t nhead, int64_t isize) {

	omp_lgate_backward_<scalar_t>(grad_cell.accessor<scalar_t, 4>(), cell.accessor<scalar_t, 4>(), fgate.accessor<scalar_t, 4>(), init_cell.accessor<scalar_t, 2>(), grad_fgate.accessor<scalar_t, 4>(), grad_igh.accessor<scalar_t, 4>(), grad_prev_cell.accessor<scalar_t, 3>(), (int)bsize, (int)seqlen, (int)nhead, (int)isize);

	return {grad_fgate, grad_igh, grad_prev_cell};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &lgate_forward<float>, "LGate forward");
	m.def("backward", &lgate_backward<float>, "LGate backward");
}
