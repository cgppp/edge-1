#include <torch/extension.h>
#include <vector>
#include "omp.h"

// x: (bsize, isize)
// rs: (bsize, osize)
// idx, weight: (osize, ncon)
inline void omp_id_weight_acc_(float *x, int *idx, float *weight, float *rs, int bsize, int isize, int osize, int ncon) {

	#pragma omp parallel for
	for (int i_osize = 0; i_osize < osize; i_osize++) {
		int osize_ncon_base = i_osize * ncon;
		int idx_cache[ncon];
		float weight_cache[ncon];
		for (int i_con = 0; i_con < ncon; i_con++) {
			int id_weight_offset = osize_ncon_base + i_con;
			idx_cache[i_con] = idx[id_weight_offset];
			weight_cache[i_con] = weight[id_weight_offset];
		}
		//#pragma omp parallel for
		for (int i_bsize = 0; i_bsize < bsize; i_bsize++) {
			float _dp = 0.0;
			int x_base = i_bsize * isize;
			for (int i_con = 0; i_con < ncon; i_con++) {
				_dp += x[x_base + idx_cache[i_con]] * weight_cache[i_con];
			}
			rs[i_bsize * osize + i_osize] = _dp;
		}
	}
}

// x, grad_x: (bsize, isize)
// grad_output: (bsize, osize)
// idx, weight, grad_weight: (osize, ncon)
inline void omp_id_weight_acc_grad_(float *x, int *idx, float *weight, torch::TensorAccessor<float, 2> grad_output, float *grad_x, float *grad_weight, int bsize, int isize, int osize, int ncon) {

	#pragma omp parallel for
	for (int i_osize = 0; i_osize < osize; i_osize++) {
		int osize_ncon_base = i_osize * ncon;
		int idx_cache[ncon];
		for (int i_con = 0; i_con < ncon; i_con++) {
			idx_cache[i_con] = idx[osize_ncon_base + i_con];
		}
		//#pragma omp parallel for
		for (int i_bsize = 0; i_bsize < bsize; i_bsize++) {
			int x_base = i_bsize * isize;
			float grad_weight_scalar = grad_output[i_bsize][i_osize];//[i_bsize * osize + i_osize]
			for (int i_con = 0; i_con < ncon; i_con++) {
				int x_ind = x_base + idx_cache[i_con];
				int weight_ind = osize_ncon_base + i_con;
				grad_x[x_ind] += weight[weight_ind] * grad_weight_scalar;
				grad_weight[weight_ind] += x[x_ind] * grad_weight_scalar;
			}
		}
	}
}

at::Tensor id_weightacc_forward_(torch::Tensor x, torch::Tensor idx, torch::Tensor weight, torch::Tensor rs, int64_t bsize, int64_t isize, int64_t osize, int64_t ncon) {

	omp_id_weight_acc_(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), rs.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon);

	return rs;
}

std::vector<torch::Tensor> id_weightacc_backward_(torch::Tensor x, torch::Tensor idx, torch::Tensor weight, torch::Tensor grad_output, torch::Tensor grad_x, torch::Tensor grad_weight, int64_t bsize, int64_t isize, int64_t osize, int64_t ncon) {

	omp_id_weight_acc_grad_(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), grad_output.accessor<float, 2>(), grad_x.data_ptr<float>(), grad_weight.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon);

	return {grad_x, grad_weight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward_", &id_weightacc_forward_, "IdWeightAcc forward CPU");
	m.def("backward_", &id_weightacc_backward_, "IdWeightAcc backward CPU");
}
