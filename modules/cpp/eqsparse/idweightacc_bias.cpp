#include <torch/extension.h>
#include <vector>
#include "omp.h"

// x: (bsize, isize)
// rs: (bsize, osize)
// idx, weight: (osize, ncon)
// bias: (osize,)
inline void omp_id_weight_bias_acc_(float *x, int *idx, float *weight, float *bias, float *rs, int bsize, int isize, int osize, int ncon) {

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
		float bias_base = bias[i_osize];
		for (int i_bsize = 0; i_bsize < bsize; i_bsize++) {
			float _dp = bias_base;
			int x_base = i_bsize * isize;
			for (int i_con = 0; i_con < ncon; i_con++) {
				_dp += x[x_base + idx_cache[i_con]] * weight_cache[i_con];
			}
			rs[i_bsize * osize + i_osize] = _dp;
		}
	}
}

at::Tensor id_weightacc_bias_forward_(torch::Tensor x, torch::Tensor idx, torch::Tensor weight, torch::Tensor bias, torch::Tensor rs, int64_t bsize, int64_t isize, int64_t osize, int64_t ncon) {

	omp_id_weight_bias_acc_(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), bias.data_ptr<float>(), rs.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon);

	return rs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward_", &id_weightacc_bias_forward_, "IdWeightAccBias forward CPU");
}
