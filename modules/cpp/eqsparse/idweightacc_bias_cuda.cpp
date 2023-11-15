#include <torch/extension.h>

at::Tensor id_weightacc_bias_cuda_forward_(torch::Tensor x, torch::Tensor idx, torch::Tensor weight, torch::Tensor bias, torch::Tensor rs, int64_t bsize, int64_t isize, int64_t osize, int64_t ncon);

at::Tensor id_weightacc_bias_forward_(torch::Tensor x, torch::Tensor idx, torch::Tensor weight, torch::Tensor bias, torch::Tensor rs, int64_t bsize, int64_t isize, int64_t osize, int64_t ncon) {

	return id_weightacc_bias_cuda_forward_(x, idx, weight, bias, rs, bsize, isize, osize, ncon);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward_", &id_weightacc_bias_forward_, "IdWeightAccBias forward CUDA");
}
