#include <torch/extension.h>
#include <vector>

at::Tensor id_weightacc_cuda_forward_(torch::Tensor x, torch::Tensor idx, torch::Tensor weight, torch::Tensor rs, int64_t bsize, int64_t isize, int64_t osize, int64_t ncon);

std::vector<torch::Tensor> id_weightacc_cuda_backward_(torch::Tensor x, torch::Tensor idx, torch::Tensor weight, torch::Tensor grad_output, torch::Tensor grad_x, torch::Tensor grad_weight, int64_t bsize, int64_t isize, int64_t osize, int64_t ncon);

at::Tensor id_weightacc_forward_(torch::Tensor x, torch::Tensor idx, torch::Tensor weight, torch::Tensor rs, int64_t bsize, int64_t isize, int64_t osize, int64_t ncon) {

	return id_weightacc_cuda_forward_(x, idx, weight, rs, bsize, isize, osize, ncon);
}

std::vector<torch::Tensor> id_weightacc_backward_(torch::Tensor x, torch::Tensor idx, torch::Tensor weight, torch::Tensor grad_output, torch::Tensor grad_x, torch::Tensor grad_weight, int64_t bsize, int64_t isize, int64_t osize, int64_t ncon) {

	return id_weightacc_cuda_backward_(x, idx, weight, grad_output, grad_x, grad_weight, bsize, isize, osize, ncon);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward_", &id_weightacc_forward_, "IdWeightAcc forward CUDA");
	m.def("backward_", &id_weightacc_backward_, "IdWeightAcc backward CUDA");
}
