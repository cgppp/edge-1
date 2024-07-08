#include <torch/extension.h>
#include <vector>

at::Tensor lgate_cuda_forward(torch::Tensor fgate, torch::Tensor igh, torch::Tensor init_cell, torch::Tensor cell, int bsize, int seqlen, int nhead, int isize);
std::vector<torch::Tensor> lgate_cuda_backward(torch::Tensor grad_cell, torch::Tensor cell, torch::Tensor fgate, torch::Tensor init_cell, torch::Tensor grad_fgate, torch::Tensor grad_igh, torch::Tensor grad_prev_cell, int bsize, int seqlen, int nhead, int isize);

at::Tensor lgate_forward_cuda(torch::Tensor fgate, torch::Tensor igh, torch::Tensor init_cell, torch::Tensor cell, int64_t bsize, int64_t seqlen, int64_t nhead, int64_t isize) {

	return lgate_cuda_forward(fgate, igh, init_cell, cell, (int)bsize, (int)seqlen, (int)nhead, (int)isize);
}

std::vector<torch::Tensor> lgate_backward_cuda(torch::Tensor grad_cell, torch::Tensor cell, torch::Tensor fgate, torch::Tensor init_cell, torch::Tensor grad_fgate, torch::Tensor grad_igh, torch::Tensor grad_prev_cell, int64_t bsize, int64_t seqlen, int64_t nhead, int64_t isize) {

	return lgate_cuda_backward(grad_cell, cell, fgate, init_cell, grad_fgate, grad_igh, grad_prev_cell, (int)bsize, (int)seqlen, (int)nhead, (int)isize);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &lgate_forward_cuda, "LGate forward CUDA");
	m.def("backward", &lgate_backward_cuda, "LGate backward CUDA");
}
