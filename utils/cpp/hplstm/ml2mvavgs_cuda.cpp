#include <torch/extension.h>

at::Tensor mvavg_cuda_forward(torch::Tensor x, torch::Tensor o, float beta, int bsize, int seqlen, int nhead, int isize);
torch::Tensor mvavg_cuda_backward(torch::Tensor grad_o, torch::Tensor grad_x, float beta, int bsize, int seqlen, int nhead, int isize);

at::Tensor mvavg_forward_cuda(torch::Tensor x, torch::Tensor o, float beta, int64_t bsize, int64_t seqlen, int64_t nhead, int64_t isize) {

	return mvavg_cuda_forward(x, o, beta, (int)bsize, (int)seqlen, (int)nhead, (int)isize);
}

torch::Tensor mvavg_backward_cuda(torch::Tensor grad_o, torch::Tensor grad_x, float beta, int64_t bsize, int64_t seqlen, int64_t nhead, int64_t isize) {

	return mvavg_cuda_backward(grad_o, grad_x, beta, (int)bsize, (int)seqlen, (int)nhead, (int)isize);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &mvavg_forward_cuda, "MvAvg forward CUDA");
	m.def("backward", &mvavg_backward_cuda, "MvAvg backward CUDA");
}
