#include <torch/extension.h>

at::Tensor cumsum_cuda_forward(torch::Tensor x, torch::Tensor o, int bsize, int seqlen, int nhead, int isize);
torch::Tensor cumsum_cuda_backward(torch::Tensor grad_o, torch::Tensor grad_x, int bsize, int seqlen, int nhead, int isize);

at::Tensor cumsum_forward_cuda(torch::Tensor x, torch::Tensor o, int64_t bsize, int64_t seqlen, int64_t nhead, int64_t isize) {

	return cumsum_cuda_forward(x, o, (int)bsize, (int)seqlen, (int)nhead, (int)isize);
}

torch::Tensor cumsum_backward_cuda(torch::Tensor grad_o, torch::Tensor grad_x, int64_t bsize, int64_t seqlen, int64_t nhead, int64_t isize) {

	return cumsum_cuda_backward(grad_o, grad_x, (int)bsize, (int)seqlen, (int)nhead, (int)isize);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &cumsum_forward_cuda, "cumsum forward CUDA");
	m.def("backward", &cumsum_backward_cuda, "cumsum backward CUDA");
}
