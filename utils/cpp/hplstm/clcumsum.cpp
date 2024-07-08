#include <torch/extension.h>

at::Tensor cumsum_forward(torch::Tensor x, int64_t dim) {

	auto prev_step = x.select(dim, 0);
	auto seqlen = x.size(dim);
	int64_t i;
	for (i = 1; i < seqlen; i++) {
		prev_step = x.select(dim, i).add_(prev_step);
	}

	return x;
}

torch::Tensor cumsum_backward(torch::Tensor grad_out, int64_t dim) {

	auto last_index = grad_out.size(dim) - 1;
	if (last_index > 0) {
		auto grad_prev_out = grad_out.select(dim, last_index);
		int64_t i;
		for (i = last_index - 1; i >= 0; i--) {
			grad_prev_out = grad_out.select(dim, i).add_(grad_prev_out);// grad_input is initialized as a copy of grad_out, performing the accumulation directly on grad_input is more efficient.
		}
	}

	return grad_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &cumsum_forward, "cumsum forward");
	m.def("backward", &cumsum_backward, "cumsum backward");
}
