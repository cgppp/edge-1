#include <torch/extension.h>

at::Tensor mvavg_forward(torch::Tensor x, int64_t dim, float beta=0.9) {

	float mbeta = 1.0 - beta;
	auto prev_step = x.select(dim, 0);
	auto seqlen = x.size(dim);
	int64_t i;
	for (i = 1; i < seqlen; i++) {
		prev_step = x.select(dim, i).mul_(mbeta).add_(prev_step, beta);
	}

	return x;
}

torch::Tensor mvavg_backward(torch::Tensor grad_out, int64_t dim, float beta=0.9) {

	float mbeta = 1.0 - beta;
	auto last_index = grad_out.size(dim) - 1;
	if (last_index > 0) {
		auto grad_prev_out = grad_out.select(dim, last_index) * beta;
		grad_out.select(dim, last_index).mul_(mbeta);
		int64_t i;
		for (i = last_index - 1; i > 0; i--) {
			auto grad_step = grad_out.select(dim, i).add_(grad_prev_out);// grad_input is initialized as a copy of grad_out, performing the accumulation directly on grad_input is more efficient.
			grad_prev_out = grad_step * beta;
			grad_step.mul_(mbeta);
		}
		grad_out.select(dim, 0).add_(grad_prev_out);
	}

	return grad_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &mvavg_forward, "MovAvg forward");
	m.def("backward", &mvavg_backward, "MovAvg backward");
}
