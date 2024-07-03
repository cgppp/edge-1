#include <torch/extension.h>

// xclone is expected to be (a clone of) x
at::Tensor mvavg_forward(torch::Tensor x, torch::Tensor xclone, int64_t dim, float beta=0.9) {

	float mbeta = 1.0 - beta;
	auto prev_step = xclone.select(dim, 0).mul_(mbeta);
	auto seqlen = x.size(dim);
	int64_t i;
	for (i = 1; i < seqlen; i++) {
		prev_step = xclone.select(dim, i).mul_(mbeta).add_(prev_step, beta);
	}

	return xclone;
}

// the backward function is identical to ``mvavg_backward'' in ``utils/cpp/mvavg.cpp''
torch::Tensor mvavg_backward(torch::Tensor grad_out, int64_t dim, float beta=0.9) {

	float mbeta = 1.0 - beta;
	auto grad_input = grad_out.clone();
	auto last_index = grad_out.size(dim) - 1;
	grad_input.select(dim, last_index).mul_(mbeta);
	if (last_index > 0) {
		auto grad_prev_out = grad_out.select(dim, last_index) * beta;
		int64_t i;
		for (i = last_index - 1; i > 0; i--) {
			auto grad_step = grad_input.select(dim, i).add_(grad_prev_out);// grad_input is initialized as a copy of grad_out, performing the accumulation directly on grad_input is more efficient.
			grad_prev_out = grad_step * beta;
			grad_step.mul_(mbeta);
		}
		grad_input.select(dim, 0).add_(grad_prev_out).mul_(mbeta);
	}

	return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &mvavg_forward, "MovAvg forward");
	m.def("backward", &mvavg_backward, "MovAvg backward");
}
