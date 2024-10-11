#include <torch/extension.h>
#include <vector>

//fgate: (bsize, seql, nheads, isize)
//init_cell: (nheads, isize, nbank)
//bfgate: (bsize, seql, nheads, nbank)
//igh: (bsize, seql, nheads, isize, nbank)
at::Tensor lgate_forward(torch::Tensor fgate, torch::Tensor igh, torch::Tensor init_cell, torch::Tensor bfgate, int64_t dim) {

	auto seqlen = igh.size(dim);
	auto _ufgate = fgate.unsqueeze(-1);
	auto _ubfgate = bfgate.unsqueeze(-2);
	igh.select(dim, 0).addcmul_(init_cell, _ufgate.select(dim, 0).mul(_ubfgate.select(dim, 0)));
	int64_t i;
	for (i = 1; i < seqlen; i++) {
		igh.select(dim, i).addcmul_(igh.select(dim, i - 1), _ufgate.select(dim, i).mul(_ubfgate.select(dim, i)));
	}

	return igh;
}

std::vector<torch::Tensor> lgate_backward(torch::Tensor grad_cell, torch::Tensor cell, torch::Tensor fgate, torch::Tensor init_cell, torch::Tensor bfgate, int64_t dim) {

	auto grad_igh = grad_cell.clone();
	auto grad_fgate = fgate.new_empty(fgate.sizes());
	auto grad_bfgate = bfgate.new_empty(fgate.sizes());
	auto last_index = grad_cell.size(dim) - 1;
	auto acc_grad_cell = grad_cell.select(dim, last_index);
	auto _ufgate = fgate.unsqueeze(-1);
	auto _ubfgate = bfgate.unsqueeze(-2);
	auto _cufgate =_ufgate.select(dim, last_index);
	auto _cubfgate = _ubfgate.select(dim, last_index);
	auto _rfgate = _cufgate.mul(_cubfgate);
	auto grad_prev_cell = acc_grad_cell * _rfgate;
	if (last_index > 0) {
		auto prev_i = last_index - 1;
		auto grad_rfgate = grad_cell.select(dim, last_index).mul(cell.select(dim, prev_i));
		at::sum_outf(grad_rfgate.mul(_cubfgate), -1, false, grad_fgate.select(dim, last_index));
		at::sum_outf(grad_rfgate.mul(_cufgate), -2, false, grad_bfgate.select(dim, last_index));
		int64_t i;
		for (i = prev_i; i > 0;) {
			at::add_outf(grad_cell.select(dim, i), grad_prev_cell, 1.0, grad_rfgate);//grad_rfgate is now used as acc_grad_cell
			grad_igh.select(dim, i).add_(grad_prev_cell);
			_cufgate =_ufgate.select(dim, i);
			_cubfgate = _ubfgate.select(dim, i);
			at::mul_outf(_cufgate, _cubfgate, _rfgate);
			grad_prev_cell = grad_rfgate * _rfgate;
			prev_i = i - 1;
			grad_rfgate.mul_(cell.select(dim, prev_i));
			// (bsize, nheads, isize, nbank) * (bsize, nheads, 1, nbank) -> (bsize, nheads, isize)
			at::sum_outf(grad_rfgate.mul(_cubfgate), -1, false, grad_fgate.select(dim, i));
			//einsum or at::matmul_outf(const at::Tensor &self, const at::Tensor &other, at::Tensor &out)?
			at::sum_outf(grad_rfgate.mul(_cufgate), -2, false, grad_bfgate.select(dim, i));
			i = prev_i;
		}
		//acc_grad_cell = grad_fgate.select(dim, 0).add_(grad_prev_cell);
		at::add_outf(grad_cell.select(dim, 0), grad_prev_cell, 1.0, grad_rfgate);
		grad_igh.select(dim, 0).add_(grad_prev_cell);
		_cufgate =_ufgate.select(dim, 0);
		_cubfgate = _ubfgate.select(dim, 0);
		at::mul_outf(_cufgate, _cubfgate, _rfgate);
		grad_prev_cell = grad_rfgate * _rfgate;
		grad_rfgate.mul_(init_cell);
		at::sum_outf(grad_rfgate.mul(_cubfgate), -1, false, grad_fgate.select(dim, 0));
		at::sum_outf(grad_rfgate.mul(_cufgate), -2, false, grad_bfgate.select(dim, 0));
	}
	else {
		auto grad_rfgate = grad_cell.select(dim, 0).mul(init_cell);
		at::sum_outf(grad_rfgate.mul(_cubfgate), -1, false, grad_fgate.select(dim, last_index));
		at::sum_outf(grad_rfgate.mul(_cufgate), -2, false, grad_bfgate.select(dim, last_index));
	}

	return {grad_fgate, grad_igh, grad_prev_cell};
}

std::vector<torch::Tensor> lgate_backward_no_fgate(torch::Tensor grad_cell, torch::Tensor fgate, torch::Tensor bfgate, int64_t dim) {

	auto grad_igh = grad_cell.clone();
	auto last_index = grad_cell.size(dim) - 1;
	auto _ufgate = fgate.unsqueeze(-1);
	auto _ubfgate = bfgate.unsqueeze(-2);
	auto _cufgate =_ufgate.select(dim, last_index);
	auto _cubfgate = _ubfgate.select(dim, last_index);
	auto _rfgate = _cufgate.mul(_cubfgate);
	auto grad_prev_cell = grad_cell.select(dim, last_index) * _rfgate;
	int64_t i;
	for (i = last_index - 1; i >= 0; i--) {
		_cufgate =_ufgate.select(dim, i);
		_cubfgate = _ubfgate.select(dim, i);
		at::mul_outf(_cufgate, _cubfgate, _rfgate);
		grad_prev_cell = grad_igh.select(dim, i).add_(grad_prev_cell) * _rfgate;
	}

	return {grad_igh, grad_prev_cell};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &lgate_forward, "MemLGate forward");
	m.def("backward", &lgate_backward, "MemLGate backward");
	m.def("backward_no_fgate", &lgate_backward_no_fgate, "MemLGate backward (no fgate)");
}
