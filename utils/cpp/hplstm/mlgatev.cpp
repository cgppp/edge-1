#include <torch/extension.h>
#include <vector>

//fgate: (bsize, seql, nheads, isize)
//init_cell: (nheads, isize, nbank)
//bfgate: (bsize, seql, nheads, nbank)
//igh: (bsize, seql, nheads, isize, nbank)
at::Tensor lgate_forward(torch::Tensor fgate, torch::Tensor igh, torch::Tensor init_cell, torch::Tensor bfgate, int64_t dim) {

	auto seqlen = igh.size(dim);
	auto _ufgate = fgate.unsqueeze(-1);//(bsize, seql, nheads, isize, 1)
	auto _ubfgate = bfgate.unsqueeze(-2);//(bsize, seql, nheads, 1, nbank)
	igh.select(dim, 0).addcmul_(init_cell.mul(_ufgate.select(dim, 0)), _ubfgate.select(dim, 0));
	int64_t i;
	for (i = 1; i < seqlen; i++) {
		igh.select(dim, i).addcmul_(igh.select(dim, i - 1).mul(_ufgate.select(dim, i)), _ubfgate.select(dim, i));
	}

	return igh;
}

//grad_cell, cell: (bsize, seql, nheads, isize, nbank)
//fgate: (bsize, seql, nheads, isize)
//init_cell: (nheads, isize, nbank)
//bfgate: (bsize, seql, nheads, nbank)
std::vector<torch::Tensor> lgate_backward(torch::Tensor grad_cell, torch::Tensor cell, torch::Tensor fgate, torch::Tensor init_cell, torch::Tensor bfgate, int64_t dim) {

	auto grad_igh = grad_cell.clone();//(bsize, seql, nheads, isize, nbank)
	auto grad_fgate = fgate.new_empty(fgate.sizes());//(bsize, seql, nheads, isize)
	auto grad_bfgate = bfgate.new_empty(bfgate.sizes());//(bsize, seql, nheads, nbank)
	auto last_index = grad_cell.size(dim) - 1;
	auto acc_grad_cell = grad_cell.select(dim, last_index);//(bsize, nheads, isize, nbank)
	auto _ufgate = fgate.unsqueeze(-1);//(bsize, seql, nheads, isize, 1)
	auto _ubfgate = bfgate.unsqueeze(-2);//(bsize, seql, nheads, 1, nbank)
	auto _mufgate = fgate.unsqueeze(-2);//(bsize, seql, nheads, 1, isize)
	auto _mubfgate = bfgate.unsqueeze(-1);//(bsize, seql, nheads, nbank, 1)
	auto _ugrad_fgate = grad_fgate.unsqueeze(-1);//(bsize, seql, nheads, isize, 1)
	auto _ugrad_bfgate = grad_bfgate.unsqueeze(-2);//(bsize, seql, nheads, 1, nbank)
	//auto _cufgate = _ufgate.select(dim, last_index);//(bsize, nheads, isize, 1)
	//auto _cubfgate = _ubfgate.select(dim, last_index);//(bsize, nheads, 1, nbank)
	auto grad_prev_cell = acc_grad_cell.mul(_ufgate.select(dim, last_index)).mul_(_ubfgate.select(dim, last_index));//(bsize, nheads, isize, nbank)
	if (last_index > 0) {
		auto prev_i = last_index - 1;
		auto grad_rfgate = acc_grad_cell.mul(cell.select(dim, prev_i));//(bsize, nheads, isize, nbank)
		//at::sum_outf(grad_rfgate.mul(_cubfgate), -1, false, grad_fgate.select(dim, last_index));//(bsize, nheads, isize, nbank) * (bsize, nheads, 1, nbank) -> (bsize, nheads, isize)
		at::matmul_outf(grad_rfgate, _mubfgate.select(dim, last_index), _ugrad_fgate.select(dim, last_index));//(bsize, nheads, isize, nbank) * (bsize, nheads, nbank, 1) -> (bsize, nheads, isize)
		//at::sum_outf(grad_rfgate.mul(_cufgate), -2, false, grad_bfgate.select(dim, last_index));//(bsize, nheads, isize, nbank) * (bsize, nheads, isize, 1) -> (bsize, nheads, nbank)
		at::matmul_outf(_mufgate.select(dim, last_index), grad_rfgate, _ugrad_bfgate.select(dim, last_index));//(bsize, nheads, 1, isize) * (bsize, nheads, isize, nbank) -> (bsize, nheads, nbank)
		int64_t i;
		for (i = prev_i; i > 0;) {
			at::add_outf(grad_cell.select(dim, i), grad_prev_cell, 1.0, grad_rfgate);//grad_rfgate is now used as acc_grad_cell, (bsize, nheads, isize, nbank)
			grad_igh.select(dim, i).add_(grad_prev_cell);
			//_cufgate = _ufgate.select(dim, i);
			//_cubfgate = _ubfgate.select(dim, i);
			//grad_prev_cell = grad_rfgate.mul(_ufgate.select(dim, i)).mul_(_ubfgate.select(dim, i));
			at::mul_outf(grad_rfgate, _ufgate.select(dim, i), grad_prev_cell).mul_(_ubfgate.select(dim, i));
			prev_i = i - 1;
			grad_rfgate.mul_(cell.select(dim, prev_i));
			//at::sum_outf(grad_rfgate.mul(_cubfgate), -1, false, grad_fgate.select(dim, i));
			at::matmul_outf(grad_rfgate, _mubfgate.select(dim, i), _ugrad_fgate.select(dim, i));
			//at::sum_outf(grad_rfgate.mul(_cufgate), -2, false, grad_bfgate.select(dim, i));
			at::matmul_outf(_mufgate.select(dim, i), grad_rfgate, _ugrad_bfgate.select(dim, i));
			i = prev_i;
		}
		//acc_grad_cell = grad_fgate.select(dim, 0).add_(grad_prev_cell);
		at::add_outf(grad_cell.select(dim, 0), grad_prev_cell, 1.0, grad_rfgate);
		grad_igh.select(dim, 0).add_(grad_prev_cell);
		//_cufgate = _ufgate.select(dim, 0);
		//_cubfgate = _ubfgate.select(dim, 0);
		//grad_prev_cell = grad_rfgate.mul(_cufgate).mul(_cubfgate);
		at::mul_outf(grad_rfgate, _ufgate.select(dim, 0), grad_prev_cell).mul_(_ubfgate.select(dim, 0));
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
