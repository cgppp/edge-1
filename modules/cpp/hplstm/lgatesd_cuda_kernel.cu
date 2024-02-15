#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define MAX_GRID_SIZE 2147483647
#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE_BASE 32
#define MAX_Share_MEM 49152

#define MAX_ISIZE_Share_MEM (MAX_Share_MEM / 4 )

// observe serious performance degradation compared to lgates, unknown reason

// fgate, igh, cell: (bsize, seql, nhead, isize)
// init_cell: (nhead, isize)
template <typename scalar_t> __global__ void cuda_lgate_(scalar_t* __restrict__ fgate, scalar_t* __restrict__ igh, scalar_t* __restrict__ init_cell, scalar_t* __restrict__ cell, int bsize, int seqlen, int nhead, int isize, int eid, int seq_shift) {

	int num_blocks = gridDim.x;
	int num_threads = blockDim.x;
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	//extern __shared__ scalar_t sm[];
	for (int _i = block_id; _i < eid; _i+=num_blocks) {
		int i_head = _i / bsize;
		int i_bsize = _i % bsize;
		for (int i_isize = thread_id; i_isize < isize; i_isize+=num_threads) {
			int _h_shift = i_head * isize;
			int _i_base = i_bsize * seqlen * seq_shift + _h_shift + i_isize;
			scalar_t c = igh[_i_base] + init_cell[_h_shift + i_isize] * fgate[_i_base];
			cell[_i_base] = c;
			for (int i = 1; i < seqlen; i++) {
				_i_base += seq_shift;
				cell[_i_base] = c = c * fgate[_i_base] + igh[_i_base];
			}
		}
	}
}

// grad_cell, cell, fgate, grad_fgate, grad_igh: (bsize, seql, nhead, isize)
// init_cell: (nhead, isize)
// grad_prev_cell: (bsize, nhead, isize)
template <typename scalar_t> __global__ void cuda_lgate_grad_(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_cell, scalar_t* __restrict__ cell, scalar_t* __restrict__ fgate, scalar_t* __restrict__ init_cell, scalar_t* __restrict__ grad_fgate, scalar_t* __restrict__ grad_igh, scalar_t* __restrict__ grad_prev_cell, int bsize, int seqlen, int nhead, int isize, int eid, int seq_shift, int last_index) {

	int num_blocks = gridDim.x;
	int num_threads = blockDim.x;
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	if (last_index > 0) {
		for (int _i = block_id; _i < eid; _i+=num_blocks) {
			int i_head = _i / bsize;
			int i_bsize = _i % bsize;
			for (int i_isize = thread_id; i_isize < isize; i_isize+=num_threads) {
				int _bhi_base = i_bsize * seq_shift;
				int _h_base = i_head * isize;
				int _i_base = _bhi_base * seqlen + last_index * seq_shift + _h_base + i_isize;
				scalar_t agc = grad_cell[i_bsize][last_index][i_head][i_isize];
				for (int i = last_index; i > 0; i--) {
					grad_igh[_i_base] = agc;
					int _i_base_new = _i_base - seq_shift;
					grad_fgate[_i_base] = agc * cell[_i_base_new];
					agc = agc * fgate[_i_base] + grad_cell[i_bsize][i - 1][i_head][i_isize];
					_i_base = _i_base_new;
				}
				grad_igh[_i_base] = agc;
				grad_prev_cell[_bhi_base + _h_base + i_isize] = agc * fgate[_i_base];
				grad_fgate[_i_base] = agc * init_cell[_h_base + i_isize];
			}
		}
	}
	else {
		for (int _i = block_id; _i < eid; _i+=num_blocks) {
			int i_head = _i / bsize;
			int i_bsize = _i % bsize;
			for (int i_isize = thread_id; i_isize < isize; i_isize+=num_threads) {
				int _bhi_base = i_bsize * seq_shift;
				int _h_base = i_head * isize;
				int _i_base = _bhi_base + _h_base + i_isize;// * seqlen equals to 1; + last_index * _seq_shift (last_index is 0)
				scalar_t gc = grad_cell[i_bsize][0][i_head][i_isize];
				grad_igh[_i_base] = gc;
				grad_prev_cell[_bhi_base + _h_base + i_isize] = gc * fgate[_i_base];
				grad_fgate[_i_base] = gc * init_cell[_h_base + i_isize];
			}
		}
	}
}

at::Tensor lgate_cuda_forward(torch::Tensor fgate, torch::Tensor igh, torch::Tensor init_cell, torch::Tensor cell, int bsize, int seqlen, int nhead, int isize) {

	int block_size = isize;
	if (block_size > MAX_BLOCK_SIZE) {
		block_size /= ((isize - 1) / MAX_BLOCK_SIZE + 1);
	}
	if (block_size > BLOCK_SIZE_BASE) {
		block_size = block_size / BLOCK_SIZE_BASE * BLOCK_SIZE_BASE;
	}
	int grid_size = nhead * bsize;
	if (grid_size >= MAX_GRID_SIZE) {
		grid_size /= ((grid_size - 1) / MAX_GRID_SIZE + 1);
	}

	// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/Dispatch.h
	AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, igh.scalar_type(), "lgate_forward_cuda", ([&] {cuda_lgate_<scalar_t><<<grid_size, block_size>>>(fgate.data_ptr<scalar_t>(), igh.data_ptr<scalar_t>(), init_cell.data_ptr<scalar_t>(), cell.data_ptr<scalar_t>(), bsize, seqlen, nhead, isize, nhead * bsize, nhead * isize);}));

	return cell;
}

std::vector<torch::Tensor> lgate_cuda_backward(torch::Tensor grad_cell, torch::Tensor cell, torch::Tensor fgate, torch::Tensor init_cell, torch::Tensor grad_fgate, torch::Tensor grad_igh, torch::Tensor grad_prev_cell, int bsize, int seqlen, int nhead, int isize) {

	int block_size = isize;
	if (block_size > MAX_BLOCK_SIZE) {
		block_size /= ((isize - 1) / MAX_BLOCK_SIZE + 1);
	}
	if (block_size > BLOCK_SIZE_BASE) {
		block_size = block_size / BLOCK_SIZE_BASE * BLOCK_SIZE_BASE;
	}
	int grid_size = nhead * bsize;
	if (grid_size >= MAX_GRID_SIZE) {
		grid_size /= ((grid_size - 1) / MAX_GRID_SIZE + 1);
	}

	AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_cell.scalar_type(), "lgate_backward_cuda", ([&] {cuda_lgate_grad_<scalar_t><<<grid_size, block_size>>>(grad_cell.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), cell.data_ptr<scalar_t>(), fgate.data_ptr<scalar_t>(), init_cell.data_ptr<scalar_t>(), grad_fgate.data_ptr<scalar_t>(), grad_igh.data_ptr<scalar_t>(), grad_prev_cell.data_ptr<scalar_t>(), bsize, seqlen, nhead, isize, nhead * bsize, nhead * isize, seqlen - 1);}));

	return {grad_fgate, grad_igh, grad_prev_cell};
}
