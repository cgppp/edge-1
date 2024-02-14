#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define MAX_GRID_SIZE 2147483647
#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE_BASE 32
#define MAX_Share_MEM 49152

#define MAX_ISIZE_Share_MEM (MAX_Share_MEM / 4 )

// fgate, igh, cell: (bsize, seql, nhead, isize)
// init_cell: (nhead, isize)
template <typename scalar_t> __global__ void cuda_lgate_(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fgate, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> igh, torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> init_cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> cell, int bsize, int seqlen, int nhead, int isize, int eid) {

	int num_blocks = gridDim.x;
	int num_threads = blockDim.x;
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	//extern __shared__ scalar_t sm[];
	for (int _i = block_id; _i < eid; _i+=num_blocks) {
		int i_head = _i / bsize;
		int i_bsize = _i % bsize;
		for (int i_isize = thread_id; i_isize < isize; i_isize+=num_threads) {
			scalar_t c = igh[i_bsize][0][i_head][i_isize] + init_cell[i_head][i_isize] * fgate[i_bsize][0][i_head][i_isize];
			cell[i_bsize][0][i_head][i_isize] = c;
			for (int i = 1; i < seqlen; i++) {
				cell[i_bsize][i][i_head][i_isize] = c = c * fgate[i_bsize][i][i_head][i_isize] + igh[i_bsize][i][i_head][i_isize];
			}
		}
	}
}

// grad_cell, cell, fgate, grad_fgate, grad_igh: (bsize, seql, nhead, isize)
// init_cell: (nhead, isize)
// grad_prev_cell: (bsize, nhead, isize)
template <typename scalar_t> __global__ void cuda_lgate_grad_(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fgate, torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> init_cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_fgate, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_igh, torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_prev_cell, int bsize, int seqlen, int nhead, int isize, int eid, int last_index) {

	int num_blocks = gridDim.x;
	int num_threads = blockDim.x;
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	if (last_index > 0) {
		for (int _i = block_id; _i < eid; _i+=num_blocks) {
			int i_head = _i / bsize;
			int i_bsize = _i % bsize;
			for (int i_isize = thread_id; i_isize < isize; i_isize+=num_threads) {
				int _prev_i;
				scalar_t agc = grad_cell[i_bsize][last_index][i_head][i_isize];
				for (int i = last_index; i > 0; ) {
					grad_igh[i_bsize][i][i_head][i_isize] = agc;
					_prev_i = i - 1;
					grad_fgate[i_bsize][i][i_head][i_isize] = agc * cell[i_bsize][_prev_i][i_head][i_isize];
					agc = agc * fgate[i_bsize][i][i_head][i_isize] + grad_cell[i_bsize][_prev_i][i_head][i_isize];
					i = _prev_i;
				}
				grad_igh[i_bsize][0][i_head][i_isize] = agc;
				grad_prev_cell[i_bsize][i_head][i_isize] = agc * fgate[i_bsize][0][i_head][i_isize];
				grad_fgate[i_bsize][0][i_head][i_isize] = agc * init_cell[i_head][i_isize];
			}
		}
	}
	else {
		for (int _i = block_id; _i < eid; _i+=num_blocks) {
			int i_head = _i / bsize;
			int i_bsize = _i % bsize;
			for (int i_isize = thread_id; i_isize < isize; i_isize+=num_threads) {
				scalar_t gc = grad_cell[i_bsize][0][i_head][i_isize];
				grad_igh[i_bsize][0][i_head][i_isize] = gc;
				grad_prev_cell[i_bsize][i_head][i_isize] = gc * fgate[i_bsize][0][i_head][i_isize];
				grad_fgate[i_bsize][0][i_head][i_isize] = gc * init_cell[i_head][i_isize];
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
	AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, igh.scalar_type(), "lgate_forward_cuda", ([&] {cuda_lgate_<scalar_t><<<grid_size, block_size>>>(fgate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), igh.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), init_cell.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), cell.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), bsize, seqlen, nhead, isize, nhead * bsize);}));

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

	AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_cell.scalar_type(), "lgate_backward_cuda", ([&] {cuda_lgate_grad_<scalar_t><<<grid_size, block_size>>>(grad_cell.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), cell.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), fgate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), init_cell.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), grad_fgate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), grad_igh.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), grad_prev_cell.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), bsize, seqlen, nhead, isize, nhead * bsize, seqlen - 1);}));

	return {grad_fgate, grad_igh, grad_prev_cell};
}
