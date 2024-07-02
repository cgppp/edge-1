#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include "../../../utils/cpp/hardlimit.h"

// fgate, igh, cell: (bsize, seql, nhead, isize)
// init_cell: (nhead, isize)
template <typename scalar_t> __global__ void cuda_lgate_(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fgate, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> igh, torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> init_cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> cell, int bsize, int seqlen, int nhead, int isize) {

	int num_blocks_x = gridDim.x;
	int num_blocks_y = gridDim.y;
	int num_threads = blockDim.x;
	int block_id_x = blockIdx.x;
	int block_id_y = blockIdx.y;
	int thread_id = threadIdx.x;
	//extern __shared__ scalar_t sm[];
	for (int i_bsize = block_id_x; i_bsize < bsize; i_bsize+=num_blocks_x) {
		for (int i_head = block_id_y; i_head < nhead; i_head+=num_blocks_y) {
			for (int i_isize = thread_id; i_isize < isize; i_isize+=num_threads) {
				scalar_t c = igh[i_bsize][0][i_head][i_isize] + init_cell[i_head][i_isize] * fgate[i_bsize][0][i_head][i_isize];
				cell[i_bsize][0][i_head][i_isize] = c;
				for (int i = 1; i < seqlen; i++) {
					cell[i_bsize][i][i_head][i_isize] = c = c * fgate[i_bsize][i][i_head][i_isize] + igh[i_bsize][i][i_head][i_isize];
				}
			}
		}
	}
}

// grad_cell, cell, fgate, grad_fgate, grad_igh: (bsize, seql, nhead, isize)
// init_cell: (nhead, isize)
// grad_prev_cell: (bsize, nhead, isize)
template <typename scalar_t> __global__ void cuda_lgate_li_grad_(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fgate, torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> init_cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_fgate, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_igh, torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_prev_cell, int bsize, int nhead, int isize, int last_index) {

	int num_blocks_x = gridDim.x;
	int num_blocks_y = gridDim.y;
	int num_threads = blockDim.x;
	int block_id_x = blockIdx.x;
	int block_id_y = blockIdx.y;
	int thread_id = threadIdx.x;
	for (int i_bsize = block_id_x; i_bsize < bsize; i_bsize+=num_blocks_x) {
		for (int i_head = block_id_y; i_head < nhead; i_head+=num_blocks_y) {
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
}

template <typename scalar_t> __global__ void cuda_lgate_nli_grad_(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fgate, torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> init_cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_fgate, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_igh, torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_prev_cell, int bsize, int nhead, int isize) {

	int num_blocks_x = gridDim.x;
	int num_blocks_y = gridDim.y;
	int num_threads = blockDim.x;
	int block_id_x = blockIdx.x;
	int block_id_y = blockIdx.y;
	int thread_id = threadIdx.x;
	for (int i_bsize = block_id_x; i_bsize < bsize; i_bsize+=num_blocks_x) {
		for (int i_head = block_id_y; i_head < nhead; i_head+=num_blocks_y) {
			for (int i_isize = thread_id; i_isize < isize; i_isize+=num_threads) {
				scalar_t gc = grad_cell[i_bsize][0][i_head][i_isize];
				grad_igh[i_bsize][0][i_head][i_isize] = gc;
				grad_prev_cell[i_bsize][i_head][i_isize] = gc * fgate[i_bsize][0][i_head][i_isize];
				grad_fgate[i_bsize][0][i_head][i_isize] = gc * init_cell[i_head][i_isize];
			}
		}
	}
}

template <typename scalar_t> __global__ void cuda_lgate_aul_(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fgate, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> igh, torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> init_cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> cell, int seqlen) {

	int i_bsize = blockIdx.x;
	int i_head = blockIdx.y;
	int i_isize = threadIdx.x;
	//extern __shared__ scalar_t sm[];
	scalar_t c = igh[i_bsize][0][i_head][i_isize] + init_cell[i_head][i_isize] * fgate[i_bsize][0][i_head][i_isize];
	cell[i_bsize][0][i_head][i_isize] = c;
	for (int i = 1; i < seqlen; i++) {
		cell[i_bsize][i][i_head][i_isize] = c = c * fgate[i_bsize][i][i_head][i_isize] + igh[i_bsize][i][i_head][i_isize];
	}
}

template <typename scalar_t> __global__ void cuda_lgate_aul_li_grad_(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fgate, torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> init_cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_fgate, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_igh, torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_prev_cell, int last_index) {

	int i_bsize = blockIdx.x;
	int i_head = blockIdx.y;
	int i_isize = threadIdx.x;
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

template <typename scalar_t> __global__ void cuda_lgate_aul_nli_grad_(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fgate, torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> init_cell, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_fgate, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_igh, torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_prev_cell) {

	int i_bsize = blockIdx.x;
	int i_head = blockIdx.y;
	int i_isize = threadIdx.x;
	scalar_t gc = grad_cell[i_bsize][0][i_head][i_isize];
	grad_igh[i_bsize][0][i_head][i_isize] = gc;
	grad_prev_cell[i_bsize][i_head][i_isize] = gc * fgate[i_bsize][0][i_head][i_isize];
	grad_fgate[i_bsize][0][i_head][i_isize] = gc * init_cell[i_head][i_isize];
}

at::Tensor lgate_cuda_forward(torch::Tensor fgate, torch::Tensor igh, torch::Tensor init_cell, torch::Tensor cell, int bsize, int seqlen, int nhead, int isize) {

	int block_size = isize;
	bool block_ul = (block_size <= MAX_BLOCK_SIZE);
	bool grid_x_ul = (bsize <= MAX_GRID_SIZE_X);
	bool grid_y_ul = (nhead <= MAX_GRID_SIZE_Y);
	if (block_ul && grid_x_ul && grid_y_ul) {
		dim3 grid_size(bsize, nhead);
		AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, igh.scalar_type(), "lgate_aul_forward_cuda", ([&] {cuda_lgate_aul_<scalar_t><<<grid_size, block_size>>>(fgate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), igh.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), init_cell.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), cell.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), seqlen);}));
	}
	else {
			if (!block_ul) {
				block_size /= ((block_size - 1) / MAX_BLOCK_SIZE + 1);
			}
			if (block_size > BLOCK_SIZE_BASE) {
				block_size = block_size / BLOCK_SIZE_BASE * BLOCK_SIZE_BASE;
			}
			int grid_size_x = bsize;
			if (!grid_x_ul) {
				grid_size_x /= ((grid_size_x - 1) / MAX_GRID_SIZE_X + 1);
			}
			int grid_size_y = nhead;
			if (!grid_y_ul) {
				grid_size_y /= ((grid_size_y - 1) / MAX_GRID_SIZE_Y + 1);
			}
			dim3 grid_size(grid_size_x, grid_size_y);

			// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/Dispatch.h
			AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, igh.scalar_type(), "lgate_forward_cuda", ([&] {cuda_lgate_<scalar_t><<<grid_size, block_size>>>(fgate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), igh.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), init_cell.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), cell.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), bsize, seqlen, nhead, isize);}));
	}

	return cell;
}

std::vector<torch::Tensor> lgate_cuda_backward(torch::Tensor grad_cell, torch::Tensor cell, torch::Tensor fgate, torch::Tensor init_cell, torch::Tensor grad_fgate, torch::Tensor grad_igh, torch::Tensor grad_prev_cell, int bsize, int seqlen, int nhead, int isize) {

	int block_size = isize;
	bool block_ul = (block_size <= MAX_BLOCK_SIZE);
	bool grid_x_ul = (bsize <= MAX_GRID_SIZE_X);
	bool grid_y_ul = (nhead <= MAX_GRID_SIZE_Y);
	bool use_li = (seqlen > 1);
	if (block_ul && grid_x_ul && grid_y_ul) {
		dim3 grid_size(bsize, nhead);
		if (use_li) {
			AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_cell.scalar_type(), "lgate_aul_li_backward_cuda", ([&] {cuda_lgate_aul_li_grad_<scalar_t><<<grid_size, block_size>>>(grad_cell.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), cell.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), fgate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), init_cell.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), grad_fgate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), grad_igh.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), grad_prev_cell.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), seqlen - 1);}));
		}
		else {
			AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_cell.scalar_type(), "lgate_aul_nli_backward_cuda", ([&] {cuda_lgate_aul_nli_grad_<scalar_t><<<grid_size, block_size>>>(grad_cell.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), cell.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), fgate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), init_cell.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), grad_fgate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), grad_igh.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), grad_prev_cell.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());}));
		}
	}
	else {
		if (!block_ul) {
			block_size /= ((block_size - 1) / MAX_BLOCK_SIZE + 1);
		}
		if (block_size > BLOCK_SIZE_BASE) {
			block_size = block_size / BLOCK_SIZE_BASE * BLOCK_SIZE_BASE;
		}
		int grid_size_x = bsize;
		if (!grid_x_ul) {
			grid_size_x /= ((grid_size_x - 1) / MAX_GRID_SIZE_X + 1);
		}
		int grid_size_y = nhead;
		if (!grid_y_ul) {
			grid_size_y /= ((grid_size_y - 1) / MAX_GRID_SIZE_Y + 1);
		}
		dim3 grid_size(grid_size_x, grid_size_y);

		if (use_li) {
			AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_cell.scalar_type(), "lgate_li_backward_cuda", ([&] {cuda_lgate_li_grad_<scalar_t><<<grid_size, block_size>>>(grad_cell.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), cell.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), fgate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), init_cell.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), grad_fgate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), grad_igh.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), grad_prev_cell.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), bsize, nhead, isize, seqlen - 1);}));
		}
		else {
			AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_cell.scalar_type(), "lgate_nli_backward_cuda", ([&] {cuda_lgate_nli_grad_<scalar_t><<<grid_size, block_size>>>(grad_cell.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), cell.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), fgate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), init_cell.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), grad_fgate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), grad_igh.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), grad_prev_cell.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), bsize, nhead, isize);}));
		}
	}

	return {grad_fgate, grad_igh, grad_prev_cell};
}
