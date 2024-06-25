#include <torch/extension.h>
#include <cuda_runtime.h>
#include "utils/cpp/hardlimit.h"

// x, o: (bsize, seql, nhead, isize)
template <typename scalar_t> __global__ void cuda_mvavg_(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> x, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> o, scalar_t beta, scalar_t mbeta, int bsize, int seqlen, int nhead, int isize) {

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
				scalar_t tmp = x[i_bsize][0][i_head][i_isize] * mbeta;
				o[i_bsize][0][i_head][i_isize] = tmp;
				for (int i = 1; i < seqlen; i++) {
					o[i_bsize][i][i_head][i_isize] = tmp = tmp * beta + x[i_bsize][i][i_head][i_isize] * mbeta;
				}
			}
		}
	}
}

// grad_o, x, grad_x: (bsize, seql, nhead, isize)
template <typename scalar_t> __global__ void cuda_mvavg_grad_(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_o, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_x, scalar_t beta, scalar_t mbeta, int bsize, int nhead, int isize, int last_index) {

	int num_blocks_x = gridDim.x;
	int num_blocks_y = gridDim.y;
	int num_threads = blockDim.x;
	int block_id_x = blockIdx.x;
	int block_id_y = blockIdx.y;
	int thread_id = threadIdx.x;
	for (int i_bsize = block_id_x; i_bsize < bsize; i_bsize+=num_blocks_x) {
		for (int i_head = block_id_y; i_head < nhead; i_head+=num_blocks_y) {
			for (int i_isize = thread_id; i_isize < isize; i_isize+=num_threads) {
				scalar_t ago = grad_o[i_bsize][last_index][i_head][i_isize];
				grad_x[i_bsize][last_index][i_head][i_isize] = ago * mbeta;
				for (int i = last_index - 1; i >= 0; i--) {
					ago = ago * beta + grad_o[i_bsize][i][i_head][i_isize];
					grad_x[i_bsize][i][i_head][i_isize] = ago * mbeta;
				}
			}
		}
	}
}

template <typename scalar_t> __global__ void cuda_mvavg_aul_(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> x, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> o, scalar_t beta, scalar_t mbeta, int seqlen) {

	int i_bsize = blockIdx.x;
	int i_head = blockIdx.y;
	int i_isize = threadIdx.x;
	//extern __shared__ scalar_t sm[];
	scalar_t tmp = x[i_bsize][0][i_head][i_isize] * mbeta;
	o[i_bsize][0][i_head][i_isize] = tmp;
	for (int i = 1; i < seqlen; i++) {
		o[i_bsize][i][i_head][i_isize] = tmp = tmp * beta + x[i_bsize][i][i_head][i_isize] * mbeta;
	}
}

template <typename scalar_t> __global__ void cuda_mvavg_aul_grad_(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_o, torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_x, scalar_t beta, scalar_t mbeta, int last_index) {

	int i_bsize = blockIdx.x;
	int i_head = blockIdx.y;
	int i_isize = threadIdx.x;
	scalar_t ago = grad_o[i_bsize][last_index][i_head][i_isize];
	grad_x[i_bsize][last_index][i_head][i_isize] = ago * mbeta;
	for (int i = last_index - 1; i >= 0; i--) {
		ago = ago * beta + grad_o[i_bsize][i][i_head][i_isize];
		grad_x[i_bsize][i][i_head][i_isize] = ago * mbeta;
	}
}

at::Tensor mvavg_cuda_forward(torch::Tensor x, torch::Tensor o, float beta, int bsize, int seqlen, int nhead, int isize) {

	int block_size = isize;
	bool block_ul = (block_size <= MAX_BLOCK_SIZE);
	bool grid_x_ul = (bsize <= MAX_GRID_SIZE_X);
	bool grid_y_ul = (nhead <= MAX_GRID_SIZE_Y);
	float mbeta = 1.0 - beta;
	if (block_ul && grid_x_ul && grid_y_ul) {
		dim3 grid_size(bsize, nhead);
		AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "mvavg_aul_forward_cuda", ([&] {cuda_mvavg_aul_<scalar_t><<<grid_size, block_size>>>(x.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), o.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), (scalar_t)beta, (scalar_t)mbeta, seqlen);}));
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
			AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "mvavg_forward_cuda", ([&] {cuda_mvavg_<scalar_t><<<grid_size, block_size>>>(x.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), o.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), (scalar_t)beta, (scalar_t)mbeta, bsize, seqlen, nhead, isize);}));
	}

	return cell;
}

torch::Tensor mvavg_cuda_backward(torch::Tensor grad_o, torch::Tensor grad_x, float beta, int bsize, int seqlen, int nhead, int isize) {

	int block_size = isize;
	bool block_ul = (block_size <= MAX_BLOCK_SIZE);
	bool grid_x_ul = (bsize <= MAX_GRID_SIZE_X);
	bool grid_y_ul = (nhead <= MAX_GRID_SIZE_Y);
	float mbeta = 1.0 - beta;
	if (block_ul && grid_x_ul && grid_y_ul) {
		dim3 grid_size(bsize, nhead);
		AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_o.scalar_type(), "mvavg_aul_backward_cuda", ([&] {cuda_mvavg_aul_grad_<scalar_t><<<grid_size, block_size>>>(grad_o.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), grad_x.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), (scalar_t)beta, (scalar_t)mbeta, seqlen - 1);}));
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
		AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_o.scalar_type(), "mvavg_backward_cuda", ([&] {cuda_mvavg_grad_<scalar_t><<<grid_size, block_size>>>(grad_o.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), grad_x.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), (scalar_t)beta, (scalar_t)mbeta, bsize, nhead, isize, seqlen - 1);}));
	}

	return grad_x;
}
