#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define MAX_GRID_SIZE 2147483647
#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE_BASE 32
#define Batch_Per_Thread 1
#define MAX_Share_MEM 49152

#define GRID_SIZE_SCALE_BSIZE (MAX_BLOCK_SIZE * Batch_Per_Thread)
#define MAX_Forward_NCON_Share_MEM (MAX_Share_MEM / 4 / 2)
#define MAX_Backward_NCON_Share_MEM (MAX_Share_MEM / 4)

// x: (bsize, isize)
// rs: (bsize, osize)
// idx, weight: (osize, ncon)
__global__ void cuda_id_weight_acc_sm_(float *x, int *idx, float *weight, float *rs, int bsize, int isize, int osize, int ncon, int odim_per_grid, int batch_per_thread) {

	int num_threads = blockDim.x;
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	int i_osize = block_id * odim_per_grid;
	int osize_eid;
	if ((block_id + 1) < gridDim.x) {
		osize_eid = i_osize + odim_per_grid;
	}
	else {
		osize_eid = osize;
	}
	int bsize_sid = thread_id * batch_per_thread;
	int bsize_eid;
	if ((thread_id + 1) < num_threads) {
		bsize_eid = bsize_sid + batch_per_thread;
	}
	else {
		bsize_eid = bsize;
	}
	extern __shared__ int sm[];
	int* idx_cache = sm;
	float* weight_cache = (float*)&idx_cache[ncon];
	bool cnt_odim = true;
	while (cnt_odim) {
		int osize_ncon_base = i_osize * ncon;
		int i_con = thread_id;
		while (i_con < ncon) {
			int id_weight_offset = osize_ncon_base + i_con;
			idx_cache[i_con] = idx[id_weight_offset];
			weight_cache[i_con] = weight[id_weight_offset];
			i_con += num_threads;
		}
		__syncthreads();
		for (int i_bsize = bsize_sid; i_bsize < bsize_eid; i_bsize++) {
			float _dp = 0.0;
			int x_base = i_bsize * isize;
			for (int i_con = 0; i_con < ncon; i_con++) {
				_dp += x[x_base + idx_cache[i_con]] * weight_cache[i_con];
			}
			rs[i_bsize * osize + i_osize] = _dp;
		}
		i_osize += 1;
		cnt_odim = i_osize < osize_eid;
		if (cnt_odim) {
			__syncthreads();
		}
	}
}

__global__ void cuda_id_weight_acc_(float *x, int *idx, float *weight, float *rs, int bsize, int isize, int osize, int ncon, int odim_per_grid, int batch_per_thread) {

	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	int i_osize = block_id * odim_per_grid;
	int osize_eid;
	if ((block_id + 1) < gridDim.x) {
		osize_eid = i_osize + odim_per_grid;
	}
	else {
		osize_eid = osize;
	}
	int bsize_sid = thread_id * batch_per_thread;
	int bsize_eid;
	if ((thread_id + 1) < blockDim.x) {
		bsize_eid = bsize_sid + batch_per_thread;
	}
	else {
		bsize_eid = bsize;
	}
	for (; i_osize < osize_eid; i_osize++) {
		int osize_ncon_base = i_osize * ncon;
		for (int i_bsize = bsize_sid; i_bsize < bsize_eid; i_bsize++) {
			float _dp = 0.0;
			int x_base = i_bsize * isize;
			for (int i_con = 0; i_con < ncon; i_con++) {
				int id_weight_offset = osize_ncon_base + i_con;
				_dp += x[x_base + idx[id_weight_offset]] * weight[id_weight_offset];
			}
			rs[i_bsize * osize + i_osize] = _dp;
		}
	}
}

// x, grad_x: (bsize, isize)
// grad_output: (bsize, osize)
// idx, weight, grad_weight: (osize, ncon)
__global__ void cuda_id_weight_acc_grad_sm_(float *x, int *idx, float *weight, torch::PackedTensorAccessor32<float, 2> grad_output, float *grad_x, float *grad_weight, int bsize, int isize, int osize, int ncon, int odim_per_grid, int batch_per_thread) {

	int num_threads = blockDim.x;
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	int i_osize = block_id * odim_per_grid;
	int osize_eid;
	if ((block_id + 1) < gridDim.x) {
		osize_eid = i_osize + odim_per_grid;
	}
	else {
		osize_eid = osize;
	}
	int bsize_sid = thread_id * batch_per_thread;
	int bsize_eid;
	if ((thread_id + 1) < num_threads) {
		bsize_eid = bsize_sid + batch_per_thread;
	}
	else {
		bsize_eid = bsize;
	}
	extern __shared__ int idx_cache[];
	bool cnt_odim = true;
	while (cnt_odim) {
		int osize_ncon_base = i_osize * ncon;
		int i_con = thread_id;
		while (i_con < ncon) {
			idx_cache[i_con] = idx[osize_ncon_base + i_con];
			i_con += num_threads;
		}
		__syncthreads();
		for (int i_bsize = bsize_sid; i_bsize < bsize_eid; i_bsize++) {
			int x_base = i_bsize * isize;
			float grad_weight_scalar = grad_output[i_bsize][i_osize];//[i_bsize * osize + i_osize]
			for (int i_con = 0; i_con < ncon; i_con++) {
				int x_ind = x_base + idx_cache[i_con];
				int weight_ind = osize_ncon_base + i_con;
				atomicAdd(&grad_x[x_ind], weight[weight_ind] * grad_weight_scalar);
				atomicAdd(&grad_weight[weight_ind], x[x_ind] * grad_weight_scalar);
			}
		}
		i_osize += 1;
		cnt_odim = i_osize < osize_eid;
		if (cnt_odim) {
			__syncthreads();
		}
	}
}

__global__ void cuda_id_weight_acc_grad_(float *x, int *idx, float *weight, torch::PackedTensorAccessor32<float, 2> grad_output, float *grad_x, float *grad_weight, int bsize, int isize, int osize, int ncon, int odim_per_grid, int batch_per_thread) {

	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	int i_osize = block_id * odim_per_grid;
	int osize_eid;
	if ((block_id + 1) < gridDim.x) {
		osize_eid = i_osize + odim_per_grid;
	}
	else {
		osize_eid = osize;
	}
	int bsize_sid = thread_id * batch_per_thread;
	int bsize_eid;
	if ((thread_id + 1) < blockDim.x) {
		bsize_eid = bsize_sid + batch_per_thread;
	}
	else {
		bsize_eid = bsize;
	}
	for (; i_osize < osize_eid; i_osize++) {
		int osize_ncon_base = i_osize * ncon;
		for (int i_bsize = bsize_sid; i_bsize < bsize_eid; i_bsize++) {
			int x_base = i_bsize * isize;
			float grad_weight_scalar = grad_output[i_bsize][i_osize];//[i_bsize * osize + i_osize]
			for (int i_con = 0; i_con < ncon; i_con++) {
				int weight_ind = osize_ncon_base + i_con;
				int x_ind = x_base + idx[weight_ind];
				atomicAdd(&grad_x[x_ind], weight[weight_ind] * grad_weight_scalar);
				atomicAdd(&grad_weight[weight_ind], x[x_ind] * grad_weight_scalar);
			}
		}
	}
}

at::Tensor id_weightacc_cuda_forward_(torch::Tensor x, torch::Tensor idx, torch::Tensor weight, torch::Tensor rs, int64_t bsize, int64_t isize, int64_t osize, int64_t ncon) {

	int block_size;
	int grid_size = osize;
	int bsize_m1 = bsize - 1;
	if (bsize >= GRID_SIZE_SCALE_BSIZE) {
		int grid_scale = bsize_m1 / GRID_SIZE_SCALE_BSIZE + 1;
		grid_size *= grid_scale;
		block_size = bsize / grid_scale;
		block_size = block_size / BLOCK_SIZE_BASE / Batch_Per_Thread * BLOCK_SIZE_BASE;
	}
	#if Batch_Per_Thread != 1
	else if (bsize >= BLOCK_SIZE_BASE * Batch_Per_Thread) {
		block_size = bsize / BLOCK_SIZE_BASE / Batch_Per_Thread * BLOCK_SIZE_BASE;
	}
	#endif
	else if (bsize >= BLOCK_SIZE_BASE) {
		block_size = bsize / BLOCK_SIZE_BASE * BLOCK_SIZE_BASE;
	}
	else {
		block_size = bsize;
	}
	if (grid_size > MAX_GRID_SIZE) {
		int grid_scale = (grid_size - 1) / MAX_GRID_SIZE + 1;
		grid_size /= grid_scale;
	}
	int osize_m1 = osize - 1;
	int odim_per_grid = osize_m1 / grid_size + 1;
	grid_size = osize_m1 / odim_per_grid + 1;
	int batch_per_thread = bsize_m1 / block_size + 1;
	block_size = bsize_m1 / batch_per_thread + 1;

	#if MAX_Share_MEM > 0
	if (ncon < MAX_Forward_NCON_Share_MEM) {
		cuda_id_weight_acc_sm_<<<grid_size, block_size, (ncon * (sizeof(int) + sizeof(float)))>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), rs.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon, odim_per_grid, batch_per_thread);
	}
	else {
		cuda_id_weight_acc_<<<grid_size, block_size>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), rs.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon, odim_per_grid, batch_per_thread);
	}
	#else
	cuda_id_weight_acc_<<<grid_size, block_size>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), rs.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon, odim_per_grid, batch_per_thread);
	#endif

	return rs;
}

std::vector<torch::Tensor> id_weightacc_cuda_backward_(torch::Tensor x, torch::Tensor idx, torch::Tensor weight, torch::Tensor grad_output, torch::Tensor grad_x, torch::Tensor grad_weight, int64_t bsize, int64_t isize, int64_t osize, int64_t ncon) {

	int block_size;
	int grid_size = osize;
	int bsize_m1 = bsize - 1;
	if (bsize >= GRID_SIZE_SCALE_BSIZE) {
		int grid_scale = bsize_m1 / GRID_SIZE_SCALE_BSIZE + 1;
		grid_size *= grid_scale;
		block_size = bsize / grid_scale;
		block_size = block_size / BLOCK_SIZE_BASE / Batch_Per_Thread * BLOCK_SIZE_BASE;
	}
	#if Batch_Per_Thread != 1
	else if (bsize >= BLOCK_SIZE_BASE * Batch_Per_Thread) {
		block_size = bsize / BLOCK_SIZE_BASE / Batch_Per_Thread * BLOCK_SIZE_BASE;
	}
	#endif
	else if (bsize >= BLOCK_SIZE_BASE) {
		block_size = bsize / BLOCK_SIZE_BASE * BLOCK_SIZE_BASE;
	}
	else {
		block_size = bsize;
	}
	if (grid_size > MAX_GRID_SIZE) {
		int grid_scale = (grid_size - 1) / MAX_GRID_SIZE + 1;
		grid_size /= grid_scale;
	}
	int osize_m1 = osize - 1;
	int odim_per_grid = osize_m1 / grid_size + 1;
	grid_size = osize_m1 / odim_per_grid + 1;
	int batch_per_thread = bsize_m1 / block_size + 1;
	block_size = bsize_m1 / batch_per_thread + 1;

	#if MAX_Share_MEM > 0
	if (ncon < MAX_Backward_NCON_Share_MEM) {
		cuda_id_weight_acc_grad_sm_<<<grid_size, block_size, ncon * sizeof(int)>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), grad_output.packed_accessor32<float, 2>(), grad_x.data_ptr<float>(), grad_weight.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon, odim_per_grid, batch_per_thread);
	}
	else {
		cuda_id_weight_acc_grad_<<<grid_size, block_size>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), grad_output.packed_accessor32<float, 2>(), grad_x.data_ptr<float>(), grad_weight.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon, odim_per_grid, batch_per_thread);
	}
	#else
	cuda_id_weight_acc_grad_<<<grid_size, block_size>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), grad_output.packed_accessor32<float, 2>(), grad_x.data_ptr<float>(), grad_weight.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon, odim_per_grid, batch_per_thread);
	#endif

	return {grad_x, grad_weight};
}
