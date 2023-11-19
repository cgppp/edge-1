#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define MAX_GRID_SIZE 2147483647
#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE_BASE 32
#define MAX_Share_MEM 49152

#define MAX_Forward_NCON_Share_MEM (MAX_Share_MEM / 4 / 2)
#define MAX_Backward_NCON_Share_MEM (MAX_Share_MEM / 4)

// x: (bsize, isize)
// rs: (bsize, osize)
// idx, weight: (osize, ncon)
__global__ void cuda_id_weight_acc_sm_(float *x, int *idx, float *weight, float *rs, int bsize, int isize, int osize, int ncon) {

	int num_threads = blockDim.x;
	int num_blocks = gridDim.x;
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	int i_osize, osize_eid, bsize_sid, bsize_eid, bsize_per_thread, num_act_threads;
	int bsize_m1 = bsize - 1;
	if (num_blocks < osize) {
		int osize_per_block = (osize - 1) / num_blocks + 1;
		i_osize = block_id * osize_per_block;
		osize_eid = i_osize + osize_per_block;
		bsize_per_thread = bsize_m1 / num_threads + 1;
		bsize_sid = thread_id * bsize_per_thread;
		num_act_threads = bsize_m1 / bsize_per_thread + 1;
	}
	else {
		int block_per_osize = num_blocks / osize;
		i_osize = block_id / block_per_osize;
		osize_eid = i_osize + 1;
		bsize_per_thread = bsize_m1 / (block_per_osize * num_threads) + 1;
		int block_id_osize = block_id % block_per_osize;
		bsize_sid = (block_id_osize * num_threads + thread_id) * bsize_per_thread;
		int bpo_m1 = block_per_osize - 1;
		if (block_id_osize < bpo_m1) {
			num_act_threads = num_threads;
		}
		else {
			num_act_threads = (bsize_m1 - bpo_m1 * num_threads * bsize_per_thread) / bsize_per_thread + 1;
		}
	}
	if ((i_osize < osize) && (bsize_sid < bsize)) {
		if (osize_eid > osize) {
			osize_eid = osize;
		}
		bsize_eid = bsize_sid + bsize_per_thread;
		if (bsize_eid > bsize) {
			bsize_eid = bsize;
		}
		extern __shared__ int sm[];
		int* idx_cache = sm;
		float* weight_cache = (float*)&idx_cache[ncon];
		bool cnt_odim = i_osize < osize_eid;
		while (cnt_odim) {
			int osize_ncon_base = i_osize * ncon;
			int i_con = thread_id;
			while (i_con < ncon) {
				int id_weight_offset = osize_ncon_base + i_con;
				idx_cache[i_con] = idx[id_weight_offset];
				weight_cache[i_con] = weight[id_weight_offset];
				i_con += num_act_threads;
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
}

__global__ void cuda_id_weight_acc_sm_opb_(float *x, int *idx, float *weight, float *rs, int bsize, int isize, int osize, int ncon, int bsize_per_thread, int osize_per_block) {

	int num_threads = blockDim.x;
	int num_blocks = gridDim.x;
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	int i_osize = block_id * osize_per_block;
	int bsize_sid = thread_id * bsize_per_thread;
	int osize_eid = i_osize + osize_per_block;
	if (osize_eid > osize) {
		osize_eid = osize;
	}
	int bsize_eid = bsize_sid + bsize_per_thread;
	if (bsize_eid > bsize) {
		bsize_eid = bsize;
	}
	extern __shared__ int sm[];
	int* idx_cache = sm;
	float* weight_cache = (float*)&idx_cache[ncon];
	bool cnt_odim = i_osize < osize_eid;
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

__global__ void cuda_id_weight_acc_sm_bpo_(float *x, int *idx, float *weight, float *rs, int bsize, int isize, int osize, int ncon, int bsize_per_thread, int block_per_osize) {

	int num_threads = blockDim.x;
	int num_blocks = gridDim.x;
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	int i_osize = block_id / block_per_osize;
	int block_id_osize = block_id % block_per_osize;
	int bsize_sid = (block_id_osize * num_threads + thread_id) * bsize_per_thread;
	int bpo_m1 = block_per_osize - 1;
	int num_act_threads;
	if (block_id_osize < bpo_m1) {
		num_act_threads = num_threads;
	}
	else {
		num_act_threads = (bsize - 1 - bpo_m1 * num_threads * bsize_per_thread) / bsize_per_thread + 1;
	}
	if ((i_osize < osize) && (bsize_sid < bsize)) {
		int bsize_eid = bsize_sid + bsize_per_thread;
		if (bsize_eid > bsize) {
			bsize_eid = bsize;
		}
		extern __shared__ int sm[];
		int* idx_cache = sm;
		float* weight_cache = (float*)&idx_cache[ncon];
		int osize_ncon_base = i_osize * ncon;
		int i_con = thread_id;
		while (i_con < ncon) {
			int id_weight_offset = osize_ncon_base + i_con;
			idx_cache[i_con] = idx[id_weight_offset];
			weight_cache[i_con] = weight[id_weight_offset];
			i_con += num_act_threads;
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
	}
}

__global__ void cuda_id_weight_acc_(float *x, int *idx, float *weight, float *rs, int bsize, int isize, int osize, int ncon, int du_per_thread) {

	int cur_id = threadIdx.x + blockIdx.x * blockDim.x;
	int num_c_t = bsize * osize;
	if (cur_id < num_c_t) {
		int e_id;
		if ((cur_id + 1) == num_c_t) {
			e_id = num_c_t;
		}
		else {
			e_id = cur_id + du_per_thread;
		}
		for (; cur_id < e_id; cur_id++) {
			int i_osize = cur_id / bsize;
			int i_bsize = cur_id % bsize;
			int osize_ncon_base = i_osize * ncon;
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
__global__ void cuda_id_weight_acc_grad_sm_(float *x, int *idx, float *weight, torch::PackedTensorAccessor32<float, 2> grad_output, float *grad_x, float *grad_weight, int bsize, int isize, int osize, int ncon) {

	int num_threads = blockDim.x;
	int num_blocks = gridDim.x;
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	int i_osize, osize_eid, bsize_sid, bsize_eid, bsize_per_thread, num_act_threads;
	int bsize_m1 = (bsize - 1);
	if (num_blocks < osize) {
		int osize_per_block = (osize - 1) / num_blocks + 1;
		i_osize = block_id * osize_per_block;
		osize_eid = i_osize + osize_per_block;
		bsize_per_thread = bsize_m1 / num_threads + 1;
		bsize_sid = thread_id * bsize_per_thread;
		num_act_threads = bsize_m1 / bsize_per_thread + 1;
	}
	else {
		int block_per_osize = num_blocks / osize;
		i_osize = block_id / block_per_osize;
		osize_eid = i_osize + 1;
		bsize_per_thread = bsize_m1 / (block_per_osize * num_threads) + 1;
		int block_id_osize = block_id % block_per_osize;
		bsize_sid = (block_id_osize * num_threads + thread_id) * bsize_per_thread;
		int bpo_m1 = block_per_osize - 1;
		if (block_id_osize < bpo_m1) {
			num_act_threads = num_threads;
		}
		else {
			num_act_threads = (bsize_m1 - bpo_m1 * num_threads * bsize_per_thread) / bsize_per_thread + 1;
		}
	}
	if ((i_osize < osize) && (bsize_sid < bsize)) {
		if (osize_eid > osize) {
			osize_eid = osize;
		}
		bsize_eid = bsize_sid + bsize_per_thread;
		if (bsize_eid > bsize) {
			bsize_eid = bsize;
		}
		extern __shared__ int idx_cache[];
		bool cnt_odim = i_osize < osize_eid;
		while (cnt_odim) {
			int osize_ncon_base = i_osize * ncon;
			int i_con = thread_id;
			while (i_con < ncon) {
				idx_cache[i_con] = idx[osize_ncon_base + i_con];
				i_con += num_act_threads;
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
}

__global__ void cuda_id_weight_acc_grad_sm_opb_(float *x, int *idx, float *weight, torch::PackedTensorAccessor32<float, 2> grad_output, float *grad_x, float *grad_weight, int bsize, int isize, int osize, int ncon, int bsize_per_thread, int osize_per_block) {

	int num_threads = blockDim.x;
	int num_blocks = gridDim.x;
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	int i_osize = block_id * osize_per_block;
	int bsize_sid = thread_id * bsize_per_thread;
	int osize_eid = i_osize + osize_per_block;
	if (osize_eid > osize) {
		osize_eid = osize;
	}
	int bsize_eid = bsize_sid + bsize_per_thread;
	if (bsize_eid > bsize) {
		bsize_eid = bsize;
	}
	extern __shared__ int idx_cache[];
	bool cnt_odim = i_osize < osize_eid;
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

__global__ void cuda_id_weight_acc_grad_sm_bpo_(float *x, int *idx, float *weight, torch::PackedTensorAccessor32<float, 2> grad_output, float *grad_x, float *grad_weight, int bsize, int isize, int osize, int ncon, int bsize_per_thread, int block_per_osize) {

	int num_threads = blockDim.x;
	int num_blocks = gridDim.x;
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	int i_osize = block_id / block_per_osize;
	int block_id_osize = block_id % block_per_osize;
	int bsize_sid = (block_id_osize * num_threads + thread_id) * bsize_per_thread;
	int bpo_m1 = block_per_osize - 1;
	int num_act_threads;
	if (block_id_osize < bpo_m1) {
		num_act_threads = num_threads;
	}
	else {
		num_act_threads = (bsize - 1 - bpo_m1 * num_threads * bsize_per_thread) / bsize_per_thread + 1;
	}
	if ((i_osize < osize) && (bsize_sid < bsize)) {
		int bsize_eid = bsize_sid + bsize_per_thread;
		if (bsize_eid > bsize) {
			bsize_eid = bsize;
		}
		extern __shared__ int idx_cache[];
		int osize_ncon_base = i_osize * ncon;
		int i_con = thread_id;
		while (i_con < ncon) {
			idx_cache[i_con] = idx[osize_ncon_base + i_con];
			i_con += num_act_threads;
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
	}
}

__global__ void cuda_id_weight_acc_grad_(float *x, int *idx, float *weight, torch::PackedTensorAccessor32<float, 2> grad_output, float *grad_x, float *grad_weight, int bsize, int isize, int osize, int ncon, int du_per_thread) {

	int cur_id = threadIdx.x + blockIdx.x * blockDim.x;
	int num_c_t = bsize * osize;
	if (cur_id < num_c_t) {
		int e_id;
		if ((cur_id + 1) == num_c_t) {
			e_id = num_c_t;
		}
		else {
			e_id = cur_id + du_per_thread;
		}
		for (; cur_id < e_id; cur_id++) {
			int i_osize = cur_id / bsize;
			int i_bsize = cur_id % bsize;
			int osize_ncon_base = i_osize * ncon;
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
	int grid_size;

	if (ncon < MAX_Forward_NCON_Share_MEM) {
		int bsize_m1 = bsize - 1;
		if (bsize > MAX_BLOCK_SIZE) {
			block_size = MAX_BLOCK_SIZE;
			grid_size = osize * (bsize_m1 / MAX_BLOCK_SIZE + 1);
		}
		else {
			block_size = bsize;
			grid_size = osize;
		}
		if (grid_size > MAX_GRID_SIZE) {
			grid_size /= ((grid_size - 1) / MAX_GRID_SIZE + 1);
		}
		//cuda_id_weight_acc_sm_<<<grid_size, block_size, (ncon * (sizeof(int) + sizeof(float)))>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), rs.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon);
		if (grid_size < osize) {
			int bsize_per_thread = bsize_m1 / block_size + 1;
			block_size = bsize_m1 / bsize_per_thread + 1;
			int osize_m1 = osize - 1;
			int osize_per_block = osize_m1 / grid_size + 1;
			grid_size = osize_m1 / osize_per_block + 1;
			cuda_id_weight_acc_sm_opb_<<<grid_size, block_size, (ncon * (sizeof(int) + sizeof(float)))>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), rs.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon, bsize_per_thread, osize_per_block);
		}
		else {
			int block_per_osize = grid_size / osize;
			int bsize_per_thread = bsize_m1 / (block_per_osize * block_size) + 1;
			block_size = bsize_m1 / (block_per_osize * bsize_per_thread) + 1;
			cuda_id_weight_acc_sm_bpo_<<<grid_size, block_size, (ncon * (sizeof(int) + sizeof(float)))>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), rs.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon, bsize_per_thread, block_per_osize);
		}
	}
	else {
		int du_per_thread = 1;
		if (bsize > MAX_BLOCK_SIZE) {
			block_size = MAX_BLOCK_SIZE;
			grid_size = (bsize * osize - 1) / MAX_BLOCK_SIZE + 1;
		}
		else {
			if (bsize > BLOCK_SIZE_BASE) {
				block_size = bsize / BLOCK_SIZE_BASE * BLOCK_SIZE_BASE;
				grid_size = (bsize * osize - 1) / block_size + 1;
			}
			else {
				block_size = bsize;
				grid_size = osize;
			}
		}
		if (grid_size > MAX_GRID_SIZE) {
			int grid_scale = (grid_size - 1) / MAX_GRID_SIZE + 1;
			grid_size /= grid_scale;
			du_per_thread *= grid_scale;
		}
		cuda_id_weight_acc_<<<grid_size, block_size>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), rs.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon, du_per_thread);
	}

	return rs;
}

std::vector<torch::Tensor> id_weightacc_cuda_backward_(torch::Tensor x, torch::Tensor idx, torch::Tensor weight, torch::Tensor grad_output, torch::Tensor grad_x, torch::Tensor grad_weight, int64_t bsize, int64_t isize, int64_t osize, int64_t ncon) {

	int block_size;
	int grid_size;

	if (ncon < MAX_Backward_NCON_Share_MEM) {
		int bsize_m1 = bsize - 1;
		if (bsize > MAX_BLOCK_SIZE) {
			block_size = MAX_BLOCK_SIZE;
			grid_size = osize * (bsize_m1 / MAX_BLOCK_SIZE + 1);
		}
		else {
			block_size = bsize;
			grid_size = osize;
		}
		if (grid_size > MAX_GRID_SIZE) {
			grid_size /= ((grid_size - 1) / MAX_GRID_SIZE + 1);
		}
		//cuda_id_weight_acc_grad_sm_<<<grid_size, block_size, ncon * sizeof(int)>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), grad_output.packed_accessor32<float, 2>(), grad_x.data_ptr<float>(), grad_weight.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon);
		if (grid_size < osize) {
			int bsize_per_thread = bsize_m1 / block_size + 1;
			block_size = bsize_m1 / bsize_per_thread + 1;
			int osize_m1 = osize - 1;
			int osize_per_block = osize_m1 / grid_size + 1;
			grid_size = osize_m1 / osize_per_block + 1;
			cuda_id_weight_acc_grad_sm_opb_<<<grid_size, block_size, ncon * sizeof(int)>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), grad_output.packed_accessor32<float, 2>(), grad_x.data_ptr<float>(), grad_weight.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon, bsize_per_thread, osize_per_block);
		}
		else {
			int block_per_osize = grid_size / osize;
			int bsize_per_thread = bsize_m1 / (block_per_osize * block_size) + 1;
			block_size = bsize_m1 / (block_per_osize * bsize_per_thread) + 1;
			cuda_id_weight_acc_grad_sm_bpo_<<<grid_size, block_size, ncon * sizeof(int)>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), grad_output.packed_accessor32<float, 2>(), grad_x.data_ptr<float>(), grad_weight.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon, bsize_per_thread, block_per_osize);
		}
	}
	else {
		int du_per_thread = 1;
		if (bsize > MAX_BLOCK_SIZE) {
			block_size = MAX_BLOCK_SIZE;
			grid_size = (bsize * osize - 1) / MAX_BLOCK_SIZE + 1;
		}
		else {
			if (bsize > BLOCK_SIZE_BASE) {
				block_size = bsize / BLOCK_SIZE_BASE * BLOCK_SIZE_BASE;
				grid_size = (bsize * osize - 1) / block_size + 1;
			}
			else {
				block_size = bsize;
				grid_size = osize;
			}
		}
		if (grid_size > MAX_GRID_SIZE) {
			int grid_scale = (grid_size - 1) / MAX_GRID_SIZE + 1;
			grid_size /= grid_scale;
			du_per_thread *= grid_scale;
		}
		cuda_id_weight_acc_grad_<<<grid_size, block_size>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), grad_output.packed_accessor32<float, 2>(), grad_x.data_ptr<float>(), grad_weight.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon, du_per_thread);
	}

	return {grad_x, grad_weight};
}
