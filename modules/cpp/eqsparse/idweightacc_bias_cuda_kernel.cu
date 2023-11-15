#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_GRID_SIZE 2147483647
#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE_BASE 32
#define Batch_Per_Thread 1
#define MAX_Share_MEM 49152

#define GRID_SIZE_SCALE_BSIZE (MAX_BLOCK_SIZE * Batch_Per_Thread)
#define MAX_Forward_NCON_Share_MEM (MAX_Share_MEM / 4 / 2)

// x: (bsize, isize)
// rs: (bsize, osize)
// idx, weight: (osize, ncon)
// bias: (osize,)
__global__ void cuda_id_weight_acc_bias_sm_(float *x, int *idx, float *weight, float *bias, float *rs, int bsize, int isize, int osize, int ncon, int odim_per_grid, int batch_per_thread) {

	int num_threads = blockDim.x;
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	int i_osize = block_id * odim_per_grid;
	if (i_osize < osize) {
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
			float bias_base = bias[i_osize];
			for (int i_bsize = bsize_sid; i_bsize < bsize_eid; i_bsize++) {
				float _dp = bias_base;
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

__global__ void cuda_id_weight_acc_bias_(float *x, int *idx, float *weight, float *bias, float *rs, int bsize, int isize, int osize, int ncon, int odim_per_grid, int batch_per_thread) {

	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;
	int i_osize = block_id * odim_per_grid;
	if (i_osize < osize) {
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
			float bias_base = bias[i_osize];
			for (int i_bsize = bsize_sid; i_bsize < bsize_eid; i_bsize++) {
				float _dp = bias_base;
				int x_base = i_bsize * isize;
				for (int i_con = 0; i_con < ncon; i_con++) {
					int id_weight_offset = osize_ncon_base + i_con;
					_dp += x[x_base + idx[id_weight_offset]] * weight[id_weight_offset];
				}
				rs[i_bsize * osize + i_osize] = _dp;
			}
		}
	}
}

at::Tensor id_weightacc_bias_cuda_forward_(torch::Tensor x, torch::Tensor idx, torch::Tensor weight, torch::Tensor bias, torch::Tensor rs, int64_t bsize, int64_t isize, int64_t osize, int64_t ncon) {

	int block_size;
	int grid_size = osize;
	if (bsize >= GRID_SIZE_SCALE_BSIZE) {
		int grid_scale = (bsize - 1) / GRID_SIZE_SCALE_BSIZE + 1;
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
	int odim_per_grid = osize / grid_size;
	if (odim_per_grid < 1) {
		odim_per_grid = 1;
	}
	int batch_per_thread = bsize / block_size;

	#if MAX_Share_MEM > 0
	if (ncon < MAX_Forward_NCON_Share_MEM) {
		cuda_id_weight_acc_bias_sm_<<<grid_size, block_size, (ncon * (sizeof(int) + sizeof(float)))>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), bias.data_ptr<float>(), rs.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon, odim_per_grid, batch_per_thread);
	}
	else {
		cuda_id_weight_acc_bias_<<<grid_size, block_size>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), bias.data_ptr<float>(), rs.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon, odim_per_grid, batch_per_thread);
	}
	#else
	cuda_id_weight_acc_bias_<<<grid_size, block_size>>>(x.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(), bias.data_ptr<float>(), rs.data_ptr<float>(), (int)bsize, (int)isize, (int)osize, (int)ncon, odim_per_grid, batch_per_thread);
	#endif

	return rs;
}
