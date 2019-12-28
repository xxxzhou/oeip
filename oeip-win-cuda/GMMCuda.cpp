#include "GMMCuda.h"
#include "common_help.h"

using namespace cv;
void updateGMM_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> mask, PtrStepSz<uchar> clusterIndex,
	float3* ksumprods, int* kindexs, gmmI& gmmbg, gmmI& gmmfg, bool bSeed = false, cudaStream_t stream = nullptr);
void learningGMM_gpu(gmmI& gmmbg, gmmI& gmmfg, cudaStream_t stream = nullptr);
void assignGMM_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> clusterIndex,
	PtrStepSz<uchar> mask, gmmI& gmmbg, gmmI& gmmfg, cudaStream_t stream = nullptr);

GMMCuda::GMMCuda() {
}

GMMCuda::~GMMCuda() {
	SAFE_GPUDATA(bg);
	SAFE_GPUDATA(fg);
	SAFE_GPUDATA(sumprods);
	SAFE_GPUDATA(indexs);
}

void GMMCuda::init(int dwidth, int dheight, cudaStream_t stream) {
	width = dwidth;
	height = dheight;
	cudaStream = stream;

	block = dim3(BLOCK_XGMM, BLOCK_YGMM);
	grid = dim3(divUp(width, block.x), divUp(height, block.y));

	reCudaAllocGpu((void**)&bg, sizeof(gmmI));
	reCudaAllocGpu((void**)&fg, sizeof(gmmI));

	cudaMemset(bg, 0, sizeof(gmmI));
	cudaMemset(fg, 0, sizeof(gmmI));

	int size = grid.x * grid.y * CUDA_GRABCUT_K2;
	reCudaAllocGpu((void**)&sumprods, size * sizeof(float3) * 4);
	reCudaAllocGpu((void**)&indexs, size * sizeof(int));
}

void GMMCuda::learning(GpuMat source, GpuMat clusterIndex, GpuMat mask, bool bSeed) {
	gmmI cbg = {};
	gmmI cfg = {};

	updateGMM_gpu(source, mask, clusterIndex, sumprods, indexs, *bg, *fg, bSeed, cudaStream);
	//cudaMemcpy(&cbg, bg, sizeof(gmmI), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&cfg, fg, sizeof(gmmI), cudaMemcpyDeviceToHost);

	learningGMM_gpu(*bg, *fg, cudaStream);
	//cudaMemcpy(&cbg, bg, sizeof(gmmI), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&cfg, fg, sizeof(gmmI), cudaMemcpyDeviceToHost);
}

void GMMCuda::assign(GpuMat source, GpuMat clusterIndex, GpuMat mask) {
	assignGMM_gpu(source, clusterIndex, mask, *bg, *fg, cudaStream);
}


