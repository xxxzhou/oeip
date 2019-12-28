#include "KmeansCuda.h"
#include "common_help.h"
//using namespace cv;
void initKmeans_gpu(kmeansI& meansbg, kmeansI& meansfg, cudaStream_t stream = nullptr);
void findNearestCluster_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> clusterIndex, PtrStepSz<uchar> mask,
	kmeansI& meansbg, kmeansI& meansfg, bool bSeed = false, cudaStream_t stream = nullptr);
void updateCluster_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> clusterIndex, PtrStepSz<uchar> mask, float4* kencter,
	int* kindexs, kmeansI& meansbg, kmeansI& meansfg, int& delta, bool bSeed = false, cudaStream_t stream = nullptr);

KmeansCuda::KmeansCuda() {
}


KmeansCuda::~KmeansCuda() {
	SAFE_GPUDATA(bg);
	SAFE_GPUDATA(fg);
	SAFE_GPUDATA(dcenters);
	SAFE_GPUDATA(dlenght);
	SAFE_GPUDATA(d_delta);
}

void KmeansCuda::init(int dwidth, int dheight, cudaStream_t stream) {
	width = dwidth;
	height = dheight;
	cudaStream = stream;

	block = dim3(BLOCK_X, BLOCK_Y);
	grid = dim3(cv::divUp(width, block.x), cv::divUp(height, block.y));

	reCudaAllocGpu((void**)&bg, sizeof(kmeansI));
	reCudaAllocGpu((void**)&fg, sizeof(kmeansI));

	cudaMemset(bg, 0, sizeof(kmeansI));
	cudaMemset(fg, 0, sizeof(kmeansI));

	int size = grid.x * grid.y * CUDA_GRABCUT_K2;
	reCudaAllocGpu((void**)&dcenters, size * sizeof(float4));
	reCudaAllocGpu((void**)&dlenght, size * sizeof(int));
	reCudaAllocGpu((void**)&d_delta, sizeof(int));
}

void KmeansCuda::compute(GpuMat source, GpuMat clusterIndex, GpuMat mask, int threshold, bool bSeed) {
	int delta = threshold + 1;
	initKmeans_gpu(*bg, *fg, cudaStream);
	while (delta > threshold) {
		findNearestCluster_gpu(source, clusterIndex, mask, *bg, *fg, bSeed, cudaStream);
		updateCluster_gpu(source, clusterIndex, mask, dcenters, dlenght, *bg, *fg, *d_delta, bSeed, cudaStream);
		cudaStreamSynchronize(cudaStream);
		//可以每隔五次调用一次这个
		cudaMemcpy(&delta, d_delta, sizeof(int), cudaMemcpyDeviceToHost);
	}
}
