#include "GraphCuda.h"
#include "common_help.h"

void calcBeta_gpu(PtrStepSz<uchar4> source, float* tempDiffs,
	int edgeCount, float& beta, cudaStream_t stream = nullptr);
void addTermWeights_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> mask, PtrStepSz<float> push, PtrStepSz<float> sink,
	gmmI& gmmbg, gmmI& gmmfg, float lambda, cudaStream_t stream = nullptr);
void addTermWeights_gpu(PtrStepSz<uchar4> source, PtrStepSz<float> push, PtrStepSz<float> sink,
	gmmI& gmmbg, gmmI& gmmfg, float weightOffset, cudaStream_t stream = nullptr);
void addEdges_gpu(PtrStepSz<uchar4> source, PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	float* beta, float gamma, cudaStream_t stream = nullptr);
void push_relabel_gpu(PtrStepSz<float> push, PtrStepSz<float> sink, PtrStepSz<int> graphHeight,
	PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	PtrStepSz<float> rightPull, PtrStepSz<float> leftPull, PtrStepSz<float> upPull, PtrStepSz<float> downPull, cudaStream_t stream = nullptr);
void bfsInit_gpu(PtrStepSz<uchar> mask, PtrStepSz<float> push, PtrStepSz<float> sink, PtrStepSz<int> graphHeight, cudaStream_t stream = nullptr);
void bfs_gpu(PtrStepSz<uchar> mask, PtrStepSz<int> graphHeight,
	PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	int count, int& over, cudaStream_t stream = nullptr);
void maxflowMask_gpu(PtrStepSz<uchar> mask, PtrStepSz<int> graphHeight, cudaStream_t stream = nullptr);
void push_gpu(PtrStepSz<float> push, PtrStepSz<float> sink, PtrStepSz<uchar> relabel, PtrStepSz<int> graphHeight,
	PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	PtrStepSz<float> rightPull, PtrStepSz<float> leftPull, PtrStepSz<float> upPull, PtrStepSz<float> downPull, cudaStream_t stream = nullptr);
void relabe_gpu(PtrStepSz<float> push, PtrStepSz<float> sink, PtrStepSz<uchar> relabel, PtrStepSz<int> graphHeight,
	PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	PtrStepSz<float> rightPull, PtrStepSz<float> leftPull, PtrStepSz<float> upPull, PtrStepSz<float> downPull, cudaStream_t stream = nullptr);
void test_pp_gpu(PtrStepSz<float> push, PtrStepSz<float> sink, PtrStepSz<int> graphHeight,
	PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	PtrStepSz<float> rightPull, PtrStepSz<float> leftPull, PtrStepSz<float> upPull, PtrStepSz<float> downPull, cudaStream_t stream = nullptr);


GraphCuda::GraphCuda() {
}


GraphCuda::~GraphCuda() {
	SAFE_GPUDATA(bOver);
	SAFE_GPUDATA(beta);
	SAFE_GPUDATA(tempDiffs);
}

void GraphCuda::init(int dwidth, int dheight, cudaStream_t stream) {
	width = dwidth;
	height = dheight;
	cudaStream = stream;

	dim3 block = dim3(BLOCK_X, BLOCK_Y);
	dim3 grid = dim3(cv::divUp(width, block.x), cv::divUp(height, block.y));

	leftEdge.create(height, width, CV_32FC1);
	rightEdge.create(height, width, CV_32FC1);
	upEdge.create(height, width, CV_32FC1);
	downEdge.create(height, width, CV_32FC1);

	leftPull.create(height, width, CV_32FC1);
	rightPull.create(height, width, CV_32FC1);
	upPull.create(height, width, CV_32FC1);
	downPull.create(height, width, CV_32FC1);

	push.create(height, width, CV_32FC1);
	sink.create(height, width, CV_32FC1);
	graphHeight.create(height, width, CV_32SC1);
	gmask.create(height, width, CV_8UC1);
	relabel.create(height, width, CV_8UC1);

	reCudaAllocGpu((void**)&bOver, sizeof(int));
	reCudaAllocGpu((void**)&beta, sizeof(float));

	int size = grid.x * grid.y;
	reCudaAllocGpu((void**)&tempDiffs, size * sizeof(float));

	edgeCount = 2 * width * height - (width + height);
}

void GraphCuda::addTermWeights(GpuMat source, GpuMat mask, gmmI& gmmbg, gmmI& gmmfg, float lambda) {
	cudaMemset2DAsync(push.ptr(), push.step, 0, push.cols * 4, push.rows, cudaStream);
	cudaMemset2DAsync(sink.ptr(), sink.step, 0, sink.cols * 4, sink.rows, cudaStream);
	addTermWeights_gpu(source, mask, push, sink, gmmbg, gmmfg, lambda, cudaStream);
}

void GraphCuda::addTermWeights(GpuMat source, gmmI& gmmbg, gmmI& gmmfg, float weightOffset) {
	cudaMemset2DAsync(push.ptr(), push.step, 0, push.cols * 4, push.rows, cudaStream);
	cudaMemset2DAsync(sink.ptr(), sink.step, 0, sink.cols * 4, sink.rows, cudaStream);
	addTermWeights_gpu(source, push, sink, gmmbg, gmmfg, weightOffset, cudaStream);
}

void GraphCuda::addEdges(GpuMat source, float gamma) {	
	//边缘部分值很小,在0-1左右,而非边缘部分值在gamma-1左右
	calcBeta_gpu(source, tempDiffs, edgeCount, *beta, cudaStream);
	addEdges_gpu(source, rightEdge, leftEdge, upEdge, downEdge, beta, gamma, cudaStream);
	//showMat(rightEdge, leftEdge, downEdge);
}

void GraphCuda::maxFlow(GpuMat mask, int maxCount) {
	int count = 0;
	cudaMemset2DAsync(graphHeight.ptr(), graphHeight.step, 1, graphHeight.cols * 4, graphHeight.rows, cudaStream);
	cudaMemset2DAsync(relabel.ptr(), relabel.step, 0, relabel.cols * 4, relabel.rows, cudaStream);
	while (count < maxCount) {
		push_relabel_gpu(push, sink, graphHeight,
			rightEdge, leftEdge, upEdge, downEdge,
			rightPull, leftPull, upPull, downPull, cudaStream);
		count++;
	}
	//showMat(push, sink, graphHeight);
	bfsInit_gpu(gmask, push, sink, graphHeight, cudaStream);
	int cbOver = true;
	count = 1;
	while (cbOver) {
		cudaMemsetAsync(bOver, 0, sizeof(int), cudaStream);
		bfs_gpu(gmask, graphHeight, rightEdge, leftEdge, upEdge, downEdge, count, *bOver, cudaStream);
		cudaStreamSynchronize(cudaStream);
		cudaMemcpy(&cbOver, bOver, sizeof(int), cudaMemcpyDeviceToHost);
		count++;
	}
	maxflowMask_gpu(mask, graphHeight, cudaStream);
	//showMat(mask);
}
