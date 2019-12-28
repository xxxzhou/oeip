#include "matting_help.h"

using namespace cv::cuda;

const dim3 block = dim3(BLOCK_X, BLOCK_Y);
const dim3 gmmBlock = dim3(BLOCK_XGMM, BLOCK_YGMM);

void showKmeans_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> clusterIndex, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(source.cols, block.x), cv::divUp(source.rows, block.y));
	showKmeans << <grid, block, 0, stream >> > (source, clusterIndex);
}

void setMask_gpu(PtrStepSz<uchar> mask, cv::Rect rect, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(mask.cols, block.x), cv::divUp(mask.rows, block.y));
	setMask << <grid, block, 0, stream >> > (mask, rect);
}

void setMask_gpu(PtrStepSz<uchar> source, PtrStepSz<uchar> mask, int radius, cv::Rect rect, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(source.cols, block.x), cv::divUp(source.rows, block.y));
	setMask << <grid, block, 0, stream >> > (source, mask, radius, rect);
}

void setMask_gpu(PtrStepSz<uchar> mask, int x, int y, int radius, int vmask, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(mask.cols, block.x), cv::divUp(mask.rows, block.y));
	setMask << <grid, block, 0, stream >> > (mask, x, y, radius, vmask);
}

void showSeedMask_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, PtrStepSz<uchar> mask, cudaStream_t stream = nullptr){
	float fx = (float)source.cols / mask.cols;
	float fy = (float)source.rows / mask.rows;
	dim3 grid(cv::divUp(source.cols, block.x), cv::divUp(source.rows, block.y));
	showSeedMask << <grid, block, 0, stream >> > (source, dest, mask, fx, fy);
}

void showMask_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> mask, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(source.cols, block.x), cv::divUp(source.rows, block.y));
	showMask << <grid, block, 0, stream >> > (source, mask);
}

void initKmeans_gpu(kmeansI& meansbg, kmeansI& meansfg, cudaStream_t stream = nullptr){
	initKmeans << <1, CUDA_GRABCUT_K, 0, stream >> > (meansbg, meansfg);
}

void findNearestCluster_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> clusterIndex, PtrStepSz<uchar> mask,
	kmeansI& meansbg, kmeansI& meansfg, bool bSeed = false, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(source.cols, block.x), cv::divUp(source.rows, block.y));
	if (bSeed)
		findNearestCluster<true> << <grid, block, 0, stream >> > (source, clusterIndex, mask, meansbg, meansfg);
	else
		findNearestCluster<false> << <grid, block, 0, stream >> > (source, clusterIndex, mask, meansbg, meansfg);
}

void updateCluster_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> clusterIndex, PtrStepSz<uchar> mask, float4* kencter,
	int* kindexs, kmeansI& meansbg, kmeansI& meansfg, int& delta, bool bSeed = false, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(source.cols, block.x), cv::divUp(source.rows, block.y));
	if (bSeed)
		updateCluster<BLOCK_X, BLOCK_Y, true> << <grid, block, 0, stream >> > (source, clusterIndex, mask, kencter, kindexs);
	else
		updateCluster<BLOCK_X, BLOCK_Y, false> << <grid, block, 0, stream >> > (source, clusterIndex, mask, kencter, kindexs);
	updateCluster<BLOCK_X, BLOCK_Y> << <1, block, 0, stream >> > (kencter, kindexs, meansbg, meansfg, delta, grid.x*grid.y);
}

void updateGMM_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> mask, PtrStepSz<uchar> clusterIndex,
	float3* ksumprods, int* kindexs, gmmI& gmmbg, gmmI& gmmfg, bool bSeed = false, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(source.cols, gmmBlock.x), cv::divUp(source.rows, gmmBlock.y));
	if (bSeed)
		updateGMM<BLOCK_XGMM, BLOCK_YGMM, true> << <grid, gmmBlock, 0, stream >> > (source, mask, clusterIndex, ksumprods, kindexs);
	else
		updateGMM<BLOCK_XGMM, BLOCK_YGMM, false> << <grid, gmmBlock, 0, stream >> > (source, mask, clusterIndex, ksumprods, kindexs);
	updateGMM<BLOCK_XGMM, BLOCK_YGMM> << <1, gmmBlock, 0, stream >> > (ksumprods, kindexs, gmmbg, gmmfg, grid.x*grid.y);
}

void learningGMM_gpu(gmmI& gmmbg, gmmI& gmmfg, cudaStream_t stream = nullptr){
	learningGMM << <1, CUDA_GRABCUT_K2, 0, stream >> > (gmmbg, gmmfg);
}

void assignGMM_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> clusterIndex, PtrStepSz<uchar> mask, gmmI& gmmbg, gmmI& gmmfg, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(source.cols, block.x), cv::divUp(source.rows, block.y));
	assignGMM << <grid, block, 0, stream >> > (source, clusterIndex, mask, gmmbg, gmmfg);
}

void calcBeta_gpu(PtrStepSz<uchar4> source, float* tempDiffs,
	int edgeCount, float& beta, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(source.cols, block.x), cv::divUp(source.rows, block.y));
	calcBeta<BLOCK_X, BLOCK_Y> << <grid, block, 0, stream >> > (source, tempDiffs);
	calcBeta<BLOCK_X, BLOCK_Y> << <1, block, 0, stream >> > (tempDiffs, grid.x*grid.y, edgeCount, beta);
}

void addTermWeights_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> mask, PtrStepSz<float> push, PtrStepSz<float> sink,
	gmmI& gmmbg, gmmI& gmmfg, float lambda, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(source.cols, block.x), cv::divUp(source.rows, block.y));
	addTermWeights << <grid, block, 0, stream >> > (source, mask, push, sink, gmmbg, gmmfg, lambda);
}

void addTermWeights_gpu(PtrStepSz<uchar4> source, PtrStepSz<float> push, PtrStepSz<float> sink,
	gmmI& gmmbg, gmmI& gmmfg, float weightOffset, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(source.cols, block.x), cv::divUp(source.rows, block.y));
	addTermWeights << <grid, block, 0, stream >> > (source, push, sink, gmmbg, gmmfg, weightOffset);
}

void addEdges_gpu(PtrStepSz<uchar4> source, PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	float *beta, float gamma, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(source.cols, block.x), cv::divUp(source.rows, block.y));
	addEdges << <grid, block, 0, stream >> > (source, rightEdge, leftEdge, upEdge, downEdge, beta, gamma);
}

void push_relabel_gpu(PtrStepSz<float> push, PtrStepSz<float> sink, PtrStepSz<int> graphHeight,
	PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	PtrStepSz<float> rightPull, PtrStepSz<float> leftPull, PtrStepSz<float> upPull, PtrStepSz<float> downPull, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(push.cols, block.x), cv::divUp(push.rows, block.y));
	push_relabel <BLOCK_X, BLOCK_Y> << <grid, block, 0, stream >> > (push, sink, graphHeight, rightEdge, leftEdge, upEdge, downEdge,
		rightPull, leftPull, upPull, downPull);
}

void test_pp_gpu(PtrStepSz<float> push, PtrStepSz<float> sink, PtrStepSz<int> graphHeight,
	PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	PtrStepSz<float> rightPull, PtrStepSz<float> leftPull, PtrStepSz<float> upPull, PtrStepSz<float> downPull, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(push.cols, block.x), cv::divUp(push.rows, block.y));
	test_pp1 << <grid, block, 0, stream >> > (push, sink, graphHeight, rightEdge, leftEdge, upEdge, downEdge,
		rightPull, leftPull, upPull, downPull);
	test_pp2 << <grid, block, 0, stream >> > (push, sink, graphHeight, rightEdge, leftEdge, upEdge, downEdge,
		rightPull, leftPull, upPull, downPull);
}

void bfsInit_gpu(PtrStepSz<uchar> mask, PtrStepSz<float> push, PtrStepSz<float> sink, PtrStepSz<int> graphHeight, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(push.cols, block.x), cv::divUp(push.rows, block.y));
	bfsInit << <grid, block, 0, stream >> > (mask, push, sink, graphHeight);
}

void bfs_gpu(PtrStepSz<uchar> mask, PtrStepSz<int> graphHeight,
	PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	int count, int& over, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(mask.cols, block.x), cv::divUp(mask.rows, block.y));
	bfs << <grid, block, 0, stream >> > (mask, graphHeight, rightEdge, leftEdge, upEdge, downEdge, count, over);
}

void maxflowMask_gpu(PtrStepSz<uchar> mask, PtrStepSz<int> graphHeight, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(mask.cols, block.x), cv::divUp(mask.rows, block.y));
	maxflowMask << <grid, block, 0, stream >> > (mask, graphHeight);
}

void combinGrabcutMask_gpu(PtrStepSz<uchar4> source, PtrStepSz<float4> dest, PtrStepSz<uchar> mask, cudaStream_t stream = nullptr){
	float fx = (float)source.cols / mask.cols;
	float fy = (float)source.rows / mask.rows;
	dim3 grid(cv::divUp(source.cols, block.x), cv::divUp(source.rows, block.y));
	combinGrabcutMask << <grid, block, 0, stream >> > (source, dest, mask, fx, fy);
}

void image2netData_gpu(PtrStepSz<uchar4> source, float* outData, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(source.cols, block.x), cv::divUp(source.rows, block.y));
	image2netData << <grid, block, 0, stream >> > (source, outData, source.cols*source.rows);
}

void drawRect_gpu(PtrStepSz<uchar4> source, cv::Rect rect, int radius, uchar4 drawColor, cudaStream_t stream = nullptr){
	dim3 grid(cv::divUp(source.cols, block.x), cv::divUp(source.rows, block.y));
	drawRect << <grid, block, 0, stream >> > (source, rect.x, rect.x + rect.width, rect.y, rect.y + rect.height, radius, drawColor);;
}
