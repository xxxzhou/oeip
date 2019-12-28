#pragma once
#include "matting_cuda.h"
class GraphCuda
{
public:
	GraphCuda();
	~GraphCuda();

private:
	int width = 0;
	int height = 0;
	int edgeCount = 0;

	GpuMat leftEdge;
	GpuMat rightEdge;
	GpuMat upEdge;
	GpuMat downEdge;

	GpuMat leftPull;
	GpuMat rightPull;
	GpuMat upPull;
	GpuMat downPull;

	GpuMat push;
	GpuMat sink;
	GpuMat graphHeight;
	GpuMat gmask;
	GpuMat relabel;

	int *bOver = nullptr;
	float *beta = nullptr;
	float *tempDiffs = nullptr;
	cudaStream_t cudaStream = nullptr;
public:
	void init(int dwidth, int dheight, cudaStream_t stream);

	void addTermWeights(GpuMat source, GpuMat mask, gmmI& gmmbg, gmmI& gmmfg, float lambda);
	void addTermWeights(GpuMat source, gmmI& gmmbg, gmmI& gmmfg, float lambda);

	void addEdges(GpuMat source, float gamma);

	void maxFlow(GpuMat mask, int maxCount);
};

