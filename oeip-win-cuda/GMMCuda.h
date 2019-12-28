#pragma once
#include "matting_cuda.h"

class GMMCuda
{
public:
	GMMCuda();
	~GMMCuda();
public:
	gmmI *bg = nullptr;
	gmmI *fg = nullptr;
private:
	int width = 0;
	int height = 0;
	float3* sumprods = nullptr;
	int* indexs = nullptr;
	dim3 block;
	dim3 grid;
	cudaStream_t cudaStream = nullptr;
public:
	void init(int dwidth, int dheight, cudaStream_t stream);
	void learning(GpuMat source, GpuMat clusterIndex, GpuMat mask, bool bSeed = false);
	void assign(GpuMat source, GpuMat clusterIndex, GpuMat mask);
};

