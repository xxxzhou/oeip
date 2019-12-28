#pragma once
#include "matting_cuda.h"

class KmeansCuda
{
public:
	KmeansCuda();
	~KmeansCuda();
public:
	//GPU上
	kmeansI *bg = nullptr;
	kmeansI *fg = nullptr;
	//GpuMat clusterIndex;
private:
	int width = 0;
	int height = 0;
	float4* dcenters = nullptr;
	int *dlenght = nullptr;
	int *d_delta = nullptr;
	dim3 block;
	dim3 grid;
	cudaStream_t cudaStream = nullptr;
public:
	void init(int dwidth, int dheight, cudaStream_t stream);
	void compute(GpuMat source, GpuMat clusterIndex, GpuMat mask, int threshold, bool bSeed = false);
};

