#pragma once

#include "KmeansCuda.h"
#include "GraphCuda.h"
#include "GMMCuda.h"

class GrabCutCude
{
public:
	GrabCutCude();
	~GrabCutCude();
public:
	GpuMat mask;
private:
	//GpuMat mask;
	//GpuMat source;
	GpuMat clusterIndex;

	GpuMat showSource;
	KmeansCuda *kmeans = nullptr;
	GraphCuda *graph = nullptr;
	GMMCuda *gmm = nullptr;

	int width = 0;
	int height = 0;

	float gamma = 90.f;
	float lambda = 450.f;
	int maxCount = 250;
	int iterCount = 1;

	bool bComputeSeed = false;
	cudaStream_t cudaStream = nullptr;
public:
	void init(int dwidth, int dheight, cudaStream_t stream = nullptr);
	void setSeedMode(bool bDraw);
	void computeSeed(GpuMat source);
	void renderFrame(GpuMat source, cv::Rect rect);
	void renderFrame(GpuMat source);
	//如果背景和前景比较像,减少weightOffset,增加egdeScale
	void setParams(int iiterCount, float egdeScale, float weightOffset, int imaxCount);
};


