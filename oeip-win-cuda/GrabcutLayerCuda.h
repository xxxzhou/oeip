#pragma once

#include "KmeansCuda.h"
#include "GraphCuda.h"
#include "GMMCuda.h"
#include "LayerCuda.h"

class GrabcutLayerCuda : public GrabcutLayer, public LayerCuda
{
public:
	GrabcutLayerCuda();
	~GrabcutLayerCuda();
private:
	int width = 0;
	int height = 0;
	bool bSeedMode = false;
	bool bComputeSeed = false;
	std::unique_ptr<KmeansCuda> kmeans = nullptr;
	std::unique_ptr<GMMCuda> gmm = nullptr;
	std::unique_ptr<GraphCuda> graph = nullptr;
	GpuMat mask;
	GpuMat clusterIndex;
	//临时保证一张grabcut要求大小的原色图
	GpuMat grabMat;
	GpuMat showMask;

	int uvX = -1;
	int uvY = -1;
	//手动画种子点模式,前景还是背景 0背景 1前景
	int groundMode = -1;

protected:
	virtual void onParametChange(GrabcutParamet oldT) override;
	virtual bool onInitBuffer() override;
	virtual void onRunLayer() override;
};

