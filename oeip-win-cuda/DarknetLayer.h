#pragma once

#if OEIPDNN
#pragma comment(lib,"darknet.lib") 
#include <functional>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include "../darknet/darknet.h"
#include "../oeip/Oeip.h"
#include "LayerCuda.h"

class DarknetLayerCuda : public DarknetLayer, public LayerCuda
{
public:
	DarknetLayerCuda();
	~DarknetLayerCuda();
private:
	network *net = nullptr;
	float* netInput = nullptr;
	cv::cuda::GpuMat netFrame;
	int32_t netWidth = 0;
	int32_t netHeight = 0;
	int32_t classs = 1;//自己训练的模型只有人物
protected:
	virtual void onParametChange(DarknetParamet oldT) override;
	virtual bool onInitBuffer() override;
	virtual void onRunLayer() override;
};

#endif