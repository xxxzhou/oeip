#pragma once

#include <memory>
#include "../oeip/BaseLayer.h"
#include "common_help.h"
#include "ImageProcessCuda.h"
#include "cuda_help.h"
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/cuda.hpp>  
#include <vector_types.h>

using namespace cv;
using namespace cv::cuda;

class LayerCuda : public BaseLayer
{
public:
	LayerCuda() :LayerCuda(1, 1) {};
	LayerCuda(int32_t inSize, int32_t outSize) :BaseLayer(inSize, outSize) {
		inMats.resize(inCount);
		outMats.resize(outCount);
	}
	virtual ~LayerCuda();
public:
	std::vector<cv::cuda::GpuMat> outMats;
	std::vector<cv::cuda::GpuMat> inMats;
protected:	
	ImageProcessCuda* ipCuda = nullptr;
protected:
	virtual void setImageProcess(ImageProcess* process) override;
	//子类如果要改变实现，没有特殊处理,一定要调用LayerCuda::onInitBuffer,用来连接上层输出
	virtual bool onInitBuffer() override;
};

