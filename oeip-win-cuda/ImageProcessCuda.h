#pragma once
#include <cuda.h>
#include <cuda_d3d11_interop.h>
#include <cuda_runtime.h>

#include "../oeip/ImageProcess.h"
#include "common_help.h" 
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

class ImageProcessCuda : public ImageProcess
{
public:
	ImageProcessCuda();
	~ImageProcessCuda();
public:
	//cudaStream的包装，自动管理cudaStreamCreate/cudaStreamDestroy
	cv::cuda::Stream stream;
	//当前运行ImageProcess的流
	cudaStream_t cudaStream = nullptr;
protected:
	// 通过 ImageProcess 继承
	virtual BaseLayer* onAddLayer(OeipLayerType layerType) override;
	virtual void onRunLayers() override;
public:
	void getGpuMat(int32_t layerIndex, cv::cuda::GpuMat& gpuMat, int32_t inIndex);
};

class ImageProcessCudaFactory :public ObjectFactory<ImageProcess>
{
public:
	ImageProcessCudaFactory() {};
	~ImageProcessCudaFactory() {};
public:
	virtual ImageProcess* create(int type) override;
};

extern "C" __declspec(dllexport) bool bCanLoad();
extern "C" __declspec(dllexport) void registerFactory();

