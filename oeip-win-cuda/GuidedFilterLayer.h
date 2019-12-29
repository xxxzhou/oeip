#pragma once
#include "LayerCuda.h"

class GuidedFilterLayerCuda :public GuidedFilterLayer, public LayerCuda
{
public:
	GuidedFilterLayerCuda();
	~GuidedFilterLayerCuda();
private:
	int32_t scaleWidth = 1;
	int32_t scaleHeight = 1;
private:
	cv::cuda::GpuMat resizeMat;
	cv::cuda::GpuMat resizeMatf;
	cv::cuda::GpuMat mean_I;
	cv::cuda::GpuMat mean_Ipv;
	cv::cuda::GpuMat var_I_rxv;
	cv::cuda::GpuMat var_I_gbxfv;
	cv::cuda::GpuMat mean_Ip;
	cv::cuda::GpuMat var_I_rx;
	cv::cuda::GpuMat var_I_gbxf;
	cv::cuda::GpuMat meanv;
	cv::cuda::GpuMat means;
	cv::cuda::GpuMat mean;
protected:
	virtual void onParametChange(GuidedFilterParamet oldT) override;
	virtual bool onInitBuffer() override;
	virtual void onRunLayer() override;
};

