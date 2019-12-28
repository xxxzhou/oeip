#pragma once
#include "LayerCuda.h"

class MapChannelLayerCuda : public MapChannelLayer, public LayerCuda
{
public:
	MapChannelLayerCuda() {};
	~MapChannelLayerCuda() {};
protected:
	virtual void onRunLayer() override;
};

class YUV2RGBALayerCuda : public YUV2RGBALayer, public LayerCuda
{
public:
	YUV2RGBALayerCuda();
	~YUV2RGBALayerCuda() {};
private:
	void initConnect();
protected:
	virtual void onParametChange(YUV2RGBAParamet oldT) override;
	virtual void onInitLayer() override;
	virtual void onRunLayer() override;
};

class ResizeLayerCuda : public ResizeLayer, public LayerCuda
{
public:
	ResizeLayerCuda() {};
	~ResizeLayerCuda() {};
protected:
	virtual void onParametChange(ResizeParamet oldT) override;
	virtual void onInitLayer() override;
	virtual void onRunLayer() override;
};

class RGBA2YUVLayerCuda : public RGBA2YUVLayer, public LayerCuda
{
public:
	RGBA2YUVLayerCuda();
	~RGBA2YUVLayerCuda() {};
private:
	void initConnect();
protected:
	virtual void onParametChange(RGBA2YUVParamet oldT) override;
	virtual void onInitLayer() override;
	virtual void onRunLayer() override;
};

class BlendLayerCuda : public BlendLayer, public LayerCuda
{
public:
	BlendLayerCuda() :LayerCuda(2, 1) {};
	~BlendLayerCuda() {};
private:
	int32_t top = 0;
	int32_t left = 0;
	cv::cuda::GpuMat tempMat;
protected:
	virtual void onParametChange(BlendParamet oldT) override;
	virtual bool onInitBuffer() override;
	virtual void onRunLayer() override;
};

class OperateLayerCuda : public OperateLayer, public LayerCuda
{
public:
	OperateLayerCuda() {};
	~OperateLayerCuda() {};
protected:
	virtual void onRunLayer() override;
};

