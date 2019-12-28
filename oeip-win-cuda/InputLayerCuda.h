#pragma once
#include "LayerCuda.h"

//现在CUDA与DX11交互，还是使用了DX11的共享纹理，需要一个独立的DX11上下文，后期考虑能不能替换掉，现在的问题是如果不使用
//这种方式，渲染线程要等CUDA的流执行完，除非后期找到一种有锁的CUDA与Dx11线程交互方式才能保证各不等待并能正确执行。
class InputLayerCuda : public InputLayer, public LayerCuda
{
public:
	InputLayerCuda();
	~InputLayerCuda();
private:
	std::vector<std::shared_ptr<Dx11SharedTex>> shardTexs;
	std::vector<Dx11CudaResource> cudaResoures;
	std::vector<cv::cuda::GpuMat> tempMat;
	CComPtr<ID3D11Device> device = nullptr;
	CComPtr<ID3D11DeviceContext> ctx = nullptr;
	std::vector<bool> cpuUpdates;
protected:
	virtual void onParametChange(InputParamet oldT) override;
	//当上下文设定大小后,开始创建对应纹理与buffer
	virtual void onInitLayer() override;
	virtual bool onInitBuffer() override;
	virtual void onRunLayer() override;
public:
	//通过 InputLayer 继承
	//GPU 数据输入肯定在引擎device上的渲染线程中,和当前ImageProcessDx11的GPU数据处理不在同一线程,texture暂时只支持RGBA32类型
	virtual void inputGpuTex(void* device, void* texture, int32_t inputIndex) override;
	virtual void inputCpuData(uint8_t* byteData, int32_t inputIndex) override;
};

