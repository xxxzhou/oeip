#pragma once
#include "LayerCuda.h"

class OutputLayerCuda : public OutputLayer, public LayerCuda
{
public:
	OutputLayerCuda();
	~OutputLayerCuda();
private:
	std::vector<std::shared_ptr<Dx11SharedTex>> shardTexs;
	std::vector<Dx11CudaResource> cudaResoures;
	std::vector<std::vector<uint8_t>> cpudatas;
	CComPtr<ID3D11Device> device = nullptr;
	CComPtr<ID3D11DeviceContext> ctx = nullptr;
protected:
	virtual void onParametChange(OutputParamet oldT) override;
	//当上下文设定大小后,开始创建对应纹理与buffer
	virtual bool onInitBuffer() override;
	virtual void onRunLayer() override;
public:
	// 通过 OutputLayer 继承
	virtual void outputGpuTex(void* device, void* texture, int32_t outputIndex) override;
};

