#pragma once
#include "LayerDx11.h"

class OutputLayerDx11 :public OutputLayer, public LayerDx11
{
public:
	OutputLayerDx11();
	~OutputLayerDx11() {};
private:
	std::vector<std::unique_ptr<Dx11SharedTex>> shardTexs;
	std::vector<std::unique_ptr<Dx11Buffer>> outBuffers;
	std::vector<CComPtr<ID3D11Buffer>> cpuReadBuffer;
	std::vector<CComPtr<ID3D11Texture2D>> outTexs;
protected:
	virtual void onParametChange(OutputParamet oldT) override;	
	virtual bool initHlsl() override;
	virtual bool onInitBuffer() override;
	virtual void onRunLayer() override;
public:
	// 通过 OutputLayer 继承
	virtual void outputGpuTex(void* device, void* texture, int32_t outputIndex) override;
};

