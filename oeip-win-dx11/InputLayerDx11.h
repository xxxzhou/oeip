#pragma once
#include "LayerDx11.h"

//StructureByteStride只能为4的倍数
class InputLayerDx11 :public InputLayer, public LayerDx11
{
public:
	InputLayerDx11();
	~InputLayerDx11() {};
private:
	std::vector<std::unique_ptr<Dx11SharedTex>> shardTexs;
	std::vector<std::unique_ptr<Dx11Buffer>> inBuffers;
protected:
	virtual void onParametChange(InputParamet oldT) override;
	virtual void onInitLayer() override;
	virtual bool initHlsl() override;
	virtual bool onInitBuffer() override;
	virtual void onRunLayer() override;
public:
	// 通过 InputLayer 继承
	//GPU 数据输入肯定在引擎device上的渲染线程中,和当前ImageProcessDx11的GPU数据处理不在同一线程,texture暂时只支持RGBA32类型
	virtual void inputGpuTex(void* device, void* texture, int32_t inputIndex) override;
	virtual void inputCpuData(uint8_t* byteData, int32_t inputIndex) override;
};

