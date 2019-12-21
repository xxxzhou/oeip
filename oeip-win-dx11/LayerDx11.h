#pragma once

#include <d3dcommon.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <atlbase.h>
#include <memory>
#include "../oeip/BaseLayer.h"
#include "../oeip-win/Dx11Resource.h"
#include "Dx11ComputeShader.h"
#include "ImageProcessDx11.h"

struct InputConstant
{
	uint32_t width;
	uint32_t height;
	//通道数据R(1),RG(2),RGBA(4)
	uint32_t elementCount;
	//每通道字节大小
	uint32_t elementByte;
};

//每层输出用outTexture，最后传buffer输出
//函数时序 initHlsl->onParametChange->initBuffer->updateCBuffer
class LayerDx11 : public BaseLayer
{
public:
	LayerDx11();
	virtual ~LayerDx11() {};
public:
	std::vector <std::shared_ptr<Dx11Texture>> outTextures;
protected:
	std::vector<ID3D11ShaderResourceView*> inSRVs;
	std::vector<ID3D11UnorderedAccessView*> outUAVs;

	std::unique_ptr<Dx11ComputeShader> computeShader = nullptr;
	std::unique_ptr<Dx11Constant> constBuffer = nullptr;
	ImageProcessDx11* dx11 = nullptr;
	std::string modeName = "oeip-win-dx11";
	std::string rctype = "HLSL";
	bool bInitHlsl = false;

	//DXGI_FORMAT dxFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
	InputConstant inputConstant = {};
protected:
	virtual void setImageProcess(ImageProcess* process) override;
	//当设定上下文后，先初始化对应HLSL资源
	virtual bool initHlsl() { return true; };
	virtual void onInitLayer() override;
	//当前层基本的大小分配，线程划分，如果要更改默认分配，在这修改
	virtual void onInitLayer(int32_t index) override {};
	//当上下文设定大小后,开始创建对应纹理与buffer
	virtual bool onInitBuffer() override;
	virtual bool onInitCBuffer();
	virtual bool updateCBuffer();
	virtual void onRunLayer() override;

};


