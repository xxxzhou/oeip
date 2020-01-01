#pragma once
#include "../oeip/ImageProcess.h"
#include "../oeip-win/Dx11Resource.h"
#include <d3dcommon.h>
#include <d3d11.h>
#include <d3dcompiler.h>

class ImageProcessDx11 : public ImageProcess
{
public:
	ImageProcessDx11();
	virtual ~ImageProcessDx11();
private:
	std::string modeName = "oeip-win-dx11";
	std::string rctype = "HLSL";
public:
	ID3DInclude* includeShader = nullptr;
	ID3D11Device* device = nullptr;
	ID3D11DeviceContext* ctx = nullptr;
protected:
	// 通过 ImageProcess 继承
	virtual BaseLayer* onAddLayer(OeipLayerType layerType) override;
	virtual void onRunLayers() override;
public:
	void getTexture(int32_t layerIndex, std::shared_ptr<Dx11Texture>& texture, int32_t inIndex);
};

OEIP_DEFINE_PLUGIN_CLASS(ImageProcess, ImageProcessDx11)

extern "C" __declspec(dllexport) bool bCanLoad();
extern "C" __declspec(dllexport) void registerFactory();

