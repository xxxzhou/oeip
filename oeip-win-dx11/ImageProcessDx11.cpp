#include "../oeip/ImageProcess.h"
#include "ImageProcessDx11.h"
#include "InputLayerDx11.h"
#include "OutputLayerDx11.h"
#include "YUV2RGBALayerDX11.h"
#include "MapChannelLayerDx11.h"
#include "ResizeLayerDx11.h"

ImageProcessDx11::ImageProcessDx11() {
	createDevice11(&device, &ctx);
	includeShader = new ShaderInclude(modeName, rctype, 104);
}

ImageProcessDx11::~ImageProcessDx11() {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	safeDelete(includeShader);
	safeRelease(ctx);
	safeRelease(device);
}

BaseLayer* ImageProcessDx11::onAddLayer(OeipLayerType layerType) {
	BaseLayer* layer = nullptr;
	switch (layerType)
	{
	case OEIP_NONE_LAYER:
		break;
	case OEIP_INPUT_LAYER:
		layer = new InputLayerDx11();
		break;
	case OEIP_YUV2RGBA_LAYER:
		layer = new YUV2RGBALayerDX11();
		break;
	case OEIP_MAPCHANNEL_LAYER:
		layer = new MapChannelLayerDx11();
		break;
	case OEIP_RGBA2YUV_LAYER:
		break;
	case OEIP_RESIZE_LAYER:
		layer = new ResizeLayerDx11();
		break;
	case OEIP_OUTPUT_LAYER:
		layer = new OutputLayerDx11();
		break;
	case OEIP_MAX_LAYER:
		break;
	default:
		break;
	}
	if (layer) {
		layer->setImageProcess(this);
	}
	else {
		std::string message = "dx11 not support layer: " + getLayerName(layerType);
		logMessage(OEIP_WARN, message.c_str());
	}
	return layer;
}

void ImageProcessDx11::onRunLayers() {
	for (auto layer : layers) {
		if (layer->bDisable || layer->bDisableList)
			continue;
		layer->runLayer();
	}
}

void ImageProcessDx11::getTexture(int32_t layerIndex, std::shared_ptr<Dx11Texture>& texture, int32_t inIndex) {
	auto layer = std::dynamic_pointer_cast<LayerDx11>(layers[layerIndex]);
	texture = layer->outTextures[inIndex];
}

ImageProcess* ImageProcessDx11Factory::create(int type) {
	ImageProcessDx11* pdx11 = new ImageProcessDx11();
	return pdx11;
}

bool bCanLoad() {
	return true;
}

void registerFactory() {
	registerFactory(new ImageProcessDx11Factory(), OeipGpgpuType::OEIP_DX11, "image process dx11");
}