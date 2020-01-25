#include "ImageProcessCuda.h"
#include "LayerCuda.h"
#include "InputLayerCuda.h"
#include "OutputLayerCuda.h"
#include "CudaComputeLayer.h"
#include "DarknetLayer.h"
#include "GrabcutLayerCuda.h"

ImageProcessCuda::ImageProcessCuda() {
	stream = {};
	cudaStream = cv::cuda::StreamAccessor::getStream(stream);
}

ImageProcessCuda::~ImageProcessCuda() {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
}

BaseLayer* ImageProcessCuda::onAddLayer(OeipLayerType layerType) {
	BaseLayer* layer = nullptr;
	switch (layerType) {
	case OEIP_NONE_LAYER:
		break;
	case OEIP_INPUT_LAYER:
		layer = new InputLayerCuda();
		break;
	case OEIP_OUTPUT_LAYER:
		layer = new OutputLayerCuda();
		break;
	case OEIP_YUV2RGBA_LAYER:
		layer = new YUV2RGBALayerCuda();
		break;
	case OEIP_MAPCHANNEL_LAYER:
		layer = new MapChannelLayerCuda();
		break;
	case OEIP_RGBA2YUV_LAYER:
		layer = new RGBA2YUVLayerCuda();
		break;
	case OEIP_RESIZE_LAYER:
		layer = new ResizeLayerCuda();
		break;
	case OEIP_OPERATE_LAYER:
		layer = new OperateLayerCuda();
		break;
	case OEIP_BLEND_LAYER:
		layer = new BlendLayerCuda();
		break;
	case OEIP_DARKNET_LAYER:
		layer = new DarknetLayerCuda();
		break;
	case OEIP_GRABCUT_LAYER:
		layer = new GrabcutLayerCuda();
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
		std::string message = "cuda not support this layer.";
		logMessage(OEIP_WARN, message.c_str());
	}
	return layer;
}

void ImageProcessCuda::onRunLayers() {
	for (auto layer : layers) {
		if (layer->bDisable || layer->bDisableList)
			continue;
		layer->runLayer();
	}
}

void ImageProcessCuda::getGpuMat(int32_t layerIndex, cv::cuda::GpuMat& gpuMat, int32_t inIndex) {
	auto layer = std::dynamic_pointer_cast<LayerCuda>(layers[layerIndex]);
	if (layer->onlyDraw())
		gpuMat = layer->inMats[inIndex];
	else
		gpuMat = layer->outMats[inIndex];
}

bool bCanLoad() {
	int32_t count = cv::cuda::getCudaEnabledDeviceCount();
	if (count > 0) {
		int32_t deviceId = 0;
		auto device = cv::cuda::DeviceInfo::DeviceInfo(deviceId);
		bool compation = device.isCompatible();
		if (!compation) {
			//cv::cuda::setDevice(deviceId);
		}
		return compation;
	}
	return false;
}

void registerFactory() {
	registerFactory(new ImageProcessCudaFactory(), OeipGpgpuType::OEIP_CUDA, "image process cuda");
}
