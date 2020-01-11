#include "ImageProcess.h"

#define OEIP_CHECKPIPEINDEXVOID \
	if (layerIndex<0 || layerIndex > layers.size())\
		return ;\
	auto layer = layers[layerIndex];
#define OEIP_CHECKPIPEINDEXBOOL \
	if (layerIndex<0 || layerIndex > layers.size())\
		return false;\
	auto layer = layers[layerIndex];


bool ImageProcess::initLayers() {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	bInitLayers = false;
	bInitBuffers = false;
	for (auto layer : layers) {
		if (layer->bDisable || layer->bDisableList)
			continue;
		if (!layer->initLayer()) {
			bInitLayers = false;
			return false;
		}
	}
	bInitLayers = onInitLayers();
	if (bInitLayers) {
		for (auto layer : layers) {
			//状态重置
			layer->bDListChange = false;
			if (layer->bDisable || layer->bDisableList)
				continue;
			if (!layer->initBuffer()) {
				bInitBuffers = false;
				return false;
			}
		}
		bInitBuffers = true;
	}
	return bInitLayers && bInitBuffers;
}

void ImageProcess::runLayers() {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	if (!bInitLayers) {
		logMessage(OEIP_INFO, "init layers.");
		initLayers();
	}
	if (bInitLayers && bInitBuffers) {
		onRunLayers();
	}
}

int32_t ImageProcess::addLayer(const std::string& name, OeipLayerType layerType, const void* paramet) {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	for (auto clayer : layers) {
		if (clayer->layerName.compare(name) == 0) {
			std::string message = "in:" + std::to_string(clayer->layerIndex) + " " + getLayerName(layerType) + " name same.";
			logMessage(OEIP_WARN, message.c_str());
		}
	}
	std::shared_ptr<BaseLayer> layer(onAddLayer(layerType));
	if (layer == nullptr) {
		std::string message = getLayerName(layerType) + " no create.";
		logMessage(OEIP_ERROR, message.c_str());
		return -1;
	}
	layer->layerName = name;
	layer->layerIndex = layers.size();
	if (paramet) {
		layer->updateParamet(paramet);
	}
	layers.push_back(layer);
	return layer->layerIndex;
}

void ImageProcess::connectLayer(int32_t layerIndex, const std::string& forwardName, int32_t inputIndex, int32_t selfIndex) {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	OEIP_CHECKPIPEINDEXVOID;
	layer->forwardNames[selfIndex] = forwardName;
	layer->forwardOutIndexs[selfIndex] = inputIndex;
}

void ImageProcess::connectLayer(int32_t layerIndex, int32_t forwardIndex, int32_t inputIndex, int32_t selfIndex) {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	OEIP_CHECKPIPEINDEXVOID;
	layer->forwardLayerIndexs[selfIndex] = forwardIndex;
	layer->forwardOutIndexs[selfIndex] = inputIndex;
}

void ImageProcess::updateInput(int32_t layerIndex, uint8_t* data, int32_t intputIndex) {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	OEIP_CHECKPIPEINDEXVOID;
	BaseInputLayer* inputLayer = dynamic_cast<BaseInputLayer*>(layer.get());
	if (inputLayer)
		inputLayer->inputCpuData(data, intputIndex);
}

void ImageProcess::outputData(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t dataType) {
	//这个回调会发给用户,如果加锁需要保证用户操作安全不会引起锁的问题
	if (onProcessEvent) {
		onProcessEvent(layerIndex, data, width, height, dataType);
	}
}

void ImageProcess::setInputGpuTex(int32_t layerIndex, void* device, void* tex, int32_t inputIndex) {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	OEIP_CHECKPIPEINDEXVOID;
	BaseInputLayer* inputLayer = dynamic_cast<BaseInputLayer*>(layer.get());
	if (inputLayer)
		inputLayer->inputGpuTex(device, tex, inputIndex);
}

void ImageProcess::setOutputGpuTex(int32_t layerIndex, void* device, void* tex, int32_t outputIndex) {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	OEIP_CHECKPIPEINDEXVOID;
	BaseOutputLayer* outputLayer = dynamic_cast<BaseOutputLayer*>(layer.get());
	if (outputLayer)
		outputLayer->outputGpuTex(device, tex, outputIndex);
}

bool ImageProcess::updateLayer(int32_t layerIndex, const void* paramet) {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	OEIP_CHECKPIPEINDEXBOOL;
	layer->updateParamet(paramet);
	return true;
}

void ImageProcess::setEnableLayer(int32_t layerIndex, bool bEnable) {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	OEIP_CHECKPIPEINDEXVOID;
	layer->bDisable = !bEnable;
	bInitLayers = false;
}

bool ImageProcess::getEnableLayer(int32_t layerIndex) {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	OEIP_CHECKPIPEINDEXBOOL;
	return !layer->bDisable;
}

void ImageProcess::setEnableLayerList(int32_t layerIndex, bool bEnable) {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	OEIP_CHECKPIPEINDEXVOID;
	layer->bDListChange = true;
	layer->bDisableList = !bEnable;
	bInitLayers = false;
}

bool ImageProcess::getEnableLayerList(int32_t layerIndex, bool& bDListChange) {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	OEIP_CHECKPIPEINDEXBOOL;
	bDListChange = layer->bDListChange;
	return !layer->bDisableList;
}

int32_t ImageProcess::findLayer(const std::string& name) {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	int32_t index = 0;
	for (auto layer : layers) {
		if (layer->layerName.compare(name) == 0) {
			return index;
		}
		index += 1;
	}
	if (index >= layers.size())
		return -1;
	return index;
}

void ImageProcess::getLayerOutConnect(int32_t layerIndex, LayerConnect& outConnect, int32_t outIndex) {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	OEIP_CHECKPIPEINDEXVOID;
	layer->getOutputSize(outConnect, outIndex);
}

void ImageProcess::getLayerInConnect(int32_t layerIndex, LayerConnect& inConnect, int32_t inIndex) {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	OEIP_CHECKPIPEINDEXVOID;
	layer->getInputSize(inConnect, inIndex);
}

void ImageProcess::setInput(int32_t layerIndex, int32_t width, int32_t height, int32_t dataType, int32_t intputIndex) {
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	OEIP_CHECKPIPEINDEXVOID;
	LayerConnect lc = {};
	layer->getInputSize(lc, intputIndex);
	if (lc.width != width || lc.height != height || lc.dataType != lc.dataType) {
		layer->setInputSize(width, height, dataType, intputIndex);
		layer->setOutputSize(width, height, dataType, intputIndex);
		bInitLayers = false;
	}
}

