#include "ImageProcess.h"

#define OEIP_CHECKPIPEINDEXVOID \
	if (layerIndex<0 || layerIndex > layers.size())\
		return ;\
	auto layer = layers[layerIndex];
#define OEIP_CHECKPIPEINDEXBOOL \
	if (layerIndex<0 || layerIndex > layers.size())\
		return false;\
	auto layer = layers[layerIndex];

#define OEIP_PIPE_LOCK std::unique_lock <std::recursive_mutex> mtx_locker(mtx, std::try_to_lock)
//#define OEIP_PIPE_LOCK std::lock_guard<std::recursive_mutex> mtx_locker(mtx)

bool ImageProcess::initLayers() {
	OEIP_PIPE_LOCK;
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

bool ImageProcess::runLayers() {
	OEIP_PIPE_LOCK;
	if (!bInitLayers && bResetLayers) {
		logMessage(OEIP_INFO, "init layers.");
		initLayers();
		bResetLayers = false;
	}
	if (bInitLayers && bInitBuffers) {
		onRunLayers();
		return true;
	}
	return false;
}

int32_t ImageProcess::addLayer(const std::string& name, OeipLayerType layerType, const void* paramet) {
	OEIP_PIPE_LOCK;
	for (auto clayer : layers) {
		if (clayer->layerName.compare(name) == 0) {
			std::string message = "in:" + std::to_string(clayer->layerIndex) + " " + name + " have name same.";
			logMessage(OEIP_WARN, message.c_str());
		}
	}
	std::shared_ptr<BaseLayer> layer(onAddLayer(layerType));
	if (layer == nullptr) {
		std::string message = name + " no create.";
		logMessage(OEIP_ERROR, message.c_str());
		return -1;
	}
	layer->layerName = name;
	layer->layerIndex = layers.size();
	if (paramet) {
		layer->updateParamet(paramet);
	}
	layers.push_back(layer);
	resetLayers();
	return layer->layerIndex;
}

void ImageProcess::connectLayer(int32_t layerIndex, const std::string& forwardName, int32_t inputIndex, int32_t selfIndex) {
	OEIP_PIPE_LOCK;
	OEIP_CHECKPIPEINDEXVOID;
	layer->forwardNames[selfIndex] = forwardName;
	layer->forwardOutIndexs[selfIndex] = inputIndex;
}

void ImageProcess::connectLayer(int32_t layerIndex, int32_t forwardIndex, int32_t inputIndex, int32_t selfIndex) {
	OEIP_PIPE_LOCK;
	OEIP_CHECKPIPEINDEXVOID;
	layer->forwardLayerIndexs[selfIndex] = forwardIndex;
	layer->forwardOutIndexs[selfIndex] = inputIndex;
}

void ImageProcess::updateInput(int32_t layerIndex, uint8_t* data, int32_t intputIndex) {
	OEIP_PIPE_LOCK;
	OEIP_CHECKPIPEINDEXVOID;
	BaseInputLayer* inputLayer = dynamic_cast<BaseInputLayer*>(layer.get());
	if (inputLayer)
		inputLayer->inputCpuData(data, intputIndex);
}

void ImageProcess::outputData(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t dataType) {
	//这个回调会发给用户,如果加锁需要保证用户操作安全不会引起锁的问题
	{
		OEIP_PIPE_LOCK;
		OEIP_CHECKPIPEINDEXVOID;
	}
	if (onProcessEvent) {
		onProcessEvent(layerIndex, data, width, height, dataType);
	}
}

void ImageProcess::setInputGpuTex(int32_t layerIndex, void* device, void* tex, int32_t inputIndex) {
	OEIP_PIPE_LOCK;
	OEIP_CHECKPIPEINDEXVOID;
	BaseInputLayer* inputLayer = dynamic_cast<BaseInputLayer*>(layer.get());
	if (inputLayer)
		inputLayer->inputGpuTex(device, tex, inputIndex);
}

void ImageProcess::setOutputGpuTex(int32_t layerIndex, void* device, void* tex, int32_t outputIndex) {
	OEIP_PIPE_LOCK;
	OEIP_CHECKPIPEINDEXVOID;
	BaseOutputLayer* outputLayer = dynamic_cast<BaseOutputLayer*>(layer.get());
	if (outputLayer)
		outputLayer->outputGpuTex(device, tex, outputIndex);
}

bool ImageProcess::updateLayer(int32_t layerIndex, const void* paramet) {
	OEIP_PIPE_LOCK;
	OEIP_CHECKPIPEINDEXBOOL;
	layer->updateParamet(paramet);
	return true;
}

void ImageProcess::setEnableLayer(int32_t layerIndex, bool bEnable) {
	OEIP_PIPE_LOCK;
	OEIP_CHECKPIPEINDEXVOID;
	layer->bDisable = !bEnable;
	bInitLayers = false;
}

bool ImageProcess::getEnableLayer(int32_t layerIndex) {
	OEIP_PIPE_LOCK;
	OEIP_CHECKPIPEINDEXBOOL;
	return !layer->bDisable;
}

bool ImageProcess::getConnecEnable(int32_t layerIndex) {
	OEIP_PIPE_LOCK;
	OEIP_CHECKPIPEINDEXBOOL;
	//原则上,上层不能是输出层，输出层不会给下层数据,故自动向上一层
	return !layer->bDisable && layer->layerType != OEIP_OUTPUT_LAYER;
}

void ImageProcess::setEnableLayerList(int32_t layerIndex, bool bEnable) {
	OEIP_PIPE_LOCK;
	OEIP_CHECKPIPEINDEXVOID;
	layer->bDListChange = true;
	layer->bDisableList = !bEnable;
	bInitLayers = false;
}

bool ImageProcess::getEnableLayerList(int32_t layerIndex, bool& bDListChange) {
	OEIP_PIPE_LOCK;
	OEIP_CHECKPIPEINDEXBOOL;
	bDListChange = layer->bDListChange;
	return !layer->bDisableList;
}

OeipLayerType ImageProcess::getLayerType(int32_t layerIndex) {
	OEIP_PIPE_LOCK;
	if (layerIndex<0 || layerIndex > layers.size())
		return OEIP_NONE_LAYER;
	auto layer = layers[layerIndex];
	return layer->layerType;
}

void ImageProcess::closePipe() {
	OEIP_PIPE_LOCK;
	layers.clear();
}

bool ImageProcess::emptyPipe() {
	OEIP_PIPE_LOCK;
	return layers.size() <= 0;
}

int32_t ImageProcess::findLayer(const std::string& name) {
	OEIP_PIPE_LOCK;
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
	OEIP_PIPE_LOCK;
	OEIP_CHECKPIPEINDEXVOID;
	layer->getOutputSize(outConnect, outIndex);
}

void ImageProcess::getLayerInConnect(int32_t layerIndex, LayerConnect& inConnect, int32_t inIndex) {
	OEIP_PIPE_LOCK;
	OEIP_CHECKPIPEINDEXVOID;
	layer->getInputSize(inConnect, inIndex);
}

void ImageProcess::setInput(int32_t layerIndex, int32_t width, int32_t height, int32_t dataType, int32_t intputIndex) {
	OEIP_PIPE_LOCK;
	OEIP_CHECKPIPEINDEXVOID;
	LayerConnect lc = {};
	layer->getInputSize(lc, intputIndex);
	if (lc.width != width || lc.height != height || lc.dataType != lc.dataType) {
		layer->setInputSize(width, height, dataType, intputIndex);
		layer->setOutputSize(width, height, dataType, intputIndex);
		resetLayers();
	}
}

