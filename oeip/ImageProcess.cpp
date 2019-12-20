#include "ImageProcess.h"

bool ImageProcess::initLayers()
{
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	bInitLayer = false;
	bInitBuffer = false;
	for (auto layer : layers) {
		if (!layer->initLayer()) {
			bInitLayer = false;
			return false;
		}
	}
	bInitLayer = onInitLayers();
	if (bInitLayer) {
		for (auto layer : layers) {
			if (!layer->initBuffer()) {
				bInitBuffer = false;
				return false;
			}
		}
		bInitBuffer = true;
	}
	return bInitLayer && bInitBuffer;
}

void ImageProcess::runLayers()
{
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	if (!bInitLayer) {
		initLayers();
	}
	if (bInitLayer && bInitBuffer) {
		onRunLayers();
	}
}

int32_t ImageProcess::addLayer(const std::string& name, OeipLayerType layerType)
{
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
	layers.push_back(layer);
	return layer->layerIndex;
}

void ImageProcess::updateInput(int32_t layerIndex, uint8_t* data, int32_t intputIndex)
{
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	auto layer = layers[layerIndex];
	BaseInputLayer* inputLayer = dynamic_cast<BaseInputLayer*>(layer.get());
	if (inputLayer)
		inputLayer->inputCpuData(data, intputIndex);
}

void ImageProcess::outputData(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t dataType)
{
	//这个回调会发给用户,如果加锁需要保证用户操作安全不会引起锁的问题
	if (onProcessEvent) {
		onProcessEvent(layerIndex, data, width, height, dataType);
	}
}

void ImageProcess::setOutputGpuTex(int32_t layerIndex, void* device, void* tex, int32_t outputIndex)
{
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	auto layer = layers[layerIndex];
	BaseOutputLayer* outputLayer = dynamic_cast<BaseOutputLayer*>(layer.get());
	if (outputLayer)
		outputLayer->outputGpuTex(device, tex, outputIndex);
}

int32_t ImageProcess::findLayer(const std::string& name)
{
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	int32_t index = -1;
	for (auto layer : layers) {
		index += 1;
		if (layer->layerName.compare(name)) {
			return index;
		}
	}
	//if (index < 0) {
	//	std::string message = "not find layer:" + name;
	//	logMessage(OEIP_ERROR, message.c_str());
	//	return index;
	//}
	return index;
}

void ImageProcess::getLayerOutConnect(int32_t layerIndex, LayerConnect& outConnect, int32_t outIndex)
{
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	layers[layerIndex]->getOutputSize(outConnect, outIndex);
}

void ImageProcess::getLayerInConnect(int32_t layerIndex, LayerConnect& inConnect, int32_t inIndex)
{
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	layers[layerIndex]->getInputSize(inConnect, inIndex);
}

void ImageProcess::setInput(int32_t layerIndex, int32_t width, int32_t height, int32_t dataType, int32_t intputIndex)
{
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	LayerConnect lc = {};
	layers[layerIndex]->getInputSize(lc, intputIndex);
	if (lc.width != width || lc.height != height || lc.dataType != lc.dataType) {
		layers[layerIndex]->setInputSize(width, height, dataType, intputIndex);
		layers[layerIndex]->setOutputSize(width, height, dataType, intputIndex);
		bInitLayer = false;
	}
}

void registerFactory(ObjectFactory<ImageProcess>* factory, int32_t type, std::string name)
{
	PluginManager<ImageProcess>::getInstance().registerFactory(factory, type, name);
}
