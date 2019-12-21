#include "BaseLayer.h"
#include "ImageProcess.h"

BaseLayer::BaseLayer(int32_t inSize, int32_t outSize)
{
	inCount = inSize;
	outCount = outSize;
	forwardNames.resize(inSize);
	forwardOutIndexs.resize(inSize);
	forwardLayerIndexs.resize(inSize);
	selfConnects.resize(inSize);
	outConnects.resize(outSize);
	//默认用OEIP_CV_8UC4来传递各层数据
	for (uint32_t i = 0; i < inSize; i++) {
		selfConnects[i].dataType = OEIP_CV_8UC4;
	}
	for (uint32_t i = 0; i < outSize; i++) {
		outConnects[i].dataType = OEIP_CV_8UC4;
	}
}

bool BaseLayer::initLayer()
{
	std::string inputMeg = layerType == OEIP_INPUT_LAYER ? " input layer " : " forward layer ";
	std::string layerMeg = "check " + layerName + " in:" + std::to_string(layerIndex) + " " + getLayerName(layerType) + inputMeg;
	for (uint32_t i = 0; i < inCount; i++) {
		LayerConnect lc = selfConnects[i];
		if (layerType <= 0) {
			std::string message = layerMeg + "layer type invalid.";
			logMessage(OEIP_ERROR, message.c_str());
			return false;
		}
		if (layerType != OEIP_INPUT_LAYER) {
			if (forwardNames[i].empty()) {
				forwardLayerIndexs[i] = layerIndex - 1;
			}
			else {
				forwardLayerIndexs[i] = imageProcess->findLayer(forwardNames[i]);
			}
			//bEnableList会关闭所有连接到当前层的后续层
			bool bEnableList = imageProcess->getEnableLayerList(forwardLayerIndexs[i]);
			if (!bEnableList) {
				this->bDisableList = true;
				continue;
			}
			bool bEnable = imageProcess->getEnableLayer(forwardLayerIndexs[i]);
			while (!bEnable) {
				//自动向上一层
				forwardLayerIndexs[i] -= 1;
				if (forwardLayerIndexs[i] < 0)
					break;
				bEnable = imageProcess->getEnableLayer(forwardLayerIndexs[i]);
			}
			if (forwardLayerIndexs[i] < 0) {
				std::string message = layerMeg + "out of range.";
				logMessage(OEIP_ERROR, message.c_str());
				return false;
			}
			imageProcess->getLayerOutConnect(forwardLayerIndexs[i], lc, forwardOutIndexs[i]);
			if (lc.dataType != selfConnects[i].dataType) {
				std::string message = layerMeg + "no match current layer element byte size.";
				logMessage(OEIP_ERROR, message.c_str());
				return false;
			}
		}
		if (lc.width <= 0) {
			std::string message = layerMeg + "width invalid value.";
			logMessage(OEIP_ERROR, message.c_str());
			return false;
		}
		if (lc.height <= 0) {
			std::string message = layerMeg + "height invalid value.";
			logMessage(OEIP_ERROR, message.c_str());
			return false;
		}
		if (layerType == OEIP_INPUT_LAYER)
			break;
		selfConnects[i].width = lc.width;
		selfConnects[i].height = lc.height;
		if (layerType == OEIP_OUTPUT_LAYER)
			selfConnects[i].dataType = lc.dataType;
	}
	threadSizeX = selfConnects[0].width;
	threadSizeY = selfConnects[0].height;
	groupSize.X = divUp(threadSizeX, sizeX);
	groupSize.Y = divUp(threadSizeY, sizeY);
	groupSize.Z = 1;
	for (uint32_t i = 0; i < outCount; i++) {
		outConnects[i].width = threadSizeX;
		outConnects[i].height = threadSizeY;
	}
	onInitLayer();
	for (uint32_t i = 0; i < inCount; i++) {
		onInitLayer(i);
	}
	return true;
}

bool BaseLayer::initBuffer()
{
	bBufferInit = onInitBuffer();
	return bBufferInit;
}

void BaseLayer::runLayer()
{
	if (!bBufferInit)
		return;
	onRunLayer();
}

void BaseLayer::updateParamet(const void* paramet)
{
	if (layerType == OEIP_INPUT_LAYER) {
		using ParametType = LayerParamet<OEIP_INPUT_LAYER, AllLayerParamet>::ParametType;
		auto xlayer = dynamic_cast<BaseLayerTemplate<ParametType>*>(this);
		xlayer->updateParamet(paramet);
	}
	else if (layerType == OEIP_YUV2RGBA_LAYER) {
		using ParametType = LayerParamet<OEIP_YUV2RGBA_LAYER, AllLayerParamet>::ParametType;
		auto xlayer = dynamic_cast<BaseLayerTemplate<ParametType>*>(this);
		xlayer->updateParamet(paramet);
	}
	else if (layerType == OEIP_OUTPUT_LAYER) {
		using ParametType = LayerParamet<OEIP_OUTPUT_LAYER, AllLayerParamet>::ParametType;
		auto xlayer = dynamic_cast<BaseLayerTemplate<ParametType>*>(this);
		xlayer->updateParamet(paramet);
	}
}

void BaseLayer::setInputSize(int32_t width, int32_t height, int32_t dataType, int32_t intputIndex)
{
	if (intputIndex >= inCount)
		return;
	selfConnects[intputIndex].width = width;
	selfConnects[intputIndex].height = height;
	selfConnects[intputIndex].dataType = dataType;
}

void BaseLayer::setOutputSize(int32_t width, int32_t height, int32_t dataType, int32_t outputIndex)
{
	if (outputIndex >= outCount)
		return;
	outConnects[outputIndex].width = width;
	outConnects[outputIndex].height = height;
	outConnects[outputIndex].dataType = dataType;
}

void BaseLayer::getInputSize(LayerConnect& inConnect, int32_t inputIndex)
{
	if (inputIndex >= inCount)
		return;
	inConnect = selfConnects[inputIndex];
}

void BaseLayer::getOutputSize(LayerConnect& outConnect, int32_t outputIndex)
{
	if (outputIndex >= outCount)
		return;
	outConnect = outConnects[outputIndex];
}

void BaseLayer::setForwardLayer(std::string forwardName, int32_t outputIndex, int32_t inputIndex)
{
	if (inputIndex >= inCount)
		return;
	forwardNames[inputIndex] = forwardName;
	forwardOutIndexs[inputIndex] = outputIndex;
}
