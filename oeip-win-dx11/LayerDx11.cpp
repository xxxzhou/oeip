#include "LayerDx11.h"
#include <math.h>

LayerDx11::LayerDx11()
{
	inSRVs.resize(inCount);
	outTextures.resize(outCount);
	outUAVs.resize(outCount);
	for (int32_t i = 0; i < outCount; i++) {
		outTextures[i] = std::make_shared< Dx11Texture>();
	}
	constBuffer = std::make_unique<Dx11Constant>();
	computeShader = std::make_unique< Dx11ComputeShader>();
}

bool LayerDx11::onInitBuffer()
{
	bool bInit = true;
	if (layerType != OEIP_OUTPUT_LAYER) {
		for (int32_t i = 0; i < outCount; i++) {
			auto dxFormat = getDxFormat(outConnects[i].dataType);
			outTextures[i]->setTextureSize(outConnects[i].width, outConnects[i].height, dxFormat);
			bInit &= outTextures[i]->initResource(dx11->device);
		}
	}
	constBuffer->setBufferSize(sizeof(InputConstant));
	constBuffer->initResource(dx11->device);
	constBuffer->cpuData = (uint8_t*)&inputConstant;
	updateCBuffer();
	if (layerType != OEIP_INPUT_LAYER) {
		for (int32_t i = 0; i < inCount; i++) {
			int32_t layerIndex = forwardLayerIndexs[i];
			int32_t inIndex = forwardOutIndexs[i];
			std::shared_ptr<Dx11Texture> texture;
			dx11->getTexture(layerIndex, texture, inIndex);
			inSRVs[i] = texture->srvView;
		}
	}
	if (layerType != OEIP_OUTPUT_LAYER) {
		for (int32_t i = 0; i < outCount; i++) {
			outUAVs[i] = outTextures[i]->uavView;
		}
	}
	return bInit;
}

bool LayerDx11::updateCBuffer()
{
	auto dxFormat = getDxFormat(selfConnects[0].dataType);
	inputConstant.width = threadSizeX;
	inputConstant.height = threadSizeY;
	inputConstant.elementCount = OEIP_CV_MAT_CN(selfConnects[0].dataType);//  getElementChannel(selfConnects[0].dataType);
	inputConstant.elementByte = OEIP_CV_ELEM_SIZE1(selfConnects[0].dataType);// getElementSize(selfConnects[0].dataType);
	constBuffer->updateResource(dx11->ctx);
	return true;
}


void LayerDx11::setImageProcess(ImageProcess* ipdx11)
{
	imageProcess = ipdx11;
	dx11 = dynamic_cast<ImageProcessDx11*>(ipdx11);
}

void LayerDx11::onInitLayer()
{
	bInitHlsl = initHlsl();
	if (!bInitHlsl) {
		std::string message = "check " + layerName + " in:" + std::to_string(layerIndex) + " " + getLayerName(layerType) + "create hlsl fail.";
		logMessage(OEIP_ERROR, message.c_str());
	}
}

void LayerDx11::onRunLayer()
{
	if (bInitHlsl) {
		computeShader->runCS(dx11->ctx, groupSize, inSRVs, outUAVs, { constBuffer->buffer });
	}
}


