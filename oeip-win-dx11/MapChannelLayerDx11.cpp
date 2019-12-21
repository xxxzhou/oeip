#include "MapChannelLayerDx11.h"

MapChannelLayerDx11::MapChannelLayerDx11()
{
	computeShader = std::make_unique<Dx11ComputeShader>();
	computeShader->setCS(109, modeName, rctype);
}

MapChannelLayerDx11::~MapChannelLayerDx11()
{
}

void MapChannelLayerDx11::onParametChange(MapChannelParamet oldT)
{
	if (!bBufferInit)
		return;
	updateCBuffer();
}

bool MapChannelLayerDx11::initHlsl()
{
	return computeShader->initResource(dx11->device, nullptr, dx11->includeShader);
}

bool MapChannelLayerDx11::onInitCBuffer()
{
	constBuffer->setBufferSize(sizeof(MapChannelConstant));
	constBuffer->initResource(dx11->device);
	constBuffer->cpuData = (uint8_t*)&mapConstant;
	return true;
}

bool MapChannelLayerDx11::updateCBuffer()
{
	auto dxFormat = getDxFormat(selfConnects[0].dataType);
	mapConstant.inputConstant.width = threadSizeX;
	mapConstant.inputConstant.height = threadSizeY;
	mapConstant.inputConstant.elementCount = OEIP_CV_MAT_CN(selfConnects[0].dataType);
	mapConstant.inputConstant.elementByte = OEIP_CV_ELEM_SIZE1(selfConnects[0].dataType);
	mapConstant.mapChannelParamet = layerParamet;
	constBuffer->updateResource(dx11->ctx);
	return false;
}
