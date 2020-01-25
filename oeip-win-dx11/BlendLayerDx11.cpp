#include "BlendLayerDx11.h"


//BlendLayerDx11::BlendLayerDx11() {
//
//}

void BlendLayerDx11::onParametChange(BlendParamet oldT) {
	if (!bBufferInit)
		return;
	updateCBuffer();
}

bool BlendLayerDx11::initHlsl() {
	return computeShader->initResource(dx11->device, nullptr, dx11->includeShader);
}

bool BlendLayerDx11::onInitCBuffer() {
	constBuffer->setBufferSize(sizeof(BlendConstant));
	constBuffer->initResource(dx11->device);
	constBuffer->cpuData = (uint8_t*)&blendConstant;
	return true;
}

bool BlendLayerDx11::updateCBuffer() {
	auto dxFormat = getDxFormat(selfConnects[0].dataType);
	blendConstant.inputConstant.width = threadSizeX;
	blendConstant.inputConstant.height = threadSizeY;
	blendConstant.inputConstant.elementCount = OEIP_CV_MAT_CN(selfConnects[0].dataType);
	blendConstant.inputConstant.elementByte = OEIP_CV_ELEM_SIZE1(selfConnects[0].dataType);
	blendConstant.blendParamet = layerParamet;
	constBuffer->updateResource(dx11->ctx);
	return true;
}
