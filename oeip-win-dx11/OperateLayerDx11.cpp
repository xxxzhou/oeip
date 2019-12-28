#include "OperateLayerDx11.h"



OperateLayerDx11::OperateLayerDx11() {
	computeShader = std::make_unique<Dx11ComputeShader>();
	computeShader->setCS(113, modeName, rctype);
}

OperateLayerDx11::~OperateLayerDx11() {

}

void OperateLayerDx11::onParametChange(OperateParamet oldT) {
	if (!bBufferInit)
		return;
	updateCBuffer();
}

bool OperateLayerDx11::initHlsl() {
	return computeShader->initResource(dx11->device, nullptr, dx11->includeShader);
}

bool OperateLayerDx11::onInitCBuffer() {
	constBuffer->setBufferSize(sizeof(OperateParamet));
	constBuffer->initResource(dx11->device);
	constBuffer->cpuData = (uint8_t*)&operateConstant;
	return true;
}

bool OperateLayerDx11::updateCBuffer() {
	auto dxFormat = getDxFormat(selfConnects[0].dataType);
	operateConstant.inputConstant.width = threadSizeX;
	operateConstant.inputConstant.height = threadSizeY;
	operateConstant.inputConstant.elementCount = OEIP_CV_MAT_CN(selfConnects[0].dataType);
	operateConstant.inputConstant.elementByte = OEIP_CV_ELEM_SIZE1(selfConnects[0].dataType);
	operateConstant.operateParamet = layerParamet;
	operateConstant.operateParamet.gamma = 1.f / layerParamet.gamma;
	constBuffer->updateResource(dx11->ctx);
	return true;
}
