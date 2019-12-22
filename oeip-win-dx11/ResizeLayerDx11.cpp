#include "ResizeLayerDx11.h"

ResizeLayerDx11::ResizeLayerDx11() {
	computeShader = std::make_unique<Dx11ComputeShader>();
	computeShader->setCS(110, modeName, rctype);
}

ResizeLayerDx11::~ResizeLayerDx11() {
}

void ResizeLayerDx11::onParametChange(ResizeParamet oldT) {
	if (!bBufferInit)
		return;
	dx11->resetLayers();
}

bool ResizeLayerDx11::initHlsl() {
	std::string msg = layerParamet.bLinear ? "1" : "0";
	// "SIZE_X", OEIP_CS_SIZE_XSTR, "SIZE_Y",OEIP_CS_SIZE_YSTR
	D3D_SHADER_MACRO defines[] = { "OEIP_LINE_SAMPLER",msg.c_str(),nullptr,nullptr };
	return computeShader->initResource(dx11->device, defines, dx11->includeShader);
}

void ResizeLayerDx11::onInitLayer() {
	threadSizeX = layerParamet.width;
	threadSizeY = layerParamet.height;
	outConnects[0].width = threadSizeX;
	outConnects[0].height = threadSizeY;
	groupSize.X = divUp(threadSizeX, sizeX);
	groupSize.Y = divUp(threadSizeY, sizeY);
	groupSize.Z = 1;
	LayerDx11::onInitLayer();
}


