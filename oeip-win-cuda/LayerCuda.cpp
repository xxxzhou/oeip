#include "LayerCuda.h"

LayerCuda::~LayerCuda() {
}

void LayerCuda::setImageProcess(ImageProcess * process) {
	imageProcess = process;
	ipCuda = dynamic_cast<ImageProcessCuda*>(process);
}

bool LayerCuda::onInitBuffer() {
	bool bInit = true;
	if (layerType != OEIP_OUTPUT_LAYER) {
		for (int32_t i = 0; i < outCount; i++) {
			outMats[i].create(outConnects[i].height, outConnects[i].width, outConnects[i].dataType);
		}
	}
	if (layerType != OEIP_INPUT_LAYER) {
		for (int32_t i = 0; i < inCount; i++) {
			int32_t layerIndex = forwardLayerIndexs[i];
			int32_t inIndex = forwardOutIndexs[i];
			ipCuda->getGpuMat(layerIndex, inMats[i], inIndex);
		}
	}
	return bInit;
}
