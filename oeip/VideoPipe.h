#pragma once
#include "OeipExport.h"

class OEIPDLL_EXPORT VideoPipe
{
public:
	VideoPipe(OeipGpgpuType gpuType);
	~VideoPipe();
private:
	int32_t pipeId = -1;
	int32_t inputIndex = -1;
	int32_t yuv2rgba = -1;
	int32_t mapChannel = -1;
	int32_t outMap = -1;
	int32_t outIndex = -1;
	int32_t outYuv2Index = -1;
	int32_t resizeIndex = -1;

	int32_t inputWidth = 0;
	int32_t inputHeight = 0;
	int32_t dataType = OEIP_CV_8UC4;
private:
	InputParamet ip = {};
	YUV2RGBAParamet yp = {};
	MapChannelParamet mp = {};
	OutputParamet op = {};
	ResizeParamet rp = {};
public:
	int32_t getPipeId() { return pipeId; };
	int32_t getOutputId() {
		return outIndex;
	}
	int32_t getOutYuvId() {
		return outYuv2Index;
	}
	void setVideoFormat(OeipVideoType videoType, int32_t width, int32_t height);
	void runVideoPipe(int32_t layerIndex, uint8_t* data);
};

