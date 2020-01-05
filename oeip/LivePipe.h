#pragma once
#include "OeipExport.h"

class OEIPDLL_EXPORT LivePipe
{
public:
	LivePipe(OeipGpgpuType gpuType);
	~LivePipe();
private:
	int32_t pipeId = -1;
	int32_t inputIndex = -1;
	int32_t yuv2rgba = -1;
	int32_t outMap = -1;
	int32_t resizeIndex = -1;
	int32_t outIndex = -1;
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
	void setVideoFormat(OeipYUVFMT yuvFmt, int32_t width, int32_t height);
	void runVideoPipe(int32_t layerIndex, uint8_t* data);
};

