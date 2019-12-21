#include "VideoPipe.h"

VideoPipe::VideoPipe()
{
	pipeId = initPipe(OEIP_DX11);
	inputIndex = addPiepLayer(pipeId, "input", OEIP_INPUT_LAYER);
	yuv2rgba = addPiepLayer(pipeId, "yuv2rgba", OEIP_YUV2RGBA_LAYER);
	mapChannel = addPiepLayer(pipeId, "map channel", OEIP_MAPCHANNEL_LAYER);
	outMap = addPiepLayer(pipeId, "out map channel", OEIP_MAPCHANNEL_LAYER);
	outIndex = addPiepLayer(pipeId, "output", OEIP_OUTPUT_LAYER);
	//mapChannel”Îyuv2rgbaÕ¨º∂
	connectLayer(pipeId, mapChannel, "input");
}


VideoPipe::~VideoPipe()
{
}

void VideoPipe::setVideoFormat(OeipVideoType videoType, int32_t width, int32_t height)
{
	yp.yuvType = getVideoYUV(videoType);
	inputWidth = width;
	inputHeight = height;
	setEnableLayer(pipeId, yuv2rgba, true);
	setEnableLayer(pipeId, mapChannel, false);
	if (yp.yuvType == OEIP_YUVFMT_OTHER) {
		setEnableLayer(pipeId, yuv2rgba, false);
		if (videoType == OEIP_VIDEO_ARGB32) {
			setEnableLayer(pipeId, mapChannel, true);
		}
		else if (videoType == OEIP_VIDEO_RGB24) {
			dataType = OEIP_CV_8UC3;
		}
	}
	else if (yp.yuvType == OEIP_YUVFMT_OTHER) {
		if (yp.yuvType == OEIP_VIDEO_NV12) {
			inputHeight = height * 3 / 2;
			dataType = OEIP_CV_8UC1;
		}
	}
	setPipeInput(pipeId, inputIndex, inputWidth, inputHeight, dataType);
}
