#include "LivePipe.h"

LivePipe::LivePipe(OeipGpgpuType gpuType) {
	pipeId = initPipe(gpuType);
	inputIndex = addPiepLayer(pipeId, "input", OEIP_INPUT_LAYER);
	yuv2rgba = addPiepLayer(pipeId, "yuv2rgba", OEIP_YUV2RGBA_LAYER);
	//resizeIndex = addPiepLayer(pipeId, "resize", OEIP_RESIZE_LAYER);
	outMap = addPiepLayer(pipeId, "out map channel", OEIP_MAPCHANNEL_LAYER);
	outIndex = addPiepLayer(pipeId, "output", OEIP_OUTPUT_LAYER);

	MapChannelParamet mp = {};
	mp.red = 2;
	mp.blue = 0;
	updatePipeParamet(pipeId, outMap, &mp);

	//ResizeParamet rp = {};
	//rp.width = 640;
	//rp.height = 480;
	//updatePipeParamet(pipeId, resizeIndex, &rp);
}

LivePipe::~LivePipe() {

}

void LivePipe::setVideoFormat(OeipYUVFMT yuvFmt, int32_t width, int32_t height) {
	int32_t dataType = OEIP_CV_8UC1;
	int32_t inputWidth = width;
	int32_t inputHeight = height;
	yp.yuvType = yuvFmt;
	updatePipeParamet(pipeId, yuv2rgba, &yp);
	if (yp.yuvType == OEIP_YUVFMT_YUV420SP || yp.yuvType == OEIP_YUVFMT_YUY2P || yp.yuvType == OEIP_YUVFMT_YUV420P) {
		dataType = OEIP_CV_8UC1;
		inputHeight = height * 3 / 2;
		if (yp.yuvType == OEIP_YUVFMT_YUY2P) {
			inputHeight = height * 2;
		}
	}
	setPipeInput(pipeId, inputIndex, inputWidth, inputHeight, dataType);
}

void LivePipe::runVideoPipe(int32_t layerIndex, uint8_t* data) {
	updatePipeInput(pipeId, inputIndex, data);
	runPipe(pipeId);
}
