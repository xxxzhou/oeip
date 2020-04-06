#include "VideoPipe.h"

using namespace std::placeholders;

VideoPipe::VideoPipe(OeipGpgpuType gpuType) {
	pipeId = initPipe(gpuType);
	inputIndex = addPiepLayer(pipeId, "input", OEIP_INPUT_LAYER);
	yuv2rgba = addPiepLayer(pipeId, "yuv2rgba", OEIP_YUV2RGBA_LAYER);
	mapChannel = addPiepLayer(pipeId, "map channel", OEIP_MAPCHANNEL_LAYER);

	int32_t t1 = addPiepLayer(pipeId, "rgba2yuv test", OEIP_RGBA2YUV_LAYER);
	int32_t t2 = addPiepLayer(pipeId, "yuv2rgba test", OEIP_YUV2RGBA_LAYER);
	//resizeIndex = addPiepLayer(pipeId, "resize", OEIP_RESIZE_LAYER);	
	//auto x = addPiepLayer(pipeId, "blend", OEIP_BLEND_LAYER);
	outMap = addPiepLayer(pipeId, "out map channel", OEIP_MAPCHANNEL_LAYER);
	outIndex = addPiepLayer(pipeId, "output", OEIP_OUTPUT_LAYER);
	outYuv2Index = addPiepLayer(pipeId, "yuv output", OEIP_OUTPUT_LAYER);
	//mapChannel与yuv2rgba同级
	connectLayerName(pipeId, mapChannel, "input");
	connectLayerName(pipeId, outYuv2Index, "rgba2yuv test");

	RGBA2YUVParamet ry = {};
	YUV2RGBAParamet yr = {};
	ry.yuvType = OEIP_YUVFMT_YUV420P;
	yr.yuvType = OEIP_YUVFMT_YUV420P;
	updatePipeParamet(pipeId, t1, &ry);
	updatePipeParamet(pipeId, t2, &yr);
	MapChannelParamet mp = {};
	mp.red = 0;
	mp.blue = 2;
	updatePipeParamet(pipeId, outMap, &mp);

}


VideoPipe::~VideoPipe() {
}

void VideoPipe::setVideoFormat(OeipVideoType videoType, int32_t width, int32_t height) {
	yp.yuvType = getVideoYUV(videoType);
	updatePipeParamet(pipeId, yuv2rgba, &yp);

	inputWidth = width;
	inputHeight = height;
	setEnableLayer(pipeId, yuv2rgba, true);
	setEnableLayer(pipeId, mapChannel, false);
	setEnableLayer(pipeId, resizeIndex, false);
	if (yp.yuvType == OEIP_YUVFMT_OTHER) {
		setEnableLayer(pipeId, yuv2rgba, false);
		if (videoType == OEIP_VIDEO_ARGB32) {
			setEnableLayer(pipeId, mapChannel, true);
		}
		else if (videoType == OEIP_VIDEO_RGB24) {
			dataType = OEIP_CV_8UC3;
		}
	}
	else if (yp.yuvType == OEIP_YUVFMT_YUV420SP || yp.yuvType == OEIP_YUVFMT_YUY2P || yp.yuvType == OEIP_YUVFMT_YUV420P) {
		dataType = OEIP_CV_8UC1;
		inputHeight = height * 3 / 2;
		if (yp.yuvType == OEIP_YUVFMT_YUY2P) {
			inputHeight = height * 2;
		}
	}
	else if (yp.yuvType == OEIP_YUVFMT_YUY2I || yp.yuvType == OEIP_YUVFMT_YVYUI || yp.yuvType == OEIP_YUVFMT_UYVYI) {
		dataType = OEIP_CV_8UC4;
		inputWidth = width / 2;
	}
	setPipeInput(pipeId, inputIndex, inputWidth, inputHeight, dataType);
}

void VideoPipe::runVideoPipe(int32_t layerIndex, uint8_t* data) {
	updatePipeInput(pipeId, inputIndex, data);
	runPipe(pipeId);
}
