// Fill out your copyright notice in the Description page of Project Settings.

#include "VideoPipe.h"

VideoPipe::VideoPipe(OeipPipe* opipe) {
	this->pipe = opipe;
	inputIndex = pipe->AddLayer("input", OEIP_INPUT_LAYER);
	yuv2rgba = pipe->AddLayer("yuv2rgba", OEIP_YUV2RGBA_LAYER);
	mapChannel = pipe->AddLayer("map channel", OEIP_MAPCHANNEL_LAYER);

	int32_t t1 = pipe->AddLayer("rgba2yuv test", OEIP_RGBA2YUV_LAYER);
	int32_t t2 = pipe->AddLayer("yuv2rgba test", OEIP_YUV2RGBA_LAYER);
	resizeIndex = pipe->AddLayer("resize", OEIP_RESIZE_LAYER);
	//auto x = addPiepLayer(pipeId, "blend", OEIP_BLEND_LAYER);
	outMap = pipe->AddLayer("out map channel", OEIP_MAPCHANNEL_LAYER);
	outIndex = pipe->AddLayer("output", OEIP_OUTPUT_LAYER);
	outYuv2Index = pipe->AddLayer("yuv output", OEIP_OUTPUT_LAYER);
	//mapChannel与yuv2rgba同级
	pipe->ConnectLayer(mapChannel, "input");
	pipe->ConnectLayer(outYuv2Index, "rgba2yuv test");

	RGBA2YUVParamet ry = {};
	YUV2RGBAParamet yr = {};
	ry.yuvType = OEIP_YUVFMT_YUV420P;
	yr.yuvType = OEIP_YUVFMT_YUV420P;
	pipe->UpdateParamet(t1, ry);
	pipe->UpdateParamet(t2, yr);
	MapChannelParamet mp = {};
	mp.red = 0;
	mp.blue = 2;
	pipe->UpdateParamet(outMap, mp);
}

VideoPipe::~VideoPipe() {
}

void VideoPipe::setVideoFormat(OeipVideoType videoType, int32_t width, int32_t height) {
	yp.yuvType = getVideoYUV(videoType);
	pipe->UpdateParamet(yuv2rgba, yp);

	inputWidth = width;
	inputHeight = height;
	pipe->SetEnableLayer(yuv2rgba, true);
	pipe->SetEnableLayer(mapChannel, false);
	pipe->SetEnableLayer(resizeIndex, false);
	if (yp.yuvType == OEIP_YUVFMT_OTHER) {
		pipe->SetEnableLayer(yuv2rgba, false);
		if (videoType == OEIP_VIDEO_ARGB32) {
			pipe->SetEnableLayer(mapChannel, true);
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
	pipe->SetInput(inputIndex, inputWidth, inputHeight, dataType);
}

void VideoPipe::runVideoPipe(int32_t layerIndex, uint8_t * data) {
	pipe->UpdateInput(layerIndex, data);
	pipe->RunPipe();
}
