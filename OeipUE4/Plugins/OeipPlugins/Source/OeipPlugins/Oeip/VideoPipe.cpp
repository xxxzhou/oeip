// Fill out your copyright notice in the Description page of Project Settings.

#include "VideoPipe.h"

VideoPipe::VideoPipe(OeipPipe* opipe) {
	this->pipe = opipe;
	inputIndex = pipe->AddLayer("input", OEIP_INPUT_LAYER);
	yuv2rgba = pipe->AddLayer("yuv2rgba", OEIP_YUV2RGBA_LAYER);
	mapChannel = pipe->AddLayer("map channel", OEIP_MAPCHANNEL_LAYER);
	//mapChannel与yuv2rgba同级
	pipe->ConnectLayer(mapChannel, inputIndex);
	//可以变化大小
	resizeIndex = pipe->AddLayer("resize", OEIP_RESIZE_LAYER);
	//输出原始图像
	outIndex = pipe->AddLayer("output", OEIP_OUTPUT_LAYER);
	op.bGpu = true;
	//没必要输出CPU数据
	op.bCpu = false;
	pipe->UpdateParamet(outIndex, op);
	if (pipe->GetGpuType() == OeipGpgpuType::OEIP_CUDA) {
		//神经网络层
		darknetIndex = pipe->AddLayer("darknet", OEIP_DARKNET_LAYER);
		pipe->ConnectLayer(darknetIndex, resizeIndex);
		//Grab cut扣像层
		grabcutIndex = pipe->AddLayer("grab cut", OEIP_GRABCUT_LAYER);
		grabcutParamet.bDrawSeed = 0;
		grabcutParamet.iterCount = 1;
		grabcutParamet.seedCount = 1000;
		grabcutParamet.count = 250;
		grabcutParamet.gamma = 90.0f;
		grabcutParamet.lambda = 450.0f;
		grabcutParamet.rect = {};
		pipe->UpdateParamet(grabcutIndex, grabcutParamet);
		//GuiderFilter
		guiderFilterIndex = pipe->AddLayer("guider filter", OEIP_GUIDEDFILTER_LAYER);
		guidedFilterParamet.zoom = 8;
		guidedFilterParamet.softness = 5;
		guidedFilterParamet.eps = 0.000001f;
		guidedFilterParamet.intensity = 0.2f;
		pipe->UpdateParamet(guiderFilterIndex, guidedFilterParamet);
		//输出第三个流，网络处理层流
		mattingOutIndex = pipe->AddLayer("matting out put", OEIP_OUTPUT_LAYER);
		pipe->UpdateParamet(mattingOutIndex, op);
	}
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

void VideoPipe::runVideoPipe(uint8_t * data) {
	pipe->UpdateInput(inputIndex, data);
	pipe->RunPipe();
}

void VideoPipe::updateDarknet(DarknetParamet& net) {
	darknetParamet = net;
	pipe->UpdateParamet(darknetIndex, darknetParamet);
}

void VideoPipe::changeGrabcutMode(bool bDrawSeed, OeipRect& rect) {
	//关闭画框 以免影响grabcut效果
	darknetParamet.bDraw = false;
	pipe->UpdateParamet(darknetIndex, darknetParamet);

	grabcutParamet.bDrawSeed = bDrawSeed ? 1 : 0;
	grabcutParamet.rect = rect;
	pipe->UpdateParamet(grabcutIndex, grabcutParamet);
}

void VideoPipe::updateVideoParamet(FGrabCutSetting * grabSetting) {
	grabcutParamet.iterCount = grabSetting->iterCount;
	grabcutParamet.seedCount = grabSetting->seedCount;
	grabcutParamet.count = grabSetting->flowCount;
	grabcutParamet.gamma = grabSetting->gamma;
	grabcutParamet.lambda = grabSetting->lambda;
	grabcutParamet.bGpuSeed = grabSetting->bGpuSeed ? 1 : 0;
	pipe->UpdateParamet(grabcutIndex, grabcutParamet);

	guidedFilterParamet.softness = grabSetting->softness;
	int epsx = (int)grabSetting->epslgn10;
	float epsf = FMath::Max(1.0f, (grabSetting->epslgn10 - epsx) * 10.0f);
	guidedFilterParamet.eps = epsf * (float)FMath::Pow(10, -epsx);
	guidedFilterParamet.intensity = grabSetting->intensity;
	pipe->UpdateParamet(guiderFilterIndex, guidedFilterParamet);
}
