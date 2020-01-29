// Fill out your copyright notice in the Description page of Project Settings.

#include "LivePushPipe.h"

LivePushPipe::LivePushPipe(OeipPipe* opipe, OeipYUVFMT fmt) {
	this->pipe = opipe;
	yuvFmt = fmt;
	inputIndex = pipe->AddLayer("input", OEIP_INPUT_LAYER);
	//UE4 UTextureRenderTarget2D 固定为B8G8R8A8，所以转下通道
	int mapIndex = pipe->AddLayer("map", OEIP_MAPCHANNEL_LAYER);
	rgba2yuv = pipe->AddLayer("rgba2yuv", OEIP_RGBA2YUV_LAYER);
	outIndex = pipe->AddLayer("output", OEIP_OUTPUT_LAYER);

	InputParamet ip = {};
	ip.bCpu = false;
	ip.bGpu = true;
	pipe->UpdateParamet(inputIndex, ip);
	OutputParamet op = {};
	op.bCpu = true;
	op.bGpu = false;
	pipe->UpdateParamet(outIndex, op);
	RGBA2YUVParamet yp = {};
	yp.yuvType = OEIP_YUVFMT_YUV420P;
	pipe->UpdateParamet(rgba2yuv, yp);
	//CUDA模块需要手动把BGRA转成RGBA
	if (pipe->GetGpuType() == OEIP_CUDA) {
		MapChannelParamet mparamet = {};
		mparamet.blue = 0;
		mparamet.green = 1;
		mparamet.red = 2;
		mparamet.alpha = 3;
		pipe->UpdateParamet(mapIndex, mparamet);
	}	
}

LivePushPipe::~LivePushPipe() {
}
