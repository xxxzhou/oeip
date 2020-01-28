// Fill out your copyright notice in the Description page of Project Settings.

#include "LivePullPipe.h"
#include "OeipLiveExport.h"

LivePullPipe::LivePullPipe(OeipPipe* opipe) {
	this->pipe = opipe;
	inputIndex = pipe->AddLayer("input", OEIP_INPUT_LAYER);
	yuv2rgb = pipe->AddLayer("yuv2rgb", OEIP_YUV2RGBA_LAYER);
	outIndex = pipe->AddLayer("output", OEIP_OUTPUT_LAYER);

	InputParamet ip = {};
	ip.bCpu = true;
	ip.bGpu = false;
	pipe->UpdateParamet(inputIndex, ip);
	OutputParamet op = {};
	op.bCpu = false;
	op.bGpu = true;
	pipe->UpdateParamet(outIndex, op);
}

LivePullPipe::~LivePullPipe() {
	data.Empty();
}

void LivePullPipe::RunPipe(OeipVideoFrame & videoFrame) {
	if (width != videoFrame.width || height != videoFrame.height) {
		yuvFmt = videoFrame.fmt;
		ResetPipe();
	}
	getVideoFrameData(data.GetData(), videoFrame);
	pipe->UpdateInput(inputIndex, data.GetData());
	pipe->RunPipe();
}

void LivePullPipe::ResetPipe() {
	int dataType = OEIP_CV_8UC1;
	int inputHeight = height * 2;
	if (yuvFmt == OEIP_YUVFMT_YUV420P) {
		inputHeight = height * 3 / 2;
	}
	YUV2RGBAParamet yp = {};
	yp.yuvType = yuvFmt;
	pipe->UpdateParamet(yuv2rgb, yp);
	pipe->SetInput(inputIndex, width, height, dataType);
	data.SetNum(width*inputHeight);
	OnPullDataEvent.Broadcast(width, height);
}
