// Fill out your copyright notice in the Description page of Project Settings.

#include "LivePushPipe.h"

LivePushPipe::LivePushPipe(OeipPipe* opipe, OeipYUVFMT fmt) {
	this->pipe = opipe;
	yuvFmt = fmt;
	inputIndex = pipe->AddLayer("input", OEIP_INPUT_LAYER);
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
}

LivePushPipe::~LivePushPipe() {
}
