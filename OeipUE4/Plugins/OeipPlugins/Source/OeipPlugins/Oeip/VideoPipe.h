// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "OeipPipe.h"

/**
 *
 */
class OEIPPLUGINS_API VideoPipe
{
public:
	VideoPipe(OeipPipe* opipe);
	~VideoPipe();
private:
	OeipPipe* pipe = nullptr;
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
	//int32_t getPipeId() { return pipe->; };
	int32_t getOutputId() {
		return outIndex;
	}
	int32_t getOutYuvId() {
		return outYuv2Index;
	}
	int32_t getResizeId() {
		return resizeIndex;
	}
	void setVideoFormat(OeipVideoType videoType, int32_t width, int32_t height);
	void runVideoPipe(int32_t layerIndex, uint8_t* data);
};
