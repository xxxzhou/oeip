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

	int32_t darknetIndex = -1;
	int32_t grabcutIndex = -1;
	int32_t guiderFilterIndex = -1;
	int32_t mattingOutIndex = -1;
private:
	InputParamet ip = {};
	YUV2RGBAParamet yp = {};
	OutputParamet op = {};
	ResizeParamet rp = {};
	DarknetParamet darknetParamet = {};
	GrabcutParamet grabcutParamet = {};
	GuidedFilterParamet guidedFilterParamet = {};	
public:
	//int32_t getPipeId() { return pipe->; };
	int32_t getInputId() {
		return inputIndex;
	}

	int32_t getOutputId() {
		if (!pipe)
			return -1;
		if (pipe->GetGpuType() == OEIP_DX11)
			return outIndex;
		return mattingOutIndex;
	}
	int32_t getDarknetId() {
		return darknetIndex;
	}
	int32_t getResizeId() {
		return resizeIndex;
	}
	void setVideoFormat(OeipVideoType videoType, int32_t width, int32_t height);
	void runVideoPipe(uint8_t* data);

	void updateDarknet(DarknetParamet& net);
	void changeGrabcutMode(bool bDrawSeed, OeipRect& rect);
	void updateVideoParamet(FGrabCutSetting* grabSetting);
};
