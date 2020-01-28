// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "OeipPipe.h"
#include "OeipLive.h"

DECLARE_MULTICAST_DELEGATE_TwoParams(FOnPullDataEvent, int32_t, int32_t);

//封装一个拉流管线,在UE4中，只考虑GPU输出
class OEIPPLUGINS_API LivePullPipe
{
public:
	LivePullPipe(OeipPipe* opipe);
	~LivePullPipe();
private:
	OeipPipe* pipe = nullptr;
	int32_t inputIndex = -1;
	int32_t yuv2rgb = -1;
	int32_t outIndex = -1;
	OeipYUVFMT yuvFmt = OEIP_YUVFMT_YUY2P;
	TArray<uint8_t> data;
	int32_t width = 0;
	int32_t height = 0;
public:
	FOnPullDataEvent OnPullDataEvent;
	int32_t getOutputId() {
		return outIndex;
	}
	void RunPipe(OeipVideoFrame& videoFrame);
private:
	void ResetPipe();
};
