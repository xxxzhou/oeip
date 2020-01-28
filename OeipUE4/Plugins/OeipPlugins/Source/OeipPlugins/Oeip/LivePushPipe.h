// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "OeipPipe.h"
#include "OeipLive.h"

//直接输入纹理
class OEIPPLUGINS_API LivePushPipe
{
public:
	//定义运行管线与输出格式
	LivePushPipe(OeipPipe* opipe, OeipYUVFMT fmt);
	~LivePushPipe();
private:
	OeipPipe* pipe = nullptr;
	int32_t inputIndex = -1;
	int32_t rgba2yuv = -1;
	int32_t outIndex = -1;
	OeipYUVFMT yuvFmt = OEIP_YUVFMT_YUY2P;
public:
	int GetInputId() {
		return inputIndex;
	}
	int GetOutputId() {
		return outIndex;
	}
};
