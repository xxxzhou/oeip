#pragma once
#include <RingBuffer.h>
#include "Oeipffmpeg.h"

#define BUFFER_SIZE 64*1024

//限定非P格式的PCM数据，输入输出用一层指针就行
class AudioResample
{
public:
	AudioResample();
	~AudioResample();
private:
	OeipAudioDesc sour = {};
	OeipAudioDesc dest = {};
	int32_t sourBlockAlign = 0;
	int32_t destBlockAlign = 0;
	//RingBuffer rbuffer = { BUFFER_SIZE };
	SwrContext* swrCtx = nullptr;
	bool bInit = false;
public:
	onAudioDataHandle onDataHandle;
public:
	int32_t init(OeipAudioDesc sour, OeipAudioDesc dest);
	//请保证inSize是sour.bitSize/8*sour.channel的倍数
	int32_t resampleData(const uint8_t* indata, int32_t inSize);
};

