#pragma once

#include "Oeipffmpeg.h"
#include "OeipFree.h"

//所有协议输出规则
class FNetOutput
{
public:
	virtual ~FNetOutput() {}

	virtual int openURL(const char* url, bool bVideo, bool bAudio) = 0;
	virtual void close() = 0;
	virtual int pushVideo(uint8_t* data, int size, uint64_t timestamp) = 0;
	virtual int pushAudio(uint8_t* data, int size, uint64_t timestamp) = 0;
};
