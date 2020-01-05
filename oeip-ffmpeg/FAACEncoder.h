#pragma once

#include "FEncoder.h"

#define  OEIP_AAC_BUFFER_MAX_SIZE    1024*16

class OEIPFMDLL_EXPORT FAACEncoder : public FEncoder
{
public:
	FAACEncoder(const OeipAudioEncoder& encoderDesc);
	~FAACEncoder();
private:
	OeipAudioEncoder encoderDesc = {};
	OAVCodecContext cdeCtx = nullptr;
	OAVFrame frame = nullptr;
	AVPacket packet = {};
	int32_t bufferSize = 0;
	OSwrContext swrCtx = nullptr;
	//限定音频数据输入格式
	AVSampleFormat inSampleFormat = AV_SAMPLE_FMT_S16;
	//限定AAC输出数据格式
	AVSampleFormat outSampleFormat = AV_SAMPLE_FMT_FLTP;
	uint8_t* samples = nullptr;
	std::vector<uint8_t> pcmBuffer;
	uint32_t pcmBufferSize = 0;
private:
	int32_t openEncode();
public:
	// 通过 FEncoder 继承
	virtual int encoder(const uint8_t* indata, int length, uint64_t timestamp) override;
	virtual int readPacket(uint8_t* outData, int& outLength, uint64_t& timestamp) override;
};

