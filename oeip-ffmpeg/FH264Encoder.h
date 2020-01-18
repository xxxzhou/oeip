#pragma once

#include "FEncoder.h"

#define  OEIP_H264_BUFFER_MAX_SIZE   1024*1024

class OEIPFMDLL_EXPORT FH264Encoder : public FEncoder
{
public:
	FH264Encoder(const OeipVideoEncoder& encoderDesc);
	~FH264Encoder();
private:
	void openEncode();
	int32_t findEncode(const char* name);
	int32_t openEncode(AVCodec* codec);
private:
	int32_t ysize = 0;
	OAVCodecContext cdeCtx = nullptr;
	OeipVideoEncoder encoderDesc = {};
	OAVFrame frame = nullptr;
	AVPacket packet = {};	
public:
	virtual int encoder(uint8_t** indata, int length, uint64_t timestamp) override;
	virtual int readPacket(uint8_t* outData, int& outLength, uint64_t& timestamp) override;
};

