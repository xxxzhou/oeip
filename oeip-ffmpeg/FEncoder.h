#pragma once

#include "Oeipffmpeg.h"
#include "OeipFree.h"

class FEncoder
{
protected:
	bool bInit = false;
public:
	virtual ~FEncoder() {};

	virtual int encoder(uint8_t** indata, int length, uint64_t timestamp) = 0;
	virtual int readPacket(uint8_t* outData, int& outLength, uint64_t& timestamp) = 0;
};
