#pragma once

#include "Oeipffmpeg.h"
#include <OeipCommon.h>

class OEIPFMDLL_EXPORT FAudioPlay
{
public:
	FAudioPlay();
	~FAudioPlay();
private:
	int deviceId = 0;
public:
	bool openDevice(const OeipAudioDesc& audioDesc);
	void playAudioData(uint8_t* data, int32_t lenght);
	void closeDevice();
};

