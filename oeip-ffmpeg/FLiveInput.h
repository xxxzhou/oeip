#pragma once
#include <mutex>
#include "FNetCommon.h"
#include "FEncoder.h"

class OEIPFMDLL_EXPORT FLiveInput
{
public:
	FLiveInput();
	~FLiveInput();
private:
	std::unique_ptr<FNetInput> input = nullptr;

	bool bVideo = true;
	bool bAudio = true;

	bool bOpen = false;
	std::string liveUrl;

	//std::recursive_mutex mtx;
	//std::mutex mtx;
public:
	int32_t open(const char* url);
	void close();

	void enableVideo(bool bVideo);
	void enableAudio(bool bAudio);

	void setVideoDataEvent(onVideoDataHandle onHandle);
	void setOperateEvent(onOperateHandle onHandle);
};

