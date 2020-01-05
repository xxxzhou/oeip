#pragma once
#include <mutex>
#include "FRtmpOutput.h"
#include "FEncoder.h"

//用于封装各种推流编码，暂时只支持RTMP
//RTM打开失败,视频大小变化等自动变更RTMP状态
class OEIPFMDLL_EXPORT FLiveOutput
{
public:
	FLiveOutput();
	~FLiveOutput();
private:
	std::unique_ptr<FEncoder> videoEncoder = nullptr;
	std::unique_ptr<FEncoder> audioEncoder = nullptr;
	std::unique_ptr<FNetOutput> output = nullptr;

	std::vector<uint8_t> videoBuffer;
	std::vector<uint8_t> audioBuffer;

	int32_t videoBitrate = 4000000;
	int32_t audioBitrate = 48000;
	int32_t fps = 30;
	bool bVideo = true;
	bool bAudio = true;

	bool bOpen = false;
	std::string liveUrl;

	int32_t videoHeight = -1;
	int32_t videoWidth = -1;

	std::mutex mtx;

	onOperateResult onOperateEvent;
public:
	int32_t open(const char* url);
	void close();

	int32_t pushVideo(const OeipVideoFrame& videoFrame);
	int32_t pushAudio(const OeipAudioFrame& audioFrame);

	void setVideoBitrate(int32_t bitrate);
	void setAudioBitrate(int32_t bitrate);

	void setFps(int32_t fps);
	void enableVideo(bool bVideo);
	void enableAudio(bool bAudio);
};

