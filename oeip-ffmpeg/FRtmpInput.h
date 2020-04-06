#pragma once

#include "Oeipffmpeg.h"
#include "OeipFree.h"
#include <mutex>
#include "FNetCommon.h"
#include "FAudioPlay.h"
//#include <future>

//拉流
class OEIPFMDLL_EXPORT FRtmpInput :public FNetInput
{
public:
	FRtmpInput();
	~FRtmpInput();
public:
	bool bTempOpen = false;
	//音频播放处理
	bool bPlayAudio = true;
private:
	OAVFormatContext fmtCtx = nullptr;
	OAVCodecContext audioCtx = nullptr;
	OAVCodecContext videoCtx = nullptr;
	OeipVideoEncoder videoEncoder = {};
	OeipAudioEncoder audioEncoder = {};

	int32_t videoIndex = -1;
	int32_t audioIndex = -1;
	uint64_t audioTimestamp = 0;
	uint64_t videoTimestamp = 0;
	std::string url = "";
	bool bVideo = true;
	bool bAudio = true;
	bool bRtmp = true;
	bool bOpenPull = false;

	std::unique_ptr<FAudioPlay> audioPlay = nullptr;

	OeipFAVFormat oformat = OEIP_AVFORMAT_FLV;

	OAVFrame frame = nullptr;
	//在这主要是把P数据格式转化成I格式
	OSwrContext swrCtx = nullptr;
	//音频数据输出格式,后面设定可变
	AVSampleFormat outSampleFormat = AV_SAMPLE_FMT_S16;
	int outChannel = AV_CH_FRONT_CENTER;
	std::mutex mtx;
	//信号量.
	std::condition_variable signal;
private:
	void readPack();
public:
	virtual int32_t open(const char* url, bool bVideo, bool bAudio) override;
	virtual void close() override;
	virtual void enablePlayAudio(bool bPlay) override {
		bPlayAudio = bPlay;
	}
public:
	virtual bool getVideoInfo(OeipVideoEncoder& videoInfo) override {
		if (videoIndex < 0)
			return false;
		videoInfo = videoEncoder;
	}

	virtual bool getAudioInfo(OeipAudioEncoder& audioInfo) override {
		if (audioIndex < 0)
			return false;
		audioInfo = audioEncoder;
	}

	virtual bool bOpen() override {
		std::unique_lock <std::mutex> lck(mtx);
		return bOpenPull;
	}
};

OEIP_DEFINE_PLUGIN_CLASS(MediaPlay, FRtmpInput)