#pragma once

#include "FNetOutput.h"

//推流 (抓到RTMP网址打不开的错误码，这个要单独处理)
//用FFmpeg IO模式，根据输入的视频与音频信息自动得到推流所需要的编码信息，断开推流与输入信息的联系
//不过这种方式需要先输出音视频信息,然后才能确定是否成功推流，并且这种方式不能同步得到是否正常打开推流
class FRtmpOutput : public FNetOutput
{
public:
	FRtmpOutput();
	~FRtmpOutput();
private:
	OAVFormatContext fmtCtx = nullptr;
	OAVFormatContext audioFmtCtx = nullptr;
	OAVFormatContext videoFmtCtx = nullptr;

	OeipVideoEncoder videoEncoder = {};
	OeipAudioEncoder audioEncoder = {};

	std::string url = "";
	int32_t videoIndex = -1;
	int32_t audioIndex = -1;
	uint64_t audioTimestamp = 0;
	uint64_t videoTimestamp = 0;

	bool bVideo = true;
	bool bAudio = true;
	bool bRtmp = true;

	bool bOpenPush = false;
	bool bIFrameFirst = false;
	bool bAACFirst = false;

	std::vector<uint8_t> audioPack;
	//记录第一个I桢
	std::vector<uint8_t> videoPack;
	std::vector<std::shared_ptr<StreamFrame>> videoCacheFrame;
private:
	int32_t startPush();
public:
	int32_t audioIO(uint8_t* buffer, int32_t size);
	int32_t videoIO(uint8_t* buffer, int32_t size);
public:
	// 通过 FNetOutput 继承
	virtual int32_t openURL(const char* url, bool bVideo, bool bAudio) override;
	virtual void close() override;
	//输入H264的流数据进来
	virtual int32_t pushVideo(uint8_t* data, int32_t size, uint64_t timestamp) override;
	//输入AAC音频数据进来
	virtual int32_t pushAudio(uint8_t* data, int32_t size, uint64_t timestamp) override;
};

