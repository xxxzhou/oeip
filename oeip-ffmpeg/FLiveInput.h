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
	//是否这边播放音频
	void enablePlayAudio(bool bPlay);
	//是否解码视频数据
	void enableVideo(bool bVideo);
	//是否解码音频数据
	void enableAudio(bool bAudio);
	//打开RTMP地址或是音视频文件
	int32_t open(const char* url);
	//关闭
	void close();
	//每桢视频数据处理完后回调
	void setVideoDataEvent(onVideoFrameHandle onHandle);
	//每桢音频数据处理完后回调
	void setAudioDataEvent(onAudioFrameHandle onHandle);
	//打开/关闭/读取 结果回调
	void setOperateEvent(onOperateHandle onHandle);
};

