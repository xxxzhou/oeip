#pragma once
#include <mutex>
#include "FNetCommon.h"
#include "FEncoder.h"
#include <MediaOutput.h>

class OEIPFMDLL_EXPORT FMuxing : public MediaOutput
{
public:
	FMuxing();
	~FMuxing();
private:
	OAVFormatContext fmtCtx = nullptr;
	std::unique_ptr<FEncoder> videoEncoder = nullptr;
	std::unique_ptr<FEncoder> audioEncoder = nullptr;
	OeipVideoEncoder videoInfo = {};
	OeipAudioEncoder audioInfo = {};

	std::vector<uint8_t> videoBuffer;
	std::vector<uint8_t> audioBuffer;

	bool bVideo = true;
	bool bAudio = true;
	bool bOpenPush = false;
	std::string url = "";
	int32_t videoIndex = -1;
	int32_t audioIndex = -1;
	bool bRtmp = false;
	std::mutex mtx;
	//操作事件
	onOperateHandle onOperateEvent;
private:
	void onOperateAction(int32_t operate, int32_t code) {
		if (onOperateEvent) {
			onOperateEvent(operate, code);
		}
	}
public:
	virtual void setOperateEvent(onOperateHandle onHandle) override {
		onOperateEvent = onHandle;
	};

	virtual void setVideoEncoder(OeipVideoEncoder vEncoder) override;
	virtual void setAudioEncoder(OeipAudioEncoder aEncoder) override;

	//void enableVideo(bool bVideo) {
	//	this->bVideo = bVideo;
	//};
	//void enableAudio(bool bAudio) {
	//	this->bAudio = bAudio;
	//};

	virtual int32_t open(const char* url, bool bVideo, bool bAudio) override;
	virtual void close();

	virtual int32_t pushVideo(const OeipVideoFrame& videoFrame) override;
	virtual int32_t pushAudio(const OeipAudioFrame& audioFrame) override;

	virtual bool bOpen() override {
		std::unique_lock <std::mutex> lck(mtx);
		return bOpenPush;
	}
};

OEIP_DEFINE_PLUGIN_CLASS(MediaOutput, FMuxing)

extern "C" __declspec(dllexport) bool bCanLoad();
extern "C" __declspec(dllexport) void registerFactory();
