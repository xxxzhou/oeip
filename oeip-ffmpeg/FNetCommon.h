#pragma once

#include "Oeipffmpeg.h"
#include "OeipFree.h"
#include <MediaPlay.h>

//所有协议输出规则
class FNetOutput
{
public:
	virtual ~FNetOutput() {};
protected:
	onOperateHandle onOperateEvent;
public:
	virtual int32_t open(const char* url, bool bVideo, bool bAudio) = 0;
	virtual void close() = 0;
	virtual int32_t pushVideo(uint8_t* data, int size, uint64_t timestamp) = 0;
	virtual int32_t pushAudio(uint8_t* data, int size, uint64_t timestamp) = 0;
protected:
	void onOperateAction(int32_t operate, int32_t code) {
		if (onOperateEvent) {
			onOperateEvent(operate, code);
		}
	}
public:
	virtual void setOperateEvent(onOperateHandle onHandle) { onOperateEvent = onHandle; };
};

class FNetInput : public MediaPlay
{
public:
	virtual ~FNetInput() {};
protected:
	onVideoFrameHandle onVideoDataEvent;
	onAudioFrameHandle onAudioFrameEvent;
	onOperateHandle onOperateEvent;
public:
	//virtual int32_t open(const char* url, bool bVideo, bool bAudio) = 0;
	//virtual void close() = 0;
	//virtual void enablePlayAudio(bool bPlay) = 0;
protected:
	void onOperateAction(int32_t operate, int32_t code) {
		if (onOperateEvent) {
			onOperateEvent(operate, code);
		}
	}
public:
	virtual void setVideoDataEvent(onVideoFrameHandle onHandle) override { onVideoDataEvent = onHandle; };
	virtual void setAudioDataEvent(onAudioFrameHandle onHandle) override { onAudioFrameEvent = onHandle; };
	virtual void setOperateEvent(onOperateHandle onHandle)override { onOperateEvent = onHandle; };
};
