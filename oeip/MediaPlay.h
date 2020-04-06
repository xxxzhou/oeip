#pragma once

#include "Oeip.h"
#include "PluginManager.h"

class MediaPlay
{
public:
	virtual void setOperateEvent(onOperateHandle onHandle) = 0;
	virtual void enablePlayAudio(bool bPlay) = 0;
	virtual int32_t open(const char* url, bool bVideo, bool bAudio) = 0;
	virtual bool getVideoInfo(OeipVideoEncoder& videoInfo) = 0;
	virtual bool getAudioInfo(OeipAudioEncoder& videoInfo) = 0;
	virtual void setVideoDataEvent(onVideoFrameHandle onHandle) = 0;
	virtual void setAudioDataEvent(onAudioFrameHandle onHandle) = 0;
	virtual void close() = 0;

	virtual bool bOpen() = 0;
};

OEIP_DEFINE_PLUGIN_TYPE(MediaPlay);