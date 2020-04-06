#pragma once

#include "Oeip.h"
#include "PluginManager.h"

class MediaOutput
{
public:
	virtual void setOperateEvent(onOperateHandle onHandle) = 0;

	virtual int32_t open(const char* url, bool bVideo, bool bAudio) = 0;
	virtual void setVideoEncoder(OeipVideoEncoder vEncoder) = 0;
	virtual void setAudioEncoder(OeipAudioEncoder aEncoder) = 0;
	virtual int32_t pushVideo(const OeipVideoFrame& videoFrame) = 0;
	virtual int32_t pushAudio(const OeipAudioFrame& audioFrame) = 0;
	virtual void close() = 0;

	virtual bool bOpen() = 0;
};

OEIP_DEFINE_PLUGIN_TYPE(MediaOutput);