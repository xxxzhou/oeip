#pragma once
#include "OeipLive.h"
#include "OeipLiveBack.h"
#include <OeipManager.h>
#include <memory>

class OEIPLIVEDLL_EXPORT OeipLiveRoom
{
public:
	OeipLiveRoom();
	virtual ~OeipLiveRoom();
protected:
	OeipLiveBack* liveBack = nullptr;
	int32_t userId = 0;
	std::string roomName = "";
	OeipLiveContext liveCtx = {};
	bool bInit = false;
	OeipLiveStatus liveStatus = OEIP_LIVE_UNINIT;

	OeipPushSetting mainPushSetting = {};
	OeipPushSetting auxPushSetting = {};
protected:
	//初始化对应数据
	virtual bool initRoom() = 0;
	//登陆房间
	virtual bool loginRoom() = 0;
public:
	virtual bool pushStream(int32_t index, const OeipPushSetting& setting) = 0;
	virtual bool stopPushStream(int32_t index) = 0;
	virtual bool pullStream(int32_t userId, int32_t index) = 0;
	virtual bool stopPullStream(int32_t userId, int32_t index) = 0;
	virtual bool logoutRoom() = 0;
	virtual bool shutdownRoom() = 0;
	//推视频信息
	virtual bool pushVideoFrame(int32_t index, const OeipVideoFrame& videoFrame) = 0;
	//推音频信息
	virtual bool pushAudioFrame(int32_t index, const OeipAudioFrame& audioFrame) = 0;
public:
	void setLiveBack(OeipLiveBack* liveBack);

	bool initRoom(const OeipLiveContext& liveCtx);
	bool loginRoom(std::string roomName, int32_t userId);
};

//OEIP_DEFINE_PLUGIN_TYPE(OeipLiveRoom);

OEIPLIVEDLL_EXPORT void registerLiveFactory(ObjectFactory<OeipLiveRoom>* factory, int32_t type, std::string name);


