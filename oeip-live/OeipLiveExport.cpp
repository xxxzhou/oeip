#include "OeipLiveExport.h"
#include "OeipLiveManager.h"

#define OEIPLIVE_CHECKINSTANCE \
	if (!oInstance) \
		return false;

#define OEIPLIVE_CHECKINSTANCEROOM \
	if (!oInstance) \
		return false;\
	OeipLiveRoom* liveRoom = oInstance->getLiveRoom();\
	if(!bInit || !liveRoom)\
		return false;

static OeipLiveManager* oInstance = nullptr;
static bool bInit = false;

void initOeipLive() {
	if (!bInit) {
		oInstance = OeipLiveManager::getInstance();
		bInit = true;
	}
}

void shutdownOeipLive() {
	if (bInit) {
		bInit = false;
		OeipLiveRoom* liveRoom = oInstance->getLiveRoom();
		if (!liveRoom) {
			liveRoom->shutdownRoom();
		}
		OeipLiveManager::shutdown();		
	}
}

bool initLiveRoom(const OeipLiveContext& liveCtx, OeipLiveBack* liveBack) {
	OEIPLIVE_CHECKINSTANCE;
	return oInstance->initRoom(liveCtx, liveBack);
}

bool initLiveRoomWrapper(const OeipLiveContext& liveCtx, const LiveBackWrapper& liveBack) {
	OEIPLIVE_CHECKINSTANCE;
	return oInstance->initRoom(liveCtx, liveBack);
}

bool loginRoom(const char* roomName, int32_t userId) {
	OEIPLIVE_CHECKINSTANCEROOM;
	return liveRoom->loginRoom(roomName, userId);
}

bool pushStream(int32_t index, const OeipPushSetting& setting) {
	OEIPLIVE_CHECKINSTANCEROOM;
	return liveRoom->pushStream(index, setting);
}

bool pushVideoFrame(int32_t index, const OeipVideoFrame& videoFrame) {
	OEIPLIVE_CHECKINSTANCEROOM;
	return liveRoom->pushVideoFrame(index, videoFrame);
}

bool pushAudioFrame(int32_t index, const OeipAudioFrame& audioFrame) {
	OEIPLIVE_CHECKINSTANCEROOM;
	return liveRoom->pushAudioFrame(index, audioFrame);
}

bool stopPushStream(int32_t index) {
	OEIPLIVE_CHECKINSTANCEROOM;
	return liveRoom->stopPushStream(index);
}

bool pullStream(int32_t userId, int32_t index) {
	OEIPLIVE_CHECKINSTANCEROOM;
	return liveRoom->pullStream(userId, index);
}

bool stopPullStream(int32_t userId, int32_t index) {
	OEIPLIVE_CHECKINSTANCEROOM;
	return liveRoom->stopPullStream(userId, index);
}

bool logoutRoom() {
	OEIPLIVE_CHECKINSTANCEROOM;
	return liveRoom->logoutRoom();
}

