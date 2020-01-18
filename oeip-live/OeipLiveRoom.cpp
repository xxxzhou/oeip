#include "OeipLiveRoom.h"
#include <OeipCommon.h>

OeipLiveRoom::OeipLiveRoom() {
}

OeipLiveRoom::~OeipLiveRoom() {
	liveStatus = OEIP_LIVE_UNINIT;
}

void OeipLiveRoom::setLiveBack(OeipLiveBack* liveBack) {
	this->liveBack = liveBack;
}

bool OeipLiveRoom::initRoom(const OeipLiveContext& liveCtx) {
	this->liveCtx = liveCtx;
	bInit = initRoom();
	if (!bInit) {
		logMessage(OEIP_ERROR, "init room error.");
	}
	else {
		liveStatus = OEIP_LIVE_INIT;
	}
	return bInit;
}

bool OeipLiveRoom::loginRoom(std::string roomName, int32_t userId) {
	this->roomName = roomName;
	this->userId = userId;

	bool bLogin = loginRoom();
	return bLogin;
}

bool OeipLiveRoom::pushStream(int32_t index, const OeipPushSetting& setting) {
	return false;
}

void registerLiveFactory(ObjectFactory<OeipLiveRoom>* factory, int32_t type, std::string name) {
	PluginManager<OeipLiveRoom>::getInstance().registerFactory(factory, type, name);
}