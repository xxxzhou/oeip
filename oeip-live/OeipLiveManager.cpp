#include "OeipLiveManager.h"

OeipLiveManager* OeipLiveManager::instance = nullptr;

OeipLiveManager::OeipLiveManager() {
}

OeipLiveManager* OeipLiveManager::getInstance() {
	if (instance == nullptr) {
		instance = new OeipLiveManager();
	}
	return instance;
}

void OeipLiveManager::shutdown() {
	safeDelete(instance);
#ifdef _DEBUG
	PluginManager<OeipLiveRoom>::clean(true);
#else
	PluginManager<OeipLiveRoom>::clean(false);
#endif	
}

OeipLiveManager::~OeipLiveManager() {
	//只置空，由PluginManager<OeipLiveRoom>管理
	liveRoom = nullptr;
}

bool OeipLiveManager::initRoom(const OeipLiveContext& liveCtx, OeipLiveBack* liveBack) {
	liveRoom = PluginManager<OeipLiveRoom>::getInstance().createModel(liveCtx.liveMode);
	if (liveRoom == nullptr)
		return false;
	liveRoom->setLiveBack(liveBack);
	return liveRoom->initRoom(liveCtx);
}

bool OeipLiveManager::initRoom(const OeipLiveContext& liveCtx, const LiveBackWrapper& liveBack) {
	liveRoom = PluginManager<OeipLiveRoom>::getInstance().createModel(liveCtx.liveMode);
	if (liveRoom == nullptr)
		return false;
	liveBackWrapper = std::make_unique< OeipLiveBackWrapper>();
	liveBackWrapper->setLiveBackWrapper(liveBack);
	liveRoom->setLiveBack(liveBackWrapper.get());
	return liveRoom->initRoom(liveCtx);
}


