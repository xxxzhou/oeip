#pragma once
#include "OeipLiveRoom.h"
#include <memory>

class OeipLiveManager
{
public:
	static OeipLiveManager* getInstance();
	static void shutdown();

	~OeipLiveManager();
private:
	OeipLiveManager();
	static OeipLiveManager* instance;
	OeipLiveRoom* liveRoom = nullptr;
	std::unique_ptr<OeipLiveBackWrapper> liveBackWrapper = nullptr;
public:
	OeipLiveRoom* getLiveRoom() {
		return liveRoom;
	}
public:
	bool initRoom(const OeipLiveContext& liveCtx, OeipLiveBack* liveBack);
	bool initRoom(const OeipLiveContext& liveCtx, const LiveBackWrapper& liveBack);
};

