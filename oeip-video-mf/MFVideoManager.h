#pragma once
#include <VideoManager.h>
#include <PluginManager.h>
#include "MediaStruct.h"

class MFVideoManager :public VideoManager
{
public:
	MFVideoManager();
	~MFVideoManager();
public:
	// 通过 VideoManager 继承
	virtual std::vector<VideoDevice*> getDeviceList() override;
};

OEIP_DEFINE_PLUGIN_CLASS(VideoManager, MFVideoManager)

extern "C" __declspec(dllexport) bool bCanLoad();
extern "C" __declspec(dllexport) void registerFactory();
