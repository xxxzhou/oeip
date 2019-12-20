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
	// Í¨¹ý VideoManager ¼Ì³Ð
	virtual std::vector<VideoDevice*> getDeviceList() override;
};

class MFVideoManagerFactory :public ObjectFactory<VideoManager>
{
public:
	MFVideoManagerFactory() {};
	~MFVideoManagerFactory() {};
public:
	virtual VideoManager* create(int type) override;
};

extern "C" __declspec(dllexport) bool bCanLoad();
extern "C" __declspec(dllexport) void registerFactory();
