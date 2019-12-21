#pragma once
#include "VideoDevice.h"
#include "PluginManager.h"

class OEIPDLL_EXPORT VideoManager
{
public:
	VideoManager();
	virtual ~VideoManager();
protected:
	std::vector<VideoDevice*> videoList;
public:
	virtual std::vector<VideoDevice*> getDeviceList() = 0;
};
//实例化供导出
template OEIPDLL_EXPORT void registerFactory<VideoManager>(ObjectFactory<VideoManager>* factory, int32_t type, std::string name);
