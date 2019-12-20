#include "VideoManager.h"



VideoManager::VideoManager()
{
}


VideoManager::~VideoManager()  
{
}

void registerFactory(ObjectFactory<VideoManager>* factory, int32_t type, std::string name)
{
	PluginManager<VideoManager>::getInstance().registerFactory(factory, type, name);
}