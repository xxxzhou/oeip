#include "OeipManager.h"
#include "VideoManager.h"
#include "AudioRecord.h"

OeipManager* OeipManager::instance = nullptr;

void cleanPlugin(bool bFactory)
{
	PluginManager<VideoManager>::clean(bFactory);
	PluginManager<AudioRecord>::clean(bFactory);
	PluginManager<ImageProcess>::clean(bFactory);
}

OeipManager* OeipManager::getInstance()
{
	if (instance == nullptr) {
		instance = new OeipManager();
	}
	return instance;
}

void OeipManager::shutdown()
{
	safeDelete(instance);
	cleanPlugin(false);
}

OeipManager::~OeipManager()
{
	for (auto& video : videoList) {
		video->closeDevice();
	}
}

OeipManager::OeipManager()
{
	initVideoList();
}

void OeipManager::initVideoList()
{
	videoList.clear();
	std::vector<VideoManager*> vmlist;
	PluginManager<VideoManager>::getInstance().getFactoryDefaultModel(vmlist, -1);
	for (VideoManager* vm : vmlist) {
		std::vector<VideoDevice*> deviceList = vm->getDeviceList();
		videoList.insert(videoList.end(), deviceList.begin(), deviceList.end());
	}
}

int32_t OeipManager::initPipe(OeipGpgpuType gpgpuType)
{
	auto vp = PluginManager<ImageProcess>::getInstance().createModel(gpgpuType);
	if (vp == nullptr)
		return -1;
	imagePipeList.push_back(vp);

	vp->addLayer("input", OEIP_INPUT_LAYER);
	vp->addLayer("nv2rgba", OEIP_YUV2RGBA_LAYER);
	vp->addLayer("output", OEIP_OUTPUT_LAYER);

	return imagePipeList.size() - 1;
}


