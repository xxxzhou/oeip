#include "OeipManager.h"
#include "VideoManager.h"


OeipManager* OeipManager::instance = nullptr;

void cleanPlugin(bool bFactory) {
	PluginManager<VideoManager>::clean(bFactory);
	PluginManager<ImageProcess>::clean(bFactory);
	PluginManager<AudioRecord>::clean(bFactory);
	PluginManager<AudioOutput>::clean(bFactory);
}

OeipManager* OeipManager::getInstance() {
	if (instance == nullptr) {
		instance = new OeipManager();
	}
	return instance;
}

void OeipManager::shutdown() {
	safeDelete(instance);
#ifdef _DEBUG
	cleanPlugin(true);
#else
	cleanPlugin(false);
#endif
}

OeipManager::~OeipManager() {
	for (auto& video : videoList) {
		video->closeDevice();
	}
	videoList.clear();
}

OeipManager::OeipManager() {
	initVideoList();
	audioOutput = PluginManager<AudioOutput>::getInstance().createModel(0);
	if (audioOutput) {

	}
}

void OeipManager::initVideoList() {
	videoList.clear();
	std::vector<VideoManager*> vmlist;
	PluginManager<VideoManager>::getInstance().getFactoryDefaultModel(vmlist, -1);
	for (auto& vm : vmlist) {
		std::vector<VideoDevice*> deviceList = vm->getDeviceList();
		videoList.insert(videoList.end(), deviceList.begin(), deviceList.end());
	}
}

int32_t OeipManager::initPipe(OeipGpgpuType gpgpuType) {
	auto vp = PluginManager<ImageProcess>::getInstance().createModel(gpgpuType);
	if (vp == nullptr)
		return -1;
	imagePipeList.push_back(vp);
	return imagePipeList.size() - 1;
}

bool OeipManager::closePipe(int32_t pipeId) {
	return 0;
}


