#pragma once

#include "OeipCommon.h"
#include "VideoDevice.h"
#include "ImageProcess.h"

void cleanPlugin(bool bFactory = false);

class OeipManager
{
public:
	static OeipManager* getInstance();
	static void shutdown();

	~OeipManager();
private:
	OeipManager();
	static OeipManager* instance;
	std::vector<VideoDevice*> videoList;
	std::vector<ImageProcess*> imagePipeList;
private:
	void initVideoList();
public:
	int32_t initPipe(OeipGpgpuType gpgpuType);
	const std::vector<VideoDevice*>& getVideoList() {
		return videoList;
	};
	VideoDevice* getVideoIndex(const int32_t index) {
		if (index < 0 || index >= videoList.size()) {
			logMessage(OEIP_ERROR, "getVideoIndex incorrect index");
			return nullptr;
		}
		return videoList[index];
	};
	ImageProcess* getPipe(const int32_t index) {
		if (index < 0 || index >= imagePipeList.size()) {
			logMessage(OEIP_ERROR, "getPipe incorrect index");
			return nullptr;
		}
		return imagePipeList[index];
	};
};

template<typename T>
inline bool updateLayer(OeipManager* om, int32_t pipeId, int32_t layerIndex, const T& t) {
	auto pipe = om->getPipe(pipeId);
	if (!pipe)
		return false;
	return pipe->updateLayer(layerIndex, t);	
}
