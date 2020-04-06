#pragma once

#include "OeipCommon.h"
#include "VideoDevice.h"
#include "ImageProcess.h"
#include "AudioOutput.h"
#include "MediaOutput.h"
#include "MediaPlay.h"
#include <memory>
#include <mutex>

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
	std::vector<MediaPlay*> mediaPlayList;
	std::vector<MediaOutput*> mediaOutputList;
	AudioOutput* audioOutput = nullptr;
	std::mutex mtx;
private:
	void initVideoList();
public:
	int32_t initPipe(OeipGpgpuType gpgpuType);
	//清理这个管理占用的资源
	bool closePipe(int32_t pipeId);
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
		std::lock_guard<std::mutex> mtx_locker(mtx);
		if (index < 0 || index >= imagePipeList.size()) {
			logMessage(OEIP_ERROR, "getPipe incorrect index");
			return nullptr;
		}
		return imagePipeList[index];
	};
	AudioOutput* getAudioOutput() {
		return audioOutput;
	};
	int32_t initReadMedia();

	int32_t initWriteMedia();

	MediaPlay* getMediaPlay(const int32_t index) {
		if (index < 0 || index >= mediaPlayList.size()) {
			logMessage(OEIP_ERROR, "getMediaPlay incorrect index");
			return nullptr;
		}
		return mediaPlayList[index];
	};

	MediaOutput* getMediaOutput(const int32_t index) {
		if (index < 0 || index >= mediaOutputList.size()) {
			logMessage(OEIP_ERROR, "getMediaOutput incorrect index");
			return nullptr;
		}
		return mediaOutputList[index];
	};
};

template<typename T>
inline bool updateLayer(OeipManager* om, int32_t pipeId, int32_t layerIndex, const T& t) {
	auto pipe = om->getPipe(pipeId);
	if (!pipe)
		return false;
	return pipe->updateLayer(layerIndex, t);
}
