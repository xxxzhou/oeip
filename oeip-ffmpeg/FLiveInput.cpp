#include "FLiveInput.h"
#include "FRtmpInput.h"
FLiveInput::FLiveInput() {
	input = std::unique_ptr<FRtmpInput>(new FRtmpInput());
}

FLiveInput::~FLiveInput() {
	close();
}

int32_t FLiveInput::open(const char* url) {		
	liveUrl = url;
	int32_t ret = 0;
	if (bOpen)
		return 1;
	std::thread ted = std::thread([&]() {	
		//std::unique_lock<std::mutex> lck(mtx);	
		//std::unique_lock <std::recursive_mutex> mtx_locker(mtx, std::try_to_lock);
		ret = input->openURL(liveUrl.c_str(), bVideo, bAudio);
		if (ret == 0)
			bOpen = true;		
	});
	ted.detach();
	return 0;
}

void FLiveInput::close() {	
	//std::unique_lock <std::recursive_mutex> mtx_locker(mtx, std::try_to_lock);
	if (!bOpen)
		return;
	bOpen = false;
	if (input) {
		input->close();
	}
}

void FLiveInput::enableVideo(bool bVideo) {
	this->bVideo = bVideo;
}

void FLiveInput::enableAudio(bool bAudio) {
	this->bAudio = bAudio;
}

void FLiveInput::setVideoDataEvent(onVideoDataHandle onHandle) {
	input->setVideoDataEvent(onHandle);
}

void FLiveInput::setOperateEvent(onOperateHandle onHandle) {
	input->setOperateEvent(onHandle);
}
