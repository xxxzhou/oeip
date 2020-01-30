#include "OeipLiveBackWrapper.h"

void OeipLiveBackWrapper::setLiveBackWrapper(LiveBackWrapper wrapper) {
	this->liveBackWrapper = wrapper;
}

void OeipLiveBackWrapper::onInitRoom(int32_t code) {
	if (liveBackWrapper.initRoomAction) {
		liveBackWrapper.initRoomAction(code);
	}
}

void OeipLiveBackWrapper::onLoginRoom(int32_t code, int32_t userId) {
	if (liveBackWrapper.loginRoomAction) {
		liveBackWrapper.loginRoomAction(code, userId);
	}
}

void OeipLiveBackWrapper::onUserChange(int32_t userId, bool bAdd) {
	if (liveBackWrapper.userChangeAction) {
		liveBackWrapper.userChangeAction(userId, bAdd);
	}
}

void OeipLiveBackWrapper::onStreamUpdate(int32_t userId, int32_t index, bool bAdd) {
	if (liveBackWrapper.streamUpdateAction) {
		liveBackWrapper.streamUpdateAction(userId, index, bAdd);
	}
}

void OeipLiveBackWrapper::onVideoFrame(int32_t userId, int32_t index, OeipVideoFrame videoFrame) {
	if (liveBackWrapper.videoFrameAction) {
		liveBackWrapper.videoFrameAction(userId, index, videoFrame);
	}
}

void OeipLiveBackWrapper::onAudioFrame(int32_t userId, int32_t index, OeipAudioFrame audioFrame) {
	if (liveBackWrapper.audioFrameAction) {
		liveBackWrapper.audioFrameAction(userId, index, audioFrame);
	}
}

void OeipLiveBackWrapper::onLogoutRoom(int32_t code) {
	if (liveBackWrapper.logoutRoomAction) {
		liveBackWrapper.logoutRoomAction(code);
	}
}

void OeipLiveBackWrapper::onOperateResult(int32_t operate, int32_t code, const char* message) {
	if (liveBackWrapper.operateResultAction) {
		liveBackWrapper.operateResultAction(operate, code, message);
	}
}

void OeipLiveBackWrapper::onPushStream(int32_t index, int32_t code) {
	if (liveBackWrapper.pushStreamAction) {
		liveBackWrapper.pushStreamAction(index, code);
	}
}

void OeipLiveBackWrapper::onPullStream(int32_t userId, int32_t index, int32_t code) {
	if (liveBackWrapper.pullStreamAction) {
		liveBackWrapper.pullStreamAction(userId, index, code);
	}
}
