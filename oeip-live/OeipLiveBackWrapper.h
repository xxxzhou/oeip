#pragma once
#include "OeipLiveBack.h"

//给C#使用的OeipLiveBack封装实现
class OeipLiveBackWrapper : public OeipLiveBack
{
public:
	OeipLiveBackWrapper() {};
	virtual ~OeipLiveBackWrapper() {};
private:
	LiveBackWrapper liveBackWrapper = {};
public:
	void setLiveBackWrapper(LiveBackWrapper wrapper);
public:
	// 通过 OeipLiveBack 继承
	virtual void onInitRoom(int32_t code) override;
	virtual void onLoginRoom(int32_t code, int32_t userId) override;
	virtual void onUserChange(int32_t userId, bool bAdd) override;
	virtual void onStreamUpdate(int32_t userId, int32_t index, bool bAdd) override;
	virtual void onVideoFrame(int32_t userId, int32_t index, OeipVideoFrame videoFrame) override;
	virtual void onAudioFrame(int32_t userId, int32_t index, OeipAudioFrame audioFrame) override;
	virtual void onLogoutRoom(int32_t code) override;
	virtual void onOperateResult(int32_t operate, int32_t code, const char* message) override;

	virtual void onPushStream(int32_t index, int32_t code) override;
	virtual void onPullStream(int32_t userId, int32_t index, int32_t code) override;
};