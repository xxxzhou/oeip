#pragma once

#include "OeipLive.h"
//C++ 请直接实现这个抽象函数
class OEIPLIVEDLL_EXPORT OeipLiveBack
{
public:
	OeipLiveBack() {};
	virtual ~OeipLiveBack() {};
public:
	//是否初始化成功 code表示各种状态,为0正确，为负有问题
	virtual void onInitRoom(int32_t code) = 0;
	//如果设定自动生成ID,这个就是传出ID
	virtual void onLoginRoom(int32_t code, int32_t userId) = 0;
	//用户添加回调
	virtual void onUserChange(int32_t userId, bool bAdd) = 0;
	//用户推流回调
	virtual void onStreamUpdate(int32_t userId, int32_t index, bool bAdd) = 0;
	//拉流视频信息
	virtual void onVideoFrame(int32_t userId, int32_t index, OeipVideoFrame videoFrame) {};
	//拉流音频信息
	virtual void onAudioFrame(int32_t userId, int32_t index, OeipAudioFrame audioFrame) {};
	//登出房间回调
	virtual void onLogoutRoom(int32_t code) = 0;
	//各种操作回调
	virtual void onOperateResult(int32_t operate, int32_t code, const char* message) = 0;
	//推流回调
	virtual void onPushStream(int32_t index, int32_t code) = 0;
	//拉流回调
	virtual void onPullStream(int32_t userId, int32_t index, int32_t code) = 0;
};

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

