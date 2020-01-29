// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "OeipLiveBack.h"
#include <mutex>

DECLARE_MULTICAST_DELEGATE_OneParam(FOnInitRoomEvent, int);
DECLARE_MULTICAST_DELEGATE_TwoParams(FOnLoginRoomEvent, int, int);
DECLARE_MULTICAST_DELEGATE_TwoParams(FOnUserChangeEvent, int, bool);
DECLARE_MULTICAST_DELEGATE_ThreeParams(FOnStreamUpdateEvent, int, int, bool);
DECLARE_MULTICAST_DELEGATE_ThreeParams(FOnVideoFrameEvent, int, int, OeipVideoFrame);
DECLARE_MULTICAST_DELEGATE_ThreeParams(FOnAudioFrameEvent, int, int, OeipAudioFrame);
DECLARE_MULTICAST_DELEGATE_OneParam(FOnLogoutRoomEvent, int);
DECLARE_MULTICAST_DELEGATE_ThreeParams(FOnOperateResultEvent, int, int, FString);
DECLARE_MULTICAST_DELEGATE_TwoParams(FOnPushStreamEvent, int, int);
DECLARE_MULTICAST_DELEGATE_ThreeParams(FOnPullStreamEvent, int, int, int);

//封装登陆直播服务器，各种通知
class OEIPPLUGINS_API OeipLiveManager :public OeipLiveBack
{
public:
	~OeipLiveManager();
	static OeipLiveManager& Get();
	static void Close();
private:
	OeipLiveManager();
	static OeipLiveManager *singleton;
public:
	FOnInitRoomEvent OnInitRoomEvent;
	FOnLoginRoomEvent OnLoginRoomEvent;
	FOnUserChangeEvent OnUserChangeEvent;
	FOnStreamUpdateEvent OnStreamUpdateEvent;
	FOnVideoFrameEvent OnVideoFrameEvent;
	FOnAudioFrameEvent OnAudioFrameEvent;
	FOnLogoutRoomEvent OnLogoutRoomEvent;
	FOnOperateResultEvent OnOperateResultEvent;
	FOnPushStreamEvent OnPushStreamEvent;
	FOnPullStreamEvent OnPullStreamEvent;
private:
	std::mutex mtx;
	bool bInit = false;
	bool bLogin = false;
	int userId = -1;
	OeipVideoFrame mainVideoFrame = {};
	OeipVideoFrame auxVideoFrame = {};
public:
	int GetUserId() {
		return userId;
	}
	//主动操作
	bool LoginRoom(FString roomName, int userId);
	bool PushStream(int index, OeipPushSetting& setting);
	bool PushVideoFrame(int index, uint8_t* data, int width, int height, OeipYUVFMT fmt);
	bool PushAudioFrame(int index, OeipAudioFrame& audioFrame);
	bool StopPushStream(int index);
	bool PullStream(int userId, int index);
	bool StopPullStream(int userId, int index);
	bool LogoutRoom();
private:
	// 通过 OeipLiveBack 继承，对应主动操作回调或是服务器通知
	virtual void onInitRoom(int32_t code) override;
	virtual void onLoginRoom(int32_t code, int32_t userId) override;
	virtual void onUserChange(int32_t userId, bool bAdd) override;
	virtual void onStreamUpdate(int32_t userId, int32_t index, bool bAdd) override;
	virtual void onLogoutRoom(int32_t code) override;
	virtual void onOperateResult(int32_t operate, int32_t code, const char* message) override;
	virtual void onPushStream(int32_t index, int32_t code) override;
	virtual void onPullStream(int32_t userId, int32_t index, int32_t code) override;
	virtual void onVideoFrame(int32_t userId, int32_t index, OeipVideoFrame videoFrame) override;
	//拉流音频信息
	virtual void onAudioFrame(int32_t userId, int32_t index, OeipAudioFrame audioFrame) override;
};
