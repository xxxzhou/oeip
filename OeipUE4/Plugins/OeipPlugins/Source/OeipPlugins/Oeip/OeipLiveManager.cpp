// Fill out your copyright notice in the Description page of Project Settings.

#include "OeipLiveManager.h"
#include "OeipLiveExport.h"
#include "OeipManager.h"
using namespace std::placeholders;


OeipLiveManager *OeipLiveManager::singleton = nullptr;

OeipLiveManager::OeipLiveManager() {
	initOeipLive();
	OeipLiveContext olc = {};
	olc.bLoopback = false;
	olc.liveMode = OIEP_FFMPEG;
	FString liveServer = OeipManager::Get().GetLiveServer();
	copycharstr(olc.liveServer, TCHAR_TO_UTF8(*liveServer), 512);
	bInit = initLiveRoom(olc, this);
}

OeipLiveManager::~OeipLiveManager() {
}

OeipLiveManager & OeipLiveManager::Get() {
	if (singleton == nullptr) {
		singleton = new OeipLiveManager();
	}
	return *singleton;
}

void OeipLiveManager::Close() {
	logoutRoom();
	shutdownOeipLive();
}

bool OeipLiveManager::LoginRoom(FString roomName, int userId) {
	this->userId = userId;
	bLogin = loginRoom(TCHAR_TO_UTF8(*roomName), userId);
	return bLogin;
}

bool OeipLiveManager::PushStream(int index, OeipPushSetting & setting) {
	//if (!bLogin)
	//	return false;
	return pushStream(index, setting);
}

bool OeipLiveManager::PushVideoFrame(int index, uint8_t * data, int width, int height, OeipYUVFMT fmt) {	
	if (!bLogin)
		return false;
	OeipVideoFrame& videoFrame = index == 0 ? mainVideoFrame : auxVideoFrame;
	getVideoFrame(data, width, height, fmt, videoFrame);
	return pushVideoFrame(index, videoFrame);
}

bool OeipLiveManager::PushAudioFrame(int index, OeipAudioFrame & audioFrame) {
	
	if (!bLogin)
		return false;
	return pushAudioFrame(index, audioFrame);
}

bool OeipLiveManager::StopPushStream(int index) {
	return stopPushStream(index);
}

bool OeipLiveManager::PullStream(int userId, int index) {
	return pullStream(userId, index);
}

bool OeipLiveManager::StopPullStream(int userId, int index) {
	return stopPullStream(userId, index);
}

bool OeipLiveManager::LogoutRoom() {	
	bLogin = false;
	return logoutRoom();
}

void OeipLiveManager::onInitRoom(int32_t code) {
	OnInitRoomEvent.Broadcast(code);
}

void OeipLiveManager::onLoginRoom(int32_t code, int32_t userId) {
	this->userId = userId;
	OnLoginRoomEvent.Broadcast(code, userId);
}

void OeipLiveManager::onUserChange(int32_t userId, bool bAdd) {
	OnUserChangeEvent.Broadcast(userId, bAdd);
}

void OeipLiveManager::onStreamUpdate(int32_t userId, int32_t index, bool bAdd) {
	OnStreamUpdateEvent.Broadcast(userId, index, bAdd);
}

void OeipLiveManager::onLogoutRoom(int32_t code) {
	OnLogoutRoomEvent.Broadcast(code);
}

void OeipLiveManager::onOperateResult(int32_t operate, int32_t code, const char* message) {
	OnOperateResultEvent.Broadcast(operate, code, UTF8_TO_TCHAR(message));
}

void OeipLiveManager::onPushStream(int32_t index, int32_t code) {
	OnPushStreamEvent.Broadcast(index, code);
}

void OeipLiveManager::onPullStream(int32_t userId, int32_t index, int32_t code) {
	OnPullStreamEvent.Broadcast(userId, index, code);
}

void OeipLiveManager::onVideoFrame(int32_t userId, int32_t index, OeipVideoFrame videoFrame) {
	OnVideoFrameEvent.Broadcast(userId, index, videoFrame);
}

void OeipLiveManager::onAudioFrame(int32_t userId, int32_t index, OeipAudioFrame audioFrame) {
	OnAudioFrameEvent.Broadcast(userId, index, audioFrame);
}
