// Fill out your copyright notice in the Description page of Project Settings.

#include "OeipManager.h"
#include <vector>

using namespace std::placeholders;


OeipManager *OeipManager::singleton = nullptr;
OeipManager::OeipManager() {
	std::function<void(int, const char*)> logHandle = std::bind(&OeipManager::onLogMessage, this, _1, _2);
	setLogHandle(logHandle);
	initOeip();
}

OeipManager::~OeipManager() {
	clearList(pipeList);
	//clearList(cameraArray);
	clearList(cameraList);
}

OeipManager & OeipManager::Get() {
	if (singleton == nullptr) {
		singleton = new OeipManager();
	}
	return *singleton;
}

void OeipManager::Close() {
	setLogHandle(nullptr);
	shutdownOeip();
}

TArray<FCameraInfo*> OeipManager::GetCameraList() {
	if (cameraList.Num() == 0) {
		int count = getDeviceCount();
		std::vector<OeipDeviceInfo> devices;
		devices.resize(count);
		getDeviceList(devices.data(), count);
		for (auto& camera : devices) {
			FCameraInfo* cameraInfo = new FCameraInfo();
			cameraInfo->index = camera.id;
			cameraInfo->deviceId = camera.deviceId;
			cameraInfo->deviceName = camera.deviceName;
			cameraList.Push(cameraInfo);
			//OeipCamera* camera = new OeipCamera();
			//camera->SetDevice(cameraInfo);
			//cameraArray.Push(camera);
		}
	}
	return cameraList;
}

FCameraInfo * OeipManager::GetCamera(int index) {
	if (index<0 || index>cameraList.Num())
		return nullptr;
	return cameraList[index];
}

OeipPipe * OeipManager::CreatePipe(OeipGpgpuType gpgpuType) {
	int pipeId = initPipe(gpgpuType);
	if (pipeId >= 0) {
		OeipPipe* pipe = new OeipPipe();
		pipe->SetPipeId(pipeId);
		pipeList.Push(pipe);
		return pipe;
	}
	return nullptr;
}

TArray<VideoFormat> OeipManager::GetCameraFormatList(int cameraIndex) {
	TArray<VideoFormat> formatList;
	int count = getFormatCount(cameraIndex);
	formatList.SetNum(count);
	getFormatList(cameraIndex, formatList.GetData(), count);
	return formatList;
}

FString OeipManager::GetLiveServer() {
	return FString("http://127.0.0.1:6110");//http://129.211.40.225:6110
}

void OeipManager::onLogMessage(int level, const char * message) {
	OnLogEvent.Broadcast(level, UTF8_TO_TCHAR(message));
}
