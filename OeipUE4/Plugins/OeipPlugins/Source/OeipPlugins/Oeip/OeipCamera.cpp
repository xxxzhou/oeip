// Fill out your copyright notice in the Description page of Project Settings.

#include "OeipCamera.h"
#include "OeipManager.h"

using namespace std::placeholders;

OeipCamera::OeipCamera() {
}

OeipCamera::~OeipCamera() {
	formatList.Empty();
	if (bInit()) {
		setDeviceDataHandle(id, nullptr);
		setDeviceEventHandle(id, nullptr);
		Close();
	}
}

bool OeipCamera::bInit() {
	if (id < 0)
		return false;
	return true;
}

void OeipCamera::onDeviceHandle(int type, int code) {
	OnDeviceEvent.Broadcast(type, code);
}

void OeipCamera::onReviceHandle(uint8_t * data, int width, int height) {
	OnDeviceDataEvent.Broadcast(data, width, height);
}

void OeipCamera::SetDevice(FCameraInfo * cameraInfo) {
	id = cameraInfo->index;
	deviceName = cameraInfo->deviceName;
	deviceId = cameraInfo->deviceId;
	if (bInit()) {
		std::function<void(int, int)> deviceHandle = std::bind(&OeipCamera::onDeviceHandle, this, _1, _2);
		setDeviceEventHandle(id, deviceHandle);
		std::function<void(uint8_t*, int, int)> deviceDataHandle = std::bind(&OeipCamera::onReviceHandle, this, _1, _2, _3);
		setDeviceDataHandle(id, deviceDataHandle);
	}
}

int OeipCamera::GetFormat() {
	if (!bInit())
		return false;
	return getFormat(id);
}

void OeipCamera::SetFormat(int index) {
	if (!bInit())
		return;
	setFormat(id, index);
}

bool OeipCamera::Open() {
	if (!bInit())
		return false;
	return openDevice(id);
}

void OeipCamera::Close() {
	if (!bInit())
		return;
	closeDevice(id);
}

bool OeipCamera::IsOpen() {
	if (!bInit())
		return false;
	return bOpen(id);
}

int OeipCamera::FindFormatIndex(int width, int height, int fps) {
	if (!bInit())
		return -1;
	return findFormatIndex(id, width, height, fps);
}

TArray<VideoFormat> OeipCamera::GetFormatList() {
	if (bInit() && formatList.Num() == 0) {
		formatList = OeipManager::Get().GetCameraFormatList(id);
	}
	return formatList;
}
