// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "OeipSetting.h"
#include "OeipExport.h"

DECLARE_MULTICAST_DELEGATE_TwoParams(FDeviceEvent, int, int);
DECLARE_MULTICAST_DELEGATE_ThreeParams(FDeviceDataEvent, uint8_t*, int, int);
/**
 *
 */
class OEIPPLUGINS_API OeipCamera
{
public:
	OeipCamera();
	~OeipCamera();
public:
	//设备打开关闭中断等回调
	FDeviceEvent OnDeviceEvent;
	//设备返回对应格式数据
	FDeviceDataEvent OnDeviceDataEvent;
private:
	int id = -1;
	FString deviceName;
	FString deviceId;
	TArray<VideoFormat> formatList;
private:
	bool bInit();
	void onDeviceHandle(int type, int code);
	void onReviceHandle(uint8_t* data, int width, int height);
public:
	void SetDevice(FCameraInfo* cameraInfo);
	//返回摄像机的当前用的索引
	int GetFormat();
	bool GetFormat(VideoFormat& videoFormat);
	void SetFormat(int index);
	bool Open();
	void Close();
	bool IsOpen();
	int FindFormatIndex(int width, int height, int fps = 30);
	TArray<VideoFormat> GetFormatList();
};
