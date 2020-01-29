// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "OeipSetting.h"
#include "OeipExport.h"
#include "OeipCamera.h"
#include "OeipPipe.h"

DECLARE_MULTICAST_DELEGATE_TwoParams(FOeipLogEvent, int, FString);
/**
 *
 */
class OEIPPLUGINS_API OeipManager
{
public:
	~OeipManager();
	static OeipManager& Get();
	static void Close();
private:
	OeipManager();
	static OeipManager *singleton;
private:
	TArray<FCameraInfo*> cameraList;
	//TArray<OeipCamera*> cameraArray;
	TArray<OeipPipe*> pipeList;
public:
	FOeipLogEvent OnLogEvent;
public:
	TArray<FCameraInfo*> GetCameraList();
	FCameraInfo* GetCamera(int index);
	OeipPipe* CreatePipe(OeipGpgpuType gpgpuType);
	TArray<VideoFormat> GetCameraFormatList(int cameraIndex);
	FString GetLiveServer();
private:
	void onLogMessage(int level, const char *message);
};
