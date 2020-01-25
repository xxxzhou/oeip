// Fill out your copyright notice in the Description page of Project Settings.

#include "OeipHelperLibrary.h"


FString UOeipHelperLibrary::FStreamQualityToFString(const FCameraInfo& cameraInfo) {
	return FString::FromInt(cameraInfo.index) + " " + cameraInfo.deviceName;
}

