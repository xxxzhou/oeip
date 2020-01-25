// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/WorldSettings.h"
#include "Oeip.h"
#include "OeipSetting.generated.h"

USTRUCT(BlueprintType)
struct FGrabCutSetting
{
	GENERATED_BODY()
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		bool bGpuSeed = false;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		int iterCount = 1;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		int seedCount = 1000;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		int	flowCount = 250;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		float gamma = 90.f;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		float lambda = 450.f;
};

USTRUCT(BlueprintType)
struct FLiveRoom
{
	GENERATED_BODY()
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		FString roomName = "";
};

USTRUCT(BlueprintType)
struct FNetSetting
{
	GENERATED_BODY()
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		FString netPath = "";
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		FString weightPath = "";
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		int fpsInterval = 1;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		float thresh = 0.3f;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		float nms = 0.4f;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		int timeDelay = 500;
};

USTRUCT(BlueprintType)
struct FCameraInfo
{
	GENERATED_BODY()
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		int index = -1;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		FString deviceName = "";
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		FString deviceId = "";
};

USTRUCT(BlueprintType)
struct FOeipSetting
{
	GENERATED_BODY()
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		FGrabCutSetting grabSetting;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		FNetSetting netSetting = {};
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		FLiveRoom roomSetting = {};
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		FCameraInfo cameraInfo = {};
};

/**
 *
 */
class OEIPPLUGINS_API OeipSetting
{
public:
	~OeipSetting();
	static OeipSetting& Get();
	static void Close();
private:
	OeipSetting();
	static OeipSetting *singleton;
private:
	FString fileName = L"OeipSetting";
	FOeipSetting setting = {};
	void loadJson();
public:
	void SaveJson();
};

template <typename T>
void safeDelete(T *&ppT) {
	if (ppT != nullptr) {
		delete ppT;
		ppT = nullptr;
	}
}

template<typename T>
void clearList(TArray<T*> list) {
	for (T* t : list) {
		safeDelete(t);
	}
	list.Empty();
}

void updateTexture(UTexture2D ** ptexture, int width, int height);
