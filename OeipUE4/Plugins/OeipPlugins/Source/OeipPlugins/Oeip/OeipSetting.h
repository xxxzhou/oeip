// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/WorldSettings.h"
#include "Oeip.h"
#include "ObjAttribute.h"
#include "OeipSetting.generated.h"

UENUM(BlueprintType)
enum class EOeipSettingType : uint8
{
	Device,
	GrabCut,
	LiveRoom,
};

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
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		bool bDraw = true;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		int softness = 5;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		float epslgn10 = 5.f;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		float intensity = 0.2f;
};

USTRUCT(BlueprintType)
struct FLiveRoom
{
	GENERATED_BODY()
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		FString roomName = "oeiplive";
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		FString userIndex = "31";
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
struct FDeviceSetting
{
	GENERATED_BODY()
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		int cameraIndex = 0;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		int formatIndex = 0;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		int rateIndex = 0;
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
		FDeviceSetting cameraSetting = {};
};

USTRUCT(BlueprintType)
struct FPipeTex
{
	GENERATED_BODY()
public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		int layerIndex;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		int texIndex;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Oeip)
		UTexture2D* tex;
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
public:
	FOeipSetting setting = {};
private:
	OeipSetting();
	static OeipSetting *singleton;
private:
	FString fileName = L"OeipSetting";

	TArray<FString> videoTypeList;
	TArray<FString> rateNameList;
	TArray<int> rateList;
	TArray<BaseAttribute*> deviceArrList;
	TArray<BaseAttribute*> grabCutArrList;
	TArray<BaseAttribute*> roomArrList;
private:
	void loadJson();
	TArray<FString> GetCameraFormat(int index);
public:
	void SaveJson();
	//得到设备UI描述信息
	TArray<BaseAttribute*> GetDeviceAttribute();
	//得到GrabCut扣图UI描述信息
	TArray<BaseAttribute*> GetGrabCutAttribute();
	//得到登陆房间的UI描述信息
	TArray<BaseAttribute*> GetRoomAttribute();
	int GetRate(int index);
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
void copycharstr(char* dest, const char* source, int32_t maxlength);