// Fill out your copyright notice in the Description page of Project Settings.

#include "OeipSetting.h"
#include "OeipManager.h"
#include "Engine/Texture2D.h"
#include "Runtime/Json/Public/Json.h"
#include "Runtime/JsonUtilities/Public/JsonUtilities.h"
#include "Runtime/JsonUtilities/Public/JsonObjectConverter.h"

using namespace std::placeholders;
OeipSetting *OeipSetting::singleton = nullptr;

OeipSetting::OeipSetting() {
	rateNameList.Push("20M");
	rateNameList.Push("8M");
	rateNameList.Push("6M");
	rateNameList.Push("4M");
	rateNameList.Push("2M");
	rateNameList.Push("1M");
	rateNameList.Push("512K");

	rateList.Push(20000000);
	rateList.Push(8000000);
	rateList.Push(6000000);
	rateList.Push(4000000);
	rateList.Push(2000000);
	rateList.Push(1000000);
	rateList.Push(512000);

	videoTypeList.Push("Other");
	videoTypeList.Push("NV12");
	videoTypeList.Push("YUY2");
	videoTypeList.Push("YVYU");
	videoTypeList.Push("UYVY");
	videoTypeList.Push("MJPG");
	videoTypeList.Push("RGB24");
	videoTypeList.Push("ARGB32");
	videoTypeList.Push("RGBA32");
	videoTypeList.Push("Depth");
}

void OeipSetting::loadJson() {
	FString filePath = FPaths::ProjectDir() + FString("Saved/") + FString("SaveGames/") + fileName + FString(".Json");
	FString json = "";
	if (FPaths::FileExists(filePath) && FFileHelper::LoadFileToString(json, *filePath)) {
		bool bresult = FJsonObjectConverter::JsonObjectStringToUStruct(json, &setting, 0, 0);
		if (bresult) {
		}
	}
}

TArray<FString> OeipSetting::GetCameraFormat(int index) {
	TArray<FString> formatOption;
	auto formats = OeipManager::Get().GetCameraFormatList(index);
	for (auto format : formats) {
		FString option = FString::FromInt(format.width) + "*" + FString::FromInt(format.height) + " " + FString::FromInt(format.fps) + "fps " + videoTypeList[format.videoType];
		formatOption.Push(option);
	}
	return formatOption;
}

void OeipSetting::SaveJson() {
	FString filePath = FPaths::ProjectDir() + FString("Saved/") + FString("SaveGames/") + fileName + FString(".Json");
	FString json = "";
	bool bresult = FJsonObjectConverter::UStructToJsonObjectString(setting, json);
	if (bresult)
		FFileHelper::SaveStringToFile(json, *filePath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
}

TArray<BaseAttribute*> OeipSetting::GetDeviceAttribute() {
	if (deviceArrList.Num() == 0) {
		TArray<FCameraInfo*> cameraList = OeipManager::Get().GetCameraList();
		TArray<FString> cameraOptions;
		for (auto camera : cameraList)
		{
			FString option = FString::FromInt(camera->index) + " " + camera->deviceName;
			cameraOptions.Push(option);
		}
		std::function<TArray<FString>(int)> getFormatHandle = std::bind(&OeipSetting::GetCameraFormat, this, _1);

		DropdownAttribute* mcDrop = new DropdownAttribute();
		mcDrop->MemberName = "cameraIndex";
		mcDrop->DisplayName = "Camera";
		mcDrop->InitOptions(true, cameraOptions);
		deviceArrList.Add(mcDrop);

		DropdownAttribute* mfDrop = new DropdownAttribute();
		mfDrop->MemberName = "formatIndex";
		mfDrop->DisplayName = "Format";
		mfDrop->InitOptions(false, "cameraIndex", getFormatHandle);
		deviceArrList.Add(mfDrop);

		DropdownAttribute* mrDrop = new DropdownAttribute();
		mrDrop->MemberName = "rateIndex";
		mrDrop->DisplayName = "Rate";
		mrDrop->InitOptions(false, rateNameList);
		deviceArrList.Add(mrDrop);
	}
	return deviceArrList;
}

TArray<BaseAttribute*> OeipSetting::GetGrabCutAttribute() {
	if (grabCutArrList.Num() == 0) {
		ToggleAttribute* tst = new ToggleAttribute();
		tst->MemberName = "bGpuSeed";
		tst->DisplayName = "Is Gpu";
		tst->DefaultValue = false;
		grabCutArrList.Add(tst);

		SliderAttribute* av = new SliderAttribute();
		av->MemberName = "iterCount";
		av->DisplayName = "Iter Count";
		av->DefaultValue = 1;
		av->bInt = true;
		av->offset = 1;
		av->range = 10;
		grabCutArrList.Add(av);

		SliderAttribute* sc = new SliderAttribute();
		sc->MemberName = "seedCount";
		sc->DisplayName = "Seed Count";
		sc->DefaultValue = 1000;
		sc->bInt = true;
		sc->offset = 250;
		sc->range = 2000;
		grabCutArrList.Add(sc);

		SliderAttribute* fc = new SliderAttribute();
		fc->MemberName = "flowCount";
		fc->DisplayName = "Flow Count";
		fc->DefaultValue = 250;
		fc->bInt = true;
		fc->offset = 20;
		fc->range = 500;
		grabCutArrList.Add(fc);

		SliderAttribute* ga = new SliderAttribute();
		ga->MemberName = "gamma";
		ga->DisplayName = "Gamma";
		ga->DefaultValue = 90.f;
		ga->offset = 1.0f;
		ga->range = 200.f;
		grabCutArrList.Add(ga);

		SliderAttribute* ld = new SliderAttribute();
		ld->MemberName = "lambda";
		ld->DisplayName = "Lambda";
		ld->DefaultValue = 450.f;
		ld->offset = 50.0f;
		ld->range = 1000.0f;
		grabCutArrList.Add(ld);

		SliderAttribute* sn = new SliderAttribute();
		sn->MemberName = "softness";
		sn->DisplayName = "Softness";
		sn->DefaultValue = 5.f;
		sn->bInt = true;
		sn->offset = 1.0f;
		sn->range = 30.0f;
		grabCutArrList.Add(sn);

		SliderAttribute* eps = new SliderAttribute();
		eps->MemberName = "epslgn10";
		eps->DisplayName = "Eps Lg10";
		eps->DefaultValue = 5.f;
		eps->offset = 1.0f;
		eps->range = 10.0f;
		grabCutArrList.Add(eps);

		SliderAttribute* it = new SliderAttribute();
		it->MemberName = "intensity";
		it->DisplayName = "Intensity";
		it->DefaultValue = 0.2f;
		it->offset = 0.0f;
		it->range = 1.0f;
		grabCutArrList.Add(it);
	}
	return grabCutArrList;
}

TArray<BaseAttribute*> OeipSetting::GetRoomAttribute() {
	if (roomArrList.Num() == 0) {
		InputeAttribute* ib = new InputeAttribute();
		ib->MemberName = "roomName";
		ib->DisplayName = "Room Name";
		ib->DefaultValue = "oeiplive";
		roomArrList.Add(ib);

		InputeAttribute* ui = new InputeAttribute();
		ui->MemberName = "userIndex";
		ui->DisplayName = "User Index";
		ui->DefaultValue = "31";
		roomArrList.Add(ui);
	}
	return roomArrList;
}

int OeipSetting::GetRate(int index) {
	if (index < 0 || index >= rateList.Num())
		return 4000000;
	return rateList[index];
}

OeipSetting::~OeipSetting() {
	clearList(deviceArrList);
	clearList(grabCutArrList);
	clearList(roomArrList);
}

OeipSetting & OeipSetting::Get() {
	if (singleton == nullptr) {
		singleton = new OeipSetting();
		singleton->loadJson();
	}
	return *singleton;
}

void OeipSetting::Close() {
	safeDelete(singleton);
}

void updateTexture(UTexture2D ** ptexture, int width, int height) {
	UTexture2D * texture = *ptexture;
	bool bValid = texture && texture->IsValidLowLevel();
	bool bChange = false;
	if (bValid) {
		int twidth = texture->GetSizeX();
		int theight = texture->GetSizeY();
		bChange = (twidth != width) || (theight != height);
		if (bChange) {
			texture->RemoveFromRoot();
			texture->ConditionalBeginDestroy();
			texture = nullptr;
		}
	}
	if (!bValid || bChange) {
		*ptexture = UTexture2D::CreateTransient(width, height, PF_R8G8B8A8);
		(*ptexture)->UpdateResource();
		(*ptexture)->AddToRoot();
	}
}

void copycharstr(char* dest, const char* source, int32_t maxlength) {
	int length = sizeof(char) * (strlen(source) + 1);
	memcpy(dest, source, FMath::Min(length, maxlength));
}
