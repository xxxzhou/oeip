// Fill out your copyright notice in the Description page of Project Settings.

#include "OeipSetting.h"
#include "Engine/Texture2D.h"
#include "Runtime/Json/Public/Json.h"
#include "Runtime/JsonUtilities/Public/JsonUtilities.h"
#include "Runtime/JsonUtilities/Public/JsonObjectConverter.h"

using namespace std::placeholders;
OeipSetting *OeipSetting::singleton = nullptr;

OeipSetting::OeipSetting() {
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

void OeipSetting::SaveJson()
{
	FString filePath = FPaths::ProjectDir() + FString("Saved/") + FString("SaveGames/") + fileName + FString(".Json");
	FString json = "";
	bool bresult = FJsonObjectConverter::UStructToJsonObjectString(setting, json);
	if (bresult)
		FFileHelper::SaveStringToFile(json, *filePath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
}

OeipSetting::~OeipSetting() {
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
