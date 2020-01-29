// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "OeipSetting.h"
#include "Runtime/Core/Public/Core.h"
#include "PanelManager.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FSettingChangeEvnet, EOeipSettingType, settingType, FString, name);
/**
 *
 */
UCLASS()
class OEIPPLUGINS_API UPanelManager : public UUserWidget
{
	GENERATED_BODY()
private:
	TArray<UClass*> templateList;
	ObjAttribute<FDeviceSetting> objDevice = {};
	ObjAttribute<FGrabCutSetting> objGrabcut = {};
	ObjAttribute<FLiveRoom> objLiveRoom = {};
public:
	UPROPERTY(BlueprintAssignable, Category = Oeip)
		FSettingChangeEvnet onSettingChangeEvent;
private:
	void OnKeyValue(ObjAttribute<FGrabCutSetting>* objAttribut, FString name);
	void OnDeviceValue(ObjAttribute<FDeviceSetting>* objAttribut, FString name);
	void OnLiveRoom(ObjAttribute<FLiveRoom>* objAttribut, FString name);
public:
	UFUNCTION(BlueprintCallable)
		void InitTemplate(UClass * toggleTemplate, UClass * inputeTemplate, UClass * sliderTemplate, UClass * dropdownTemplate);
	UFUNCTION(BlueprintCallable)
		void BindDevice(UVerticalBox * keyBox);
	UFUNCTION(BlueprintCallable)
		void BindGrabCut(UVerticalBox * keyBox);
	UFUNCTION(BlueprintCallable)
		void BindLiveRoom(UVerticalBox * keyBox);
	//蓝图里优先调用OnSettingChangeEvent重载实现，如果没有，调用OnSettingChangeEvent_Implementation
	//UFUNCTION(BlueprintNativeEvent)
	//	void OnSettingChangeEvent(const EOeipSettingType& settingType, const FString& name);
	//virtual void OnSettingChangeEvent_Implementation(const EOeipSettingType& settingType, const FString& name) {
	//	onSettingChangeEvent.Broadcast(settingType, name);
	//};
};
