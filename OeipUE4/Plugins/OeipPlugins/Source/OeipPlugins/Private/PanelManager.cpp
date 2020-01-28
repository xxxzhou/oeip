// Fill out your copyright notice in the Description page of Project Settings.

#include "PanelManager.h"
#include "OeipManager.h"

void UPanelManager::OnKeyValue(ObjAttribute<FGrabCutSetting>* objAttribut, FString name) {
	onSettingChangeEvent.Broadcast(EOeipSettingType::GrabCut, name);
}

void UPanelManager::OnDeviceValue(ObjAttribute<FDeviceSetting>* objAttribut, FString name) {
	onSettingChangeEvent.Broadcast(EOeipSettingType::Device, name);
}

void UPanelManager::InitTemplate(UClass * toggleTemplate, UClass * inputeTemplate, UClass * sliderTemplate, UClass * dropdownTemplate) {
	templateList.Add(toggleTemplate);
	templateList.Add(inputeTemplate);
	templateList.Add(sliderTemplate);
	templateList.Add(dropdownTemplate);
}

void UPanelManager::BindDevice(UVerticalBox * keyBox) {
	auto word = this->GetWorld();
	objDevice.SetOnObjChangeAction(std::bind(&UPanelManager::OnDeviceValue, this, _1, _2));

	objDevice.Bind(&OeipSetting::Get().setting.cameraSetting, keyBox, OeipSetting::Get().GetDeviceAttribute(), templateList, word);
	objDevice.Update();
}

void UPanelManager::BindGrabCut(UVerticalBox * keyBox) {
	auto word = this->GetWorld();
	objGrabcut.SetOnObjChangeAction(std::bind(&UPanelManager::OnKeyValue, this, _1, _2));

	objGrabcut.Bind(&OeipSetting::Get().setting.grabSetting, keyBox, OeipSetting::Get().GetGrabCutAttribute(), templateList, word);
	objGrabcut.Update();
}
