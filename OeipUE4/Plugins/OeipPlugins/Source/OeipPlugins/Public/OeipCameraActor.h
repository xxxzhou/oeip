// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "OeipDisplayActor.h"
#include "OeipCamera.h"
#include "VideoPipe.h"
#include "OeipDisplayActor.h"
#include "OeipSetting.h"
#include "OeipCameraActor.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPersonChangeEvnet, FString, msg);

UCLASS()
class OEIPPLUGINS_API AOeipCameraActor : public AActor
{
	GENERATED_BODY()

public:
	// Sets default values for this actor's properties
	AOeipCameraActor();
private:
	bool bDraw = false;
	OeipCamera* oeipCamera = nullptr;
	OeipPipe* gpuPipe = nullptr;
	VideoPipe* videoPipe = nullptr;
	TArray<PersonBox> personBoxs;
	//UTexture2D *cameraTex;
	FDelegateHandle cameraReviceHandle = {};
public:
	//后期不放出来，前期主要在编辑器里查看
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		UTexture2D *cameraTex;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		AOeipDisplayActor *CameraShow;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		UTexture2D *nullTex;
	UPROPERTY(BlueprintAssignable, Category = Oeip)
		FOnPersonChangeEvnet OnPeronChange;
public:
	UFUNCTION(BlueprintCallable)
		void SettingChange(const EOeipSettingType& settingType, const FString& name);
	UFUNCTION(BlueprintCallable)
		void LoadNet();
	UFUNCTION(BlueprintCallable)
		void GrabCut(bool bSeedMode);
	void onLogMessage(int level, FString message);
	////给蓝图去显示人数变化
	//UFUNCTION(BlueprintNativeEvent)
	//	void OnPeronChange(const FString& message);
	//virtual void OnPeronChange_Implementation(const FString& message) {
	//};
private:
	void onReviceHandle(uint8* data, int width, int height);
	void onPipeDataHandle(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t outputIndex);
	void changeFormat();
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;
};
