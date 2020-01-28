// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "OeipDisplayActor.h"
#include "LivePullPipe.h"
#include "LivePushPipe.h"
#include "OeipDisplayActor.h"
#include "OeipSetting.h"
#include "OeipLiveActor.generated.h"

UCLASS()
class OEIPPLUGINS_API AOeipLiveActor : public AActor
{
	GENERATED_BODY()
public:
	// Sets default values for this actor's properties
	AOeipLiveActor();
private:
	OeipYUVFMT yuvFmt = OEIP_YUVFMT_YUV420P;
	bool IsPullSelf = false;
	OeipPipe* pullPipe = nullptr;
	OeipPipe* pushPipe = nullptr;
	LivePullPipe* livePullPipe = nullptr;
	LivePushPipe* livePushPipe = nullptr;
public:
	//后期不放出来，前期主要在编辑器里查看
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		UTexture2D *liveTex;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		AOeipDisplayActor *LiveShow;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		UTexture2D *nullTex;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		UTexture *uePushTex;
private:
	void onPipeDataHandle(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t outputIndex);
	void onPullTexChange(int width, int height);
	void onLoginRoom(int32_t code, int32_t userId);
	void onStreamUpdate(int32_t userId, int32_t index, bool bAdd);
	void onVideoFrame(int32_t userId, int32_t index, OeipVideoFrame videoFrame);
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;
public:
	UFUNCTION(BlueprintCallable)
		void SetPushTex(UTextureRenderTarget2D* tex);
	UFUNCTION(BlueprintCallable)
		void LoginRoom(FString roomName, int userId);
	UFUNCTION(BlueprintCallable)
		void LogoutRoom();
};
