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
	//bool IsPullSelf = false;
	OeipPipe* pullPipe = nullptr;
	OeipPipe* pushPipe = nullptr;
	LivePullPipe* livePullPipe = nullptr;
	LivePushPipe* livePushPipe = nullptr;
	bool bPush = false;
public:
	//后期不放出来，前期主要在编辑器里查看
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		UTexture2D *liveTex = nullptr;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		AOeipDisplayActor *LiveShow;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		UTexture2D *nullTex;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		UTextureRenderTarget2D *uePushTex;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		bool IsPullSelf = false;
private:
	//推流管线处理数据后让直播模块推出去
	void onPipeDataHandle(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t outputIndex);
	//拉流得到的大小变化
	void onPullTexChange(int width, int height);
	//用户登陆
	void onLoginRoom(int32_t code, int32_t userId);
	//有流更新的情况
	void onStreamUpdate(int32_t userId, int32_t index, bool bAdd);
	//拉流的视频流更新
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
