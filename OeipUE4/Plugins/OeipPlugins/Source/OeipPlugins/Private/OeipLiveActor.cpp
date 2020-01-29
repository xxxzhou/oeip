// Fill out your copyright notice in the Description page of Project Settings.

#include "OeipLiveActor.h"
#include "OeipManager.h"
#include "OeipLiveManager.h"
// Sets default values
AOeipLiveActor::AOeipLiveActor() {
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
}

void AOeipLiveActor::onPipeDataHandle(int32_t layerIndex, uint8_t * data, int32_t width, int32_t height, int32_t outputIndex) {
	//只推一个流，主流
	if (livePushPipe->GetOutputId() == layerIndex)
		OeipLiveManager::Get().PushVideoFrame(0, data, width, height, yuvFmt);
}

void AOeipLiveActor::onPullTexChange(int width, int height) {
	AsyncTask(ENamedThreads::GameThread, [=]()
	{
		updateTexture(&liveTex, width, height);
		LiveShow->SetTexture(liveTex);
	});
}

void AOeipLiveActor::onLoginRoom(int32_t code, int32_t userId) {
	OeipPushSetting setting = {};
	setting.bAudio = false;
	setting.bVideo = true;
	bPush = OeipLiveManager::Get().PushStream(0, setting);
}

void AOeipLiveActor::onStreamUpdate(int32_t userId, int32_t index, bool bAdd) {
	if (!IsPullSelf && OeipLiveManager::Get().GetUserId() == userId)
		return;
	if (bAdd) {
		OeipLiveManager::Get().PullStream(userId, index);
	}
	else {
		OeipLiveManager::Get().StopPullStream(userId, index);
	}
}

void AOeipLiveActor::onVideoFrame(int32_t userId, int32_t index, OeipVideoFrame videoFrame) {
	livePullPipe->RunPipe(videoFrame);
}

// Called when the game starts or when spawned
void AOeipLiveActor::BeginPlay() {
	Super::BeginPlay();
	pushPipe = OeipManager::Get().CreatePipe(OeipGpgpuType::OEIP_CUDA);
	livePushPipe = new LivePushPipe(pushPipe, yuvFmt);
	pushPipe->OnOeipDataEvent.AddUObject(this, &AOeipLiveActor::onPipeDataHandle);
	pullPipe = OeipManager::Get().CreatePipe(OeipGpgpuType::OEIP_DX11);
	livePullPipe = new LivePullPipe(pullPipe);
	livePullPipe->OnPullDataEvent.AddUObject(this, &AOeipLiveActor::onPullTexChange);
	//用户连接上服务器
	OeipLiveManager::Get().OnLoginRoomEvent.AddUObject(this, &AOeipLiveActor::onLoginRoom);
	//服务器通知用户服务器上对应房间有流的变化
	OeipLiveManager::Get().OnStreamUpdateEvent.AddUObject(this, &AOeipLiveActor::onStreamUpdate);
	//拉到视频流数据
	OeipLiveManager::Get().OnVideoFrameEvent.AddUObject(this, &AOeipLiveActor::onVideoFrame);
	//
	SetPushTex(uePushTex);
	LiveShow->SetTexture(nullTex);
}

void AOeipLiveActor::EndPlay(const EEndPlayReason::Type EndPlayReason) {
	OeipLiveManager::Get().LogoutRoom();
	OeipLiveManager::Get().OnLoginRoomEvent.RemoveAll(this);
	OeipLiveManager::Get().OnStreamUpdateEvent.RemoveAll(this);
	OeipLiveManager::Get().OnVideoFrameEvent.RemoveAll(this);
	safeDelete(livePushPipe);
	safeDelete(livePullPipe);
	Super::EndPlay(EndPlayReason);
}

// Called every frame
void AOeipLiveActor::Tick(float DeltaTime) {
	Super::Tick(DeltaTime);
	//更新输出结果到UE4纹理上
	if (liveTex) {
		pullPipe->UpdateOutputGpuTex(livePullPipe->getOutputId(), liveTex);
	}
	//把UE4的纹理当做输入
	if (bPush && uePushTex) {
		//uePushTex->TextureReference.TextureReferenceRHI->GetNativeResource();
		pushPipe->UpdateInputGpuTex(livePushPipe->GetInputId(), uePushTex);
		pushPipe->RunPipe();
	}
}

void AOeipLiveActor::SetPushTex(UTextureRenderTarget2D * tex) {
	//告诉这个管线 输入的格式	
	pushPipe->SetInput(livePushPipe->GetInputId(), tex->GetSurfaceWidth(), tex->GetSurfaceHeight(), OEIP_CV_8UC3);
	//uePushTex = tex;
}

void AOeipLiveActor::LoginRoom(FString roomName, int userId) {
	OeipLiveManager::Get().LoginRoom(roomName, userId);	
}

void AOeipLiveActor::LogoutRoom() {
	OeipLiveManager::Get().LogoutRoom();
	LiveShow->SetTexture(nullTex);
	bPush = false;
}