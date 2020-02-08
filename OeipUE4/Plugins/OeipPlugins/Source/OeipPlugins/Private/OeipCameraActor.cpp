// Fill out your copyright notice in the Description page of Project Settings.

#include "OeipCameraActor.h"
#include "OeipManager.h"

// Sets default values
AOeipCameraActor::AOeipCameraActor() {
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
}

void AOeipCameraActor::SettingChange(const EOeipSettingType & settingType, const FString & name) {
	if (settingType == EOeipSettingType::Device) {
		FDeviceSetting& deviceSetting = OeipSetting::Get().setting.cameraSetting;
		if (deviceSetting.cameraIndex < 0) {
			CameraShow->SetTexture(nullTex);
			return;
		}
		//如果更新了摄像机
		if ((name == L"cameraIndex" || name == L"formatIndex") && deviceSetting.cameraIndex >= 0) {
			//先把原来摄像机关闭
			oeipCamera->Close();
			//重新设置
			auto* cameraInfo = OeipManager::Get().GetCamera(deviceSetting.cameraIndex);
			oeipCamera->SetDevice(cameraInfo);
			oeipCamera->SetFormat(deviceSetting.formatIndex);
			changeFormat();
			oeipCamera->Open();
		}
	}
	else if (settingType == EOeipSettingType::GrabCut) {
		if (videoPipe) {
			videoPipe->updateVideoParamet(&(OeipSetting::Get().setting.grabSetting));
		}
	}
}

void AOeipCameraActor::LoadNet() {
	//网络加载比较费时
	AsyncTask(ENamedThreads::AnyThread, [=]()
	{
		DarknetParamet darknetParamet = {};
		darknetParamet.bLoad = 1;
		copycharstr(darknetParamet.confile, "D:/WorkSpace/github/oeip/ThirdParty/yolov3-tiny-test.cfg", 512);
		copycharstr(darknetParamet.weightfile, "D:/WorkSpace/github/oeip/ThirdParty/yolov3-tiny_745000.weights", 512);
		darknetParamet.thresh = 0.3f;
		darknetParamet.nms = 0.3f;
		darknetParamet.bDraw = 1;
		darknetParamet.drawColor = getColor(1.0f, 0.1f, 0.1f, 0.1f);
		videoPipe->updateDarknet(darknetParamet);
	});
}

void AOeipCameraActor::GrabCut(bool bSeedMode) {
	if (personBoxs.Num() > 0)
		videoPipe->changeGrabcutMode(bSeedMode, personBoxs[0].rect);
}

void AOeipCameraActor::onLogMessage(int level, FString message) {
	AsyncTask(ENamedThreads::GameThread, [=]()
	{
		FColor color = FColor::Blue;
		if (level == 1)
			color = FColor::Yellow;
		else if (level > 1)
			color = FColor::Red;
		GEngine->AddOnScreenDebugMessage(-1, 5.f, color, message);
		UE_LOG(LogTemp, Log, TEXT("%s"), *message);
	});
}

void AOeipCameraActor::changeFormat() {
	//得到设备选择的数据大小
	int formatIndex = oeipCamera->GetFormat();
	VideoFormat vformat = {};
	bool bGet = oeipCamera->GetFormat(vformat);
	if (bGet && videoPipe) {
		videoPipe->setVideoFormat(vformat.videoType, vformat.width, vformat.height);
	}
	updateTexture(&cameraTex, vformat.width, vformat.height);
	CameraShow->SetTexture(cameraTex);
}

void AOeipCameraActor::onReviceHandle(uint8 * data, int width, int height) {
	if (videoPipe) {
		videoPipe->runVideoPipe(data);
	}
}

void AOeipCameraActor::onPipeDataHandle(int32_t layerIndex, uint8_t * data, int32_t width, int32_t height, int32_t outputIndex) {
	if (layerIndex == videoPipe->getDarknetId()) {
		//AsyncTask(ENamedThreads::GameThread, [=]()
		//{
		//	FString personMsg = "Person:" + FString::FromInt(width) + " ";
		//	personBoxs.SetNum(width);
		//	if (width > 0) {
		//		memcpy(personBoxs.GetData(), data, sizeof(PersonBox)*width);
		//		for (auto& person : personBoxs) {
		//			personMsg += FString::SanitizeFloat(person.prob, 2) + " ";
		//		}
		//	}
		//	OnPeronChange.Broadcast(personMsg);
		//});
	}
}

// Called when the game starts or when spawned
void AOeipCameraActor::BeginPlay() {
	//设备管理类
	oeipCamera = new OeipCamera();
	//这个绑定会自动没有?
	//cameraReviceHandle = oeipCamera->OnDeviceDataEvent.AddUObject(this, &AOeipCameraActor::onReviceHandle);
	cameraReviceHandle = oeipCamera->OnDeviceDataEvent.AddUObject(this, &AOeipCameraActor::onReviceHandle);
	//设备处理类
	gpuPipe = OeipManager::Get().CreatePipe(OeipGpgpuType::OEIP_CUDA);
	videoPipe = new VideoPipe(gpuPipe);
	pipeDataHandle = gpuPipe->OnOeipDataEvent.AddUObject(this, &AOeipCameraActor::onPipeDataHandle);
	OeipManager::Get().OnLogEvent.AddUObject(this, &AOeipCameraActor::onLogMessage);
	Super::BeginPlay();
}

void AOeipCameraActor::EndPlay(const EEndPlayReason::Type EndPlayReason) {
	Super::EndPlay(EndPlayReason);
	OeipManager::Get().OnLogEvent.RemoveAll(this);
	if (gpuPipe)
		gpuPipe->OnOeipDataEvent.Remove(pipeDataHandle);
	if (oeipCamera)
		oeipCamera->OnDeviceDataEvent.Remove(cameraReviceHandle);
	safeDelete(videoPipe);
	safeDelete(oeipCamera);
}

// Called every frame
void AOeipCameraActor::Tick(float DeltaTime) {
	Super::Tick(DeltaTime);
	if (oeipCamera && oeipCamera->IsOpen()) {
		//告诉运行管线要更新的UE4纹理
		gpuPipe->UpdateOutputGpuTex(videoPipe->getOutputId(), cameraTex);
	}
	if (gpuPipe && !gpuPipe->OnOeipDataEvent.IsBoundToObject(this)) {
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Yellow, L"lost pipe data handle");
	}
	if (oeipCamera && !oeipCamera->OnDeviceDataEvent.IsBoundToObject(this)) {
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Yellow, L"lost camera data handle");
	}
}

