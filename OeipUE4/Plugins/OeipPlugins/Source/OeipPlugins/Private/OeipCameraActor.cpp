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
			if (oeipCamera != nullptr) {
				oeipCamera->Close();
			}
			oeipCamera = OeipManager::Get().GetCamera(deviceSetting.cameraIndex);
			//如果没绑定过事件
			if (oeipCamera != nullptr) {
				if (!oeipCamera->OnDeviceDataEvent.IsBound()) {
					oeipCamera->OnDeviceDataEvent.AddUObject(this, &AOeipCameraActor::onReviceHandle);
				}
				if (deviceSetting.formatIndex >= 0)
					oeipCamera->SetFormat(deviceSetting.formatIndex);
				changeFormat();
				oeipCamera->Open();
			}
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
	});
}

void AOeipCameraActor::changeFormat() {
	if (gpuPipe == nullptr) {
		gpuPipe = OeipManager::Get().CreatePipe(OeipGpgpuType::OEIP_CUDA);
		//声明一条封装设备处理的管线
		if (gpuPipe != nullptr) {
			videoPipe = new VideoPipe(gpuPipe);
		}
		gpuPipe->OnOeipDataEvent.AddUObject(this, &AOeipCameraActor::onPipeDataHandle);
	}
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

void AOeipCameraActor::onReviceHandle(uint8_t * data, int width, int height) {
	if (videoPipe) {
		videoPipe->runVideoPipe(data);
	}
}

void AOeipCameraActor::onPipeDataHandle(int32_t layerIndex, uint8_t * data, int32_t width, int32_t height, int32_t outputIndex) {
	if (layerIndex == videoPipe->getDarknetId()) {
		AsyncTask(ENamedThreads::GameThread, [=]()
		{
			FString personMsg = "Person:" + FString::FromInt(width) + " ";
			personBoxs.SetNum(width);
			if (width > 0) {
				memcpy(personBoxs.GetData(), data, sizeof(PersonBox)*width);
				for (auto& person : personBoxs) {
					personMsg += FString::SanitizeFloat(person.prob) + " ";
				}
			}
			OnPeronChange.Broadcast(personMsg);
		});
	}
}

// Called when the game starts or when spawned
void AOeipCameraActor::BeginPlay() {
	Super::BeginPlay();
	OeipManager::Get().OnLogEvent.AddUObject(this, &AOeipCameraActor::onLogMessage);
}

void AOeipCameraActor::EndPlay(const EEndPlayReason::Type EndPlayReason) {
	Super::EndPlay(EndPlayReason);
	OeipManager::Get().OnLogEvent.RemoveAll(this);
	if (gpuPipe && gpuPipe->OnOeipDataEvent.IsBound())
		gpuPipe->OnOeipDataEvent.RemoveAll(this);
	if (oeipCamera && oeipCamera->OnDeviceDataEvent.IsBound())
		oeipCamera->OnDeviceDataEvent.RemoveAll(this);
	safeDelete(videoPipe);
}

// Called every frame
void AOeipCameraActor::Tick(float DeltaTime) {
	Super::Tick(DeltaTime);
	if (oeipCamera && oeipCamera->IsOpen()) {
		//告诉运行管线要更新的UE4纹理
		gpuPipe->UpdateOutputGpuTex(videoPipe->getOutputId(), cameraTex);
	}
}

