// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "OeipSetting.h"
#include "OeipExport.h"
#include <vector>

DECLARE_MULTICAST_DELEGATE_FiveParams(FOeipDataEvent, int32_t, uint8_t*, int32_t, int32_t, int32_t);
/**
 *
 */
class OEIPPLUGINS_API OeipPipe
{
public:
	OeipPipe();
	~OeipPipe();
private:
	int pipeId = -1;
	OeipGpgpuType gpgpuType = OeipGpgpuType::OEIP_DX11;
public:
	FOeipDataEvent OnOeipDataEvent;
private:
	//管线运行，各层如果有输出，相应的输出结果
	void onPipeDataHandle(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t outputIndex);
public:
	void SetPipeId(int id);
	void Close();
	bool IsEmpty();
	OeipGpgpuType GetGpuType() {
		return gpgpuType;
	}
	int AddLayer(FString layerName, OeipLayerType layerType);
	template<typename T>
	int AddLayer(FString layerName, OeipLayerType layerType, T& t) {
		return addPiepLayer(pipeId, TCHAR_TO_UTF8(*layerName), layerType, &t);
	};
	template<typename T>
	bool UpdateParamet(int layerIndex, T& t) {
		updatePipeParamet(pipeId, layerIndex, &t);
	};
	//bool UpdateParamet(int layerIndex, const void* t) {
	//	updatePipeParamet(pipeId, layerIndex, t);
	//}
	void ConnectLayer(int layerIndex, FString forwardName, int inputIndex = 0, int selfIndex = 0);
	void ConnectLayer(int layerIndex, int forwardIndex, int inputIndex = 0, int selfIndex = 0);
	void SetEnableLayer(int layerIndex, bool bEnable);
	void SetEnableLayerList(int layerIndex, bool bEnable);
	//(0 8UC1) (24 8UC3)
	void SetInput(int layerIndex, int width, int height, int dataType = 0, int inputIndex = 0);
	void UpdateInput(int layerIndex, uint8_t* data, int inputIndex = 0);
	//UE4的RHI资源当做输入，自动转入渲染线程
	void UpdateInputGpuTex(int layerIndex, UTexture2D* tex, int inputIndex = 0);
	void UpdateInputGpuTex(int layerIndex, UTextureRenderTarget2D* tex, int inputIndex = 0);
	//更新UE4渲染相应的RHI资源，自动转入渲染线程
	void UpdateOutputGpuTex(int layerIndex, UTexture* tex, int inputIndex = 0);
	void UpdateOutputGpuTex(int layerIndex, UTextureRenderTarget2D* tex, int inputIndex = 0);
	//运行管线，任何线程都可以，在设备线程最好
	bool RunPipe();
};
