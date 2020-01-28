// Fill out your copyright notice in the Description page of Project Settings.

#include "OeipPipe.h"

OeipPipe::OeipPipe() {
}

OeipPipe::~OeipPipe() {
	setPipeDataHandle(pipeId, nullptr);
	Close();
}

void OeipPipe::onPipeDataHandle(int32_t layerIndex, uint8_t * data, int32_t width, int32_t height, int32_t outputIndex) {
	OnOeipDataEvent.Broadcast(layerIndex, data, width, height, outputIndex);
}

void OeipPipe::SetPipeId(int id) {
	this->pipeId = id;
	std::function<void(int, uint8_t *, int, int, int)> pipeHandle = std::bind(&OeipPipe::onPipeDataHandle, this, _1, _2, _3, _4, _5);
	setPipeDataHandle(pipeId, pipeHandle);
	gpgpuType = getPipeType(pipeId);
}

void OeipPipe::Close() {
	closePipe(pipeId);
}

bool OeipPipe::IsEmpty() {
	return emptyPipe(pipeId);
}

int OeipPipe::AddLayer(FString layerName, OeipLayerType layerType) {
	return addPiepLayer(pipeId, TCHAR_TO_UTF8(*layerName), layerType);
}

void OeipPipe::ConnectLayer(int layerIndex, FString forwardName, int inputIndex, int selfIndex) {
	connectLayerName(pipeId, layerIndex, TCHAR_TO_UTF8(*forwardName), inputIndex, selfIndex);
}

void OeipPipe::ConnectLayer(int layerIndex, int forwardIndex, int inputIndex, int selfIndex) {
	connectLayerIndex(pipeId, layerIndex, forwardIndex, inputIndex, selfIndex);
}

void OeipPipe::SetEnableLayer(int layerIndex, bool bEnable) {
	setEnableLayer(pipeId, layerIndex, bEnable);
}

void OeipPipe::SetEnableLayerList(int layerIndex, bool bEnable) {
	setEnableLayerList(pipeId, layerIndex, bEnable);
}

void OeipPipe::SetInput(int layerIndex, int width, int height, int dataType, int inputIndex) {
	setPipeInput(pipeId, layerIndex, width, height, dataType, inputIndex);
}

void OeipPipe::UpdateInput(int layerIndex, uint8_t * data, int inputIndex) {
	updatePipeInput(pipeId, layerIndex, data, inputIndex);
}

void OeipPipe::UpdateInputGpuTex(int layerIndex, UTexture * tex, int inputIndex) {	
	ENQUEUE_UNIQUE_RENDER_COMMAND_FOURPARAMETER(
		UpdatePipeInputGpuTex,
		int, pipeId, pipeId,
		int, layerIndex, layerIndex,
		UTexture*, uTex, tex,
		int, texIndex, inputIndex,
		{
			if (pipeId < 0)
				return;
			void* device = RHICmdList.GetNativeDevice();
			if (!uTex->Resource || !uTex->Resource->TextureRHI)
				return;
			void* textureResource = uTex->Resource->TextureRHI->GetNativeResource();
			updatePipeInputGpuTex(pipeId, layerIndex, device, textureResource, texIndex);
		});
}

void OeipPipe::UpdateOutputGpuTex(int layerIndex, UTexture * tex, int inputIndex) {
	ENQUEUE_UNIQUE_RENDER_COMMAND_FOURPARAMETER(
		UpdatePipeOutputGpuTex,
		int, pipeId, pipeId,
		int, layerIndex, layerIndex,
		UTexture*, uTex, tex,
		int, texIndex, inputIndex,
		{
			if (pipeId < 0)
				return;
			void* device = RHICmdList.GetNativeDevice();
			if (!uTex->Resource || !uTex->Resource->TextureRHI)
				return;
			void* textureResource = uTex->Resource->TextureRHI->GetNativeResource();
			updatePipeOutputGpuTex(pipeId, layerIndex, device, textureResource, texIndex);
		});
}

bool OeipPipe::RunPipe() {
	return runPipe(pipeId);
}
