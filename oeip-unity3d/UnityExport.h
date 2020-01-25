#pragma once

#include "IUnityGraphics.h"
#include <stdint.h>

typedef void (UNITY_INTERFACE_API* UnityRenderingEvent)(int eventId);

#define UNITY_EXPORT UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API

struct PipeTex
{
	int32_t pipeId;
	int32_t layerIndex;
	int32_t texIndex;
	void* tex;
};

#ifdef __cplusplus
extern "C" {
#endif

	// If exported by a plugin, this function will be called when the plugin is loaded.
	void UNITY_EXPORT UnityPluginLoad(IUnityInterfaces* unityInterfaces);
	// If exported by a plugin, this function will be called when the plugin is about to be unloaded.
	void UNITY_EXPORT UnityPluginUnload();
	void UNITY_EXPORT OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);
	//给Unity3D游戏主线程调用
	void UNITY_EXPORT SetPipeInputGpuTex(int32_t pipeId, int32_t layerIndex, void* tex, int32_t inputIndex = 0);
	void UNITY_EXPORT SetPipeOutputGpuTex(int32_t pipeId, int32_t layerIndex, void* tex, int32_t outputIndex = 0);
	void UNITY_EXPORT UpdateTex(int32_t pipeId);
	void UNITY_EXPORT SetUpdateTex(int32_t pipeId);
	//给Unity3D游戏渲染线程调用
	UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API GetUpdateTexFunc();
	UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API SetUpdateTexFunc();

#ifdef __cplusplus
}
#endif