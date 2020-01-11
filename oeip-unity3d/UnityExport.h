#pragma once

#include "IUnityGraphics.h"
#include <stdint.h>

typedef void (UNITY_INTERFACE_API * UnityRenderingEvent)(int eventId);

#define UNITY_EXPORT UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API

#ifdef __cplusplus
extern "C" {
#endif

	// If exported by a plugin, this function will be called when the plugin is loaded.
	void UNITY_EXPORT UnityPluginLoad(IUnityInterfaces* unityInterfaces);
	// If exported by a plugin, this function will be called when the plugin is about to be unloaded.
	void UNITY_EXPORT UnityPluginUnload();
	void UNITY_EXPORT OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);

	void UNITY_EXPORT SetPipeInputGpuTex(int32_t pipeId, int32_t layerIndex, void* tex, int32_t inputIndex = 0);
	void UNITY_EXPORT SetPipeOutputGpuTex(int32_t pipeId, int32_t layerIndex, void* tex, int32_t outputIndex = 0);
	void UNITY_EXPORT RunPipe(int32_t pipeId);
	UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API GetRunPipeFunc();

#ifdef __cplusplus
}
#endif