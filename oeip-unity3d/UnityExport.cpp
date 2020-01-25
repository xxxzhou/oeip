#include "UnityExport.h"
#include <d3d11.h>

#include "IUnityGraphicsD3D11.h"
#include <atlbase.h>
#include <map>
#include <memory>
#include "../oeip/OeipExport.h"
#include "../oeip/OeipCommon.h"

static IUnityInterfaces* s_UnityInterfaces = nullptr;
static IUnityGraphics* s_Graphics = nullptr;
static UnityGfxRenderer s_DeviceType = kUnityGfxRendererNull;

static CComPtr<ID3D11Device> g_D3D11Device = nullptr;
static std::vector<PipeTex> inPipeTexs;
static std::vector<PipeTex> outPipeTexs;
//static CComPtr<ID3D11DeviceContext>  g_pContext = nullptr;

void UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unityInterfaces) {
	s_UnityInterfaces = unityInterfaces;
	s_Graphics = s_UnityInterfaces->Get<IUnityGraphics>();
	s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

	// Run OnGraphicsDeviceEvent(initialize) manually on plugin load
	OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
}

void UNITY_INTERFACE_API UnityPluginUnload() {
	s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
	inPipeTexs.clear();
	outPipeTexs.clear();
}

void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType) {
	switch (eventType) {
	case kUnityGfxDeviceEventInitialize: {
		s_DeviceType = s_Graphics->GetRenderer();
		if (s_DeviceType != kUnityGfxRendererD3D11) {
			logMessage(OEIP_ERROR, "only support for the time being dx11.");
		}
		IUnityGraphicsD3D11* d3d11 = s_UnityInterfaces->Get<IUnityGraphicsD3D11>();
		g_D3D11Device = d3d11->GetDevice();
		//g_D3D11Device->GetImmediateContext(&g_pContext);
		break;
	}
	case kUnityGfxDeviceEventShutdown: {
		s_DeviceType = kUnityGfxRendererNull;
		break;
	}
	default:
		break;
	};
}

void UNITY_INTERFACE_API SetPipeInputGpuTex(int32_t pipeId, int32_t layerIndex, void* tex, int32_t inputIndex) {
	for (auto& pipeTex : inPipeTexs) {
		if (pipeTex.pipeId == pipeId && pipeTex.layerIndex == layerIndex && pipeTex.texIndex == inputIndex) {
			pipeTex.tex = tex;
			return;
		}
	}
	PipeTex pipeTex = {};
	pipeTex.pipeId = pipeId;
	pipeTex.layerIndex = layerIndex;
	pipeTex.texIndex = inputIndex;
	pipeTex.tex = tex;
	inPipeTexs.push_back(pipeTex);
}

void UNITY_INTERFACE_API SetPipeOutputGpuTex(int32_t pipeId, int32_t layerIndex, void* tex, int32_t outputIndex) {
	for (auto& pipeTex : outPipeTexs) {
		if (pipeTex.pipeId == pipeId && pipeTex.layerIndex == layerIndex && pipeTex.texIndex == outputIndex) {
			pipeTex.tex = tex;
			return;
		}
	}
	PipeTex pipeTex = {};
	pipeTex.pipeId = pipeId;
	pipeTex.layerIndex = layerIndex;
	pipeTex.texIndex = outputIndex;
	pipeTex.tex = tex;
	outPipeTexs.push_back(pipeTex);
}

void UNITY_INTERFACE_API UpdateTex(int32_t pipeId) {
	for (const auto& pipeTex : outPipeTexs) {
		if (pipeTex.pipeId == pipeId) {
			updatePipeOutputGpuTex(pipeTex.pipeId, pipeTex.layerIndex, g_D3D11Device, pipeTex.tex, pipeTex.texIndex);
		}
	}
}

void UNITY_EXPORT SetUpdateTex(int32_t pipeId) {
	for (const auto& pipeTex : inPipeTexs) {
		if (pipeTex.pipeId == pipeId) {
			updatePipeInputGpuTex(pipeTex.pipeId, pipeTex.layerIndex, g_D3D11Device, pipeTex.tex, pipeTex.texIndex);
		}
	}
}

UnityRenderingEvent UNITY_INTERFACE_API GetUpdateTexFunc() {
	return UpdateTex;
}

UnityRenderingEvent UNITY_INTERFACE_API SetUpdateTexFunc() {
	return SetUpdateTex;
}

