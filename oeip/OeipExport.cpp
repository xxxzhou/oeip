#include "OeipExport.h"
#include "OeipManager.h"
#include "PluginManager.h"
#include "VideoManager.h"
#include <math.h>

#pragma region check 
#define OEIP_CHECKINSTANCEVOID \
	if (!oInstance) \
		return;
#define OEIP_CHECKDEVICEVOID \
	auto device = oInstance->getVideoIndex(deviceIndex);\
	if (!device)\
		return;
#define OEIP_CHECKPIPEVOID \
	auto pipe = oInstance->getPipe(pipeId);\
	if (!pipe)\
		return;
#define OEIP_CHECKAUDIOVOID \
	auto audioOutput = oInstance->getAudioOutput();\
	if(!audioOutput)\
		return;

#define OEIP_CHECKINSTANCEINT \
	if (!oInstance) \
		return -1;
#define OEIP_CHECKDEVICEINT \
	auto device = oInstance->getVideoIndex(deviceIndex);\
	if (!device)\
		return 0;
#define OEIP_CHECKPIPEINT \
	auto pipe = oInstance->getPipe(pipeId);\
	if (!pipe)\
		return -1;

#define OEIP_CHECKINSTANCEBOOL \
	if (!oInstance) \
		return false;
#define OEIP_CHECKDEVICEBOOL \
	auto device = oInstance->getVideoIndex(deviceIndex);\
	if (!device)\
		return false;
#define OEIP_CHECKPIPEBOOL \
	auto pipe = oInstance->getPipe(pipeId);\
	if (!pipe)\
		return false;
#pragma endregion

static OeipManager* oInstance = nullptr;
static bool bInit = false;

void setLogAction(logEventAction logHandle) {
	setLogEvent(logHandle);
}

void setLogHandle(logEventHandle logHandle) {
	setLogEvent(logHandle);
}

void initOeip() {
	if (!bInit) {
		oInstance = OeipManager::getInstance();
		bInit = true;
	}
}

void shutdownOeip() {
	if (bInit) {
		OeipManager::shutdown();
		bInit = false;
	}
}

OeipYUVFMT getVideoYUV(OeipVideoType videoType) {
	OeipYUVFMT fmt = OEIP_YUVFMT_OTHER;
	switch (videoType) {
	case OEIP_VIDEO_NV12:
		fmt = OEIP_YUVFMT_YUV420SP;
		break;
	case OEIP_VIDEO_YUY2:
		fmt = OEIP_YUVFMT_YUY2I;
		break;
	case OEIP_VIDEO_YVYU:
		fmt = OEIP_YUVFMT_YVYUI;
		break;
	case OEIP_VIDEO_UYVY:
		fmt = OEIP_YUVFMT_UYVYI;
		break;
	case OEIP_VIDEO_MJPG:
		fmt = OEIP_YUVFMT_YUY2I;
		break;
	default:
		break;
	}
	return fmt;
}

uint32_t getColor(float r, float g, float b, float a) {
	return ((uint32_t)(a * 255.0f) << 24) |
		((uint32_t)(b * 255.0f) << 16) |
		((uint32_t)(g * 255.0f) << 8) |
		((uint32_t)(r * 255.0f));
}

int getDeviceCount() {
	OEIP_CHECKINSTANCEINT;
	return oInstance->getVideoList().size();
}

void getDeviceList(OeipDeviceInfo* deviceList, int32_t lenght, int32_t index) {
	OEIP_CHECKINSTANCEVOID;
	auto videoList = oInstance->getVideoList();
	if (index + lenght > videoList.size()) {
		logMessage(OEIP_INFO, "getDeviceList length and index out of range.");
	}
	for (int32_t i = index; i < lenght; i++) {
		if (i >= videoList.size())
			break;
		OeipDeviceInfo& device = deviceList[i - index];
		VideoDevice* capture = videoList[i];
		copywcharstr(device.deviceName, capture->getDeviceName(), sizeof(device.deviceName));
		copywcharstr(device.deviceId, capture->getDeviceID(), sizeof(device.deviceId));
		device.id = i;
	}
}

int32_t getFormatCount(int32_t deviceIndex) {
	OEIP_CHECKINSTANCEINT;
	OEIP_CHECKDEVICEINT;
	return device->getFormats().size();
}

int32_t getFormatList(int32_t deviceIndex, VideoFormat* formatList, int32_t lenght, int32_t index) {
	OEIP_CHECKINSTANCEINT;
	OEIP_CHECKDEVICEINT;
	int32_t size = device->getFormats().size();
	if (index + lenght > size) {
		logMessage(OEIP_INFO, "getFormatList length and index out of range.");
	}
	for (int32_t i = index; i < lenght; i++) {
		if (i >= size)
			break;
		formatList[i - index] = device->getFormats()[i];
	}
	return fmin(size - index, lenght);
}

int32_t getFormat(int32_t deviceIndex) {
	OEIP_CHECKINSTANCEINT;
	OEIP_CHECKDEVICEINT;
	return device->getFormat();
}

void setFormat(int32_t deviceIndex, int32_t formatIndex) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKDEVICEVOID;
	device->setFormat(formatIndex);
}

bool openDevice(int32_t deviceIndex) {
	OEIP_CHECKINSTANCEBOOL;
	OEIP_CHECKDEVICEBOOL;
	return device->openDevice();
}

void closeDevice(int32_t deviceIndex) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKDEVICEVOID;
	device->closeDevice();
}

bool bOpen(int32_t deviceIndex) {
	OEIP_CHECKINSTANCEBOOL;
	OEIP_CHECKDEVICEBOOL;
	return device->bOpen();
}

void getDeviceParametrs(int32_t deviceIndex, CamParametrs* parametrs) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKDEVICEVOID;
	memcpy(parametrs, &device->getParametrs(), sizeof(CamParametrs));
}

void setDeviceParametrs(int32_t deviceIndex, const CamParametrs* parametrs) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKDEVICEVOID;
	device->setParametrs(*parametrs);
}

void setDeviceDataAction(int32_t deviceIndex, onReviceAction onProcessData) {
	setDeviceDataHandle(deviceIndex, onProcessData);
}

void setDeviceDataHandle(int32_t deviceIndex, onReviceHandle onProcessData) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKDEVICEVOID;
	device->setDataHandle(onProcessData);
}

void setDeviceEventAction(int32_t deviceIndex, onEventAction onDeviceEvent) {
	setDeviceEventHandle(deviceIndex, onDeviceEvent);
}

void setDeviceEventHandle(int32_t deviceIndex, onEventHandle onDeviceEvent) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKDEVICEVOID;
	device->setDeviceHandle(onDeviceEvent);
}

void startAudioOutput(bool bMic, bool bLoopback, OeipAudioDesc desc, onAudioDataHandle dataHandle) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKAUDIOVOID;
	audioOutput->onDataHandle = dataHandle;
	audioOutput->start(bMic, bLoopback, desc);
}

void setAudioOutputHandle(onAudioOutputHandle outDatahandle) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKAUDIOVOID;
	audioOutput->onAudioHandle = outDatahandle;
}

void closeAudioOutput() {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKAUDIOVOID;
	audioOutput->stop();
}

int32_t initPipe(OeipGpgpuType gpgpuType) {
	OEIP_CHECKINSTANCEINT;
	return oInstance->initPipe(gpgpuType);
}

bool closePipe(int32_t pipeId)
{
	OEIP_CHECKINSTANCEBOOL;
	OEIP_CHECKPIPEBOOL;
	return oInstance->closePipe(pipeId);
}

int32_t addPiepLayer(int32_t pipeId, const char* layerName, OeipLayerType layerType, const void* paramet) {
	OEIP_CHECKINSTANCEINT;
	OEIP_CHECKPIPEINT;
	return pipe->addLayer(layerName, layerType, paramet);
}

void connectLayerName(int32_t pipeId, int32_t layerIndex, const char* forwardName, int32_t inputIndex, int32_t selfIndex) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKPIPEVOID;
	pipe->connectLayer(layerIndex, forwardName, inputIndex, selfIndex);
}

void connectLayerIndex(int32_t pipeId, int32_t layerIndex, int32_t forwardIndex, int32_t inputIndex, int32_t selfIndex) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKPIPEVOID;
	pipe->connectLayer(layerIndex, forwardIndex, inputIndex, selfIndex);
}

void setEnableLayer(int32_t pipeId, int32_t layerIndex, bool bEnable) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKPIPEVOID;
	pipe->setEnableLayer(layerIndex, bEnable);
}

void setEnableLayerList(int32_t pipeId, int32_t layerIndex, bool bEnable) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKPIPEVOID;
	pipe->setEnableLayerList(layerIndex, bEnable);
}

void setPipeInput(int32_t pipeId, int32_t layerIndex, int32_t width, int32_t height, int32_t dataType, int32_t intputIndex) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKPIPEVOID;
	pipe->setInput(layerIndex, width, height, dataType, intputIndex);
}

void updatePipeInput(int32_t pipeId, int32_t layerIndex, uint8_t* data, int32_t intputIndex) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKPIPEVOID;
	pipe->updateInput(layerIndex, data, intputIndex);
}

void setPipeDataAction(int32_t pipeId, onProcessAction onProcessData) {
	setPipeDataHandle(pipeId, onProcessData);
}

void setPipeDataHandle(int32_t pipeId, onProcessHandle onProcessData) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKPIPEVOID;
	pipe->setDataProcess(onProcessData);
}

void runPipe(int32_t pipeId) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKPIPEVOID;
	pipe->runLayers();
}

void setPipeInputGpuTex(int32_t pipeId, int32_t layerIndex, void* ctx, void* tex, int32_t inputIndex) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKPIPEVOID;
	pipe->setInputGpuTex(layerIndex, ctx, tex, inputIndex);
}

void setPipeOutputGpuTex(int32_t pipeId, int32_t layerIndex, void* ctx, void* tex, int32_t outputIndex) {
	OEIP_CHECKINSTANCEVOID;
	OEIP_CHECKPIPEVOID;
	pipe->setOutputGpuTex(layerIndex, ctx, tex, outputIndex);
}

bool updatePipeParamet(int32_t pipeId, int32_t layerIndex, const void* paramet) {
	OEIP_CHECKINSTANCEBOOL;
	OEIP_CHECKPIPEBOOL;
	return pipe->updateLayer(layerIndex, paramet);
}

//template<typename T>
//inline bool updatePipeParamet(int32_t pipeId, int32_t layerIndex, const T& t)
//{
//	return updateLayer(oInstance, pipeId, layerIndex, t);
//}



