#include "OeipExport.h"
#include "OeipManager.h"
#include "PluginManager.h"
#include "VideoManager.h"
#include <math.h>

static OeipManager* oInstance = nullptr;
static bool bInit = false;

void setLogAction(logEventAction logHandle)
{
	setLogEvent(logHandle);
}

void setLogHandle(logEventHandle logHandle)
{
	setLogEvent(logHandle);
}

void initOeip()
{
	if (!bInit) {
		oInstance = OeipManager::getInstance();
		bInit = true;
	}
}

void shutdownOeip()
{
	if (bInit) {
		OeipManager::shutdown();
		bInit = false;
	}
}

OeipYUVFMT getVideoYUV(OeipVideoType videoType)
{
	OeipYUVFMT fmt = OEIP_YUVFMT_OTHER;
	switch (videoType)
	{
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

int32_t initPipe(OeipGpgpuType gpgpuType)
{
	return oInstance->initPipe(gpgpuType);
}

int getDeviceCount()
{
	return oInstance->getVideoList().size();
}

void getDeviceList(OeipDeviceInfo* deviceList, int32_t lenght, int32_t index)
{
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
		copywcharstr(device.deviceID, capture->getDeviceID(), sizeof(device.deviceID));
		device.id = i;
	}
}

int32_t getFormatCount(int32_t deviceIndex)
{
	auto device = oInstance->getVideoIndex(deviceIndex);
	if (!device)
		return 0;
	return device->getFormats().size();
}

int32_t getFormatList(int32_t deviceIndex, VideoFormat* formatList, int32_t lenght, int32_t index)
{
	auto device = oInstance->getVideoIndex(deviceIndex);
	if (!device)
		return 0;
	int32_t size = device->getFormats().size();
	if (index + lenght > size) {
		logMessage(OEIP_INFO, "getFormatList length and index out of range.");
	}
	for (int32_t i = index; i < lenght; i++) {
		if (i >= size)
			break;
		formatList[i - index] = device->getFormats()[i];
	}
}

int32_t getFormat(int32_t deviceIndex)
{
	auto device = oInstance->getVideoIndex(deviceIndex);
	if (!device)
		return -1;
	return device->getFormat();
}

void setFormat(int32_t deviceIndex, int32_t formatIndex)
{
	auto device = oInstance->getVideoIndex(deviceIndex);
	if (!device)
		return;
	device->setFormat(formatIndex);
}

bool openDevice(int32_t deviceIndex)
{
	auto device = oInstance->getVideoIndex(deviceIndex);
	if (!device)
		return false;
	return device->openDevice();
}

void setDeviceDataAction(int32_t deviceIndex, onReviceAction onProcessData)
{
	setDeviceDataHandle(deviceIndex, onProcessData);
}

void setDeviceDataHandle(int32_t deviceIndex, onReviceHandle onProcessData)
{
	auto device = oInstance->getVideoIndex(deviceIndex);
	if (!device)
		return;
	device->setDataHandle(onProcessData);
}

void setDeviceEventAction(int32_t deviceIndex, onEventAction onDeviceEvent)
{
	setDeviceEventHandle(deviceIndex, onDeviceEvent);
}

void setDeviceEventHandle(int32_t deviceIndex, onEventHandle onDeviceEvent)
{
	auto device = oInstance->getVideoIndex(deviceIndex);
	if (!device)
		return;
	device->setDeviceHandle(onDeviceEvent);
}

void setPipeInput(int32_t pipeId, int32_t layerIndex, int32_t width, int32_t height, int32_t dataType, int32_t intputIndex)
{
	auto pipe = oInstance->getPipe(pipeId);
	if (!pipe)
		return;
	pipe->setInput(layerIndex, width, height, dataType, intputIndex);
}

void updatePipeInput(int32_t pipeId, int32_t layerIndex, uint8_t* data, int32_t intputIndex)
{
	auto pipe = oInstance->getPipe(pipeId);
	if (!pipe)
		return;
	pipe->updateInput(layerIndex, data, intputIndex);
}

void setPipeDataAction(int32_t pipeId, onProcessAction onProcessData)
{
	setPipeDataHandle(pipeId, onProcessData);
}

void setPipeDataHandle(int32_t pipeId, onProcessHandle onProcessData)
{
	auto pipe = oInstance->getPipe(pipeId);
	if (!pipe)
		return;
	pipe->setDataProcess(onProcessData);
}

void runPipe(int32_t pipeId)
{
	auto pipe = oInstance->getPipe(pipeId);
	if (!pipe)
		return;
	pipe->runLayers();
}

void setPipeInputGpuTex(int32_t pipeId, int32_t layerIndex, void* ctx, void* tex, int32_t inputIndex)
{
}

void setPipeOutputGpuTex(int32_t pipeId, int32_t layerIndex, void* ctx, void* tex, int32_t outputIndex)
{
	auto pipe = oInstance->getPipe(pipeId);
	if (!pipe)
		return;
	pipe->setOutputGpuTex(layerIndex, ctx, tex, outputIndex);
}

template<typename T>
inline bool updatePipeParamet(int32_t pipeId, int32_t layerIndex, const T& t)
{
	return updateLayer(oInstance, pipeId, layerIndex, t);
}



