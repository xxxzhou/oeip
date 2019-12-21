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

int getDeviceCount()
{
	OEIP_CHECKINSTANCEINT
	return oInstance->getVideoList().size();
}

void getDeviceList(OeipDeviceInfo* deviceList, int32_t lenght, int32_t index)
{
	OEIP_CHECKINSTANCEVOID
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
	OEIP_CHECKINSTANCEINT
	OEIP_CHECKDEVICEINT
	return device->getFormats().size();
}

int32_t getFormatList(int32_t deviceIndex, VideoFormat* formatList, int32_t lenght, int32_t index)
{
	OEIP_CHECKINSTANCEINT
	OEIP_CHECKDEVICEINT
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
	OEIP_CHECKINSTANCEINT
	OEIP_CHECKDEVICEINT
	return device->getFormat();
}

void setFormat(int32_t deviceIndex, int32_t formatIndex)
{
	OEIP_CHECKINSTANCEVOID
	OEIP_CHECKDEVICEVOID
	device->setFormat(formatIndex);
}

bool openDevice(int32_t deviceIndex)
{
	OEIP_CHECKINSTANCEBOOL
	OEIP_CHECKDEVICEBOOL
	return device->openDevice();
}

void setDeviceDataAction(int32_t deviceIndex, onReviceAction onProcessData)
{
	setDeviceDataHandle(deviceIndex, onProcessData);
}

void setDeviceDataHandle(int32_t deviceIndex, onReviceHandle onProcessData)
{
	OEIP_CHECKINSTANCEVOID
	OEIP_CHECKDEVICEVOID
	device->setDataHandle(onProcessData);
}

void setDeviceEventAction(int32_t deviceIndex, onEventAction onDeviceEvent)
{
	setDeviceEventHandle(deviceIndex, onDeviceEvent);
}

void setDeviceEventHandle(int32_t deviceIndex, onEventHandle onDeviceEvent)
{
	OEIP_CHECKINSTANCEVOID
	OEIP_CHECKDEVICEVOID
	device->setDeviceHandle(onDeviceEvent);
}

int32_t initPipe(OeipGpgpuType gpgpuType)
{
	OEIP_CHECKINSTANCEINT
	return oInstance->initPipe(gpgpuType);
}

int32_t addPiepLayer(int32_t pipeId, const char* layerName, OeipLayerType layerType, const void* paramet)
{
	OEIP_CHECKINSTANCEINT
	OEIP_CHECKPIPEINT
	pipe->addLayer(layerName, layerType, paramet);
}

void connectLayer(int32_t pipeId, int32_t layerIndex, const char * forwardName, int32_t inputIndex, int32_t selfIndex)
{
	OEIP_CHECKINSTANCEVOID
	OEIP_CHECKPIPEVOID
	pipe->connectLayer(layerIndex, forwardName, inputIndex, selfIndex);
}

void setEnableLayer(int32_t pipeId, int32_t layerIndex, bool bEnable)
{
	OEIP_CHECKINSTANCEVOID
	OEIP_CHECKPIPEVOID
	pipe->setEnableLayer(layerIndex, bEnable);
}

void setEnableLayerList(int32_t pipeId, int32_t layerIndex, bool bEnable)
{
	OEIP_CHECKINSTANCEVOID
	OEIP_CHECKPIPEVOID
	pipe->setEnableLayerList(layerIndex, bEnable);
}

void setPipeInput(int32_t pipeId, int32_t layerIndex, int32_t width, int32_t height, int32_t dataType, int32_t intputIndex)
{
	OEIP_CHECKINSTANCEVOID
	OEIP_CHECKPIPEVOID
	pipe->setInput(layerIndex, width, height, dataType, intputIndex);
}

void updatePipeInput(int32_t pipeId, int32_t layerIndex, uint8_t* data, int32_t intputIndex)
{
	OEIP_CHECKINSTANCEVOID
	OEIP_CHECKPIPEVOID
	pipe->updateInput(layerIndex, data, intputIndex);
}

void setPipeDataAction(int32_t pipeId, onProcessAction onProcessData)
{
	setPipeDataHandle(pipeId, onProcessData);
}

void setPipeDataHandle(int32_t pipeId, onProcessHandle onProcessData)
{
	OEIP_CHECKINSTANCEVOID
	OEIP_CHECKPIPEVOID
	pipe->setDataProcess(onProcessData);
}

void runPipe(int32_t pipeId)
{
	OEIP_CHECKINSTANCEVOID
	OEIP_CHECKPIPEVOID
	pipe->runLayers();
}

void setPipeInputGpuTex(int32_t pipeId, int32_t layerIndex, void* ctx, void* tex, int32_t inputIndex)
{
	OEIP_CHECKINSTANCEVOID
	OEIP_CHECKPIPEVOID
}

void setPipeOutputGpuTex(int32_t pipeId, int32_t layerIndex, void* ctx, void* tex, int32_t outputIndex)
{
	OEIP_CHECKINSTANCEVOID
	OEIP_CHECKPIPEVOID
	pipe->setOutputGpuTex(layerIndex, ctx, tex, outputIndex);
}

bool updatePipeParamet(int32_t pipeId, int32_t layerIndex, const void* paramet)
{
	OEIP_CHECKINSTANCEBOOL
	OEIP_CHECKPIPEBOOL
	return pipe->updateLayer(layerIndex, paramet);
}

//template<typename T>
//inline bool updatePipeParamet(int32_t pipeId, int32_t layerIndex, const T& t)
//{
//	return updateLayer(oInstance, pipeId, layerIndex, t);
//}



