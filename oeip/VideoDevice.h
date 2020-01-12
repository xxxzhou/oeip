#pragma once
#include "OeipCommon.h"
#include <vector>

class OEIPDLL_EXPORT VideoDevice
{
public:
	VideoDevice();
	virtual ~VideoDevice();
protected:
	onEventHandle onDeviceEvent = nullptr;
	onReviceHandle onProcessData = nullptr;
	VideoFormat videoFormat = {};
	int32_t selectIndex = 0;
	std::vector<VideoFormat> videoFormats;
	int32_t type = -1;
protected:
	void onDeviceHandle(OeipDeviceEventType eventType, int32_t code);
	void onDataHandle(unsigned long lenght, uint8_t* data);
public:
	const std::vector<VideoFormat>& getFormats() { return videoFormats; };
	int32_t getFormat() {
		return selectIndex;
	};
	int32_t findFormatIndex(int32_t width, int32_t height, int32_t fps = 30);
public:
	virtual bool setFormat(uint32_t index);
	virtual wchar_t* getDeviceName() { return nullptr; };
	virtual wchar_t* getDeviceID() { return nullptr; };

	virtual void getFormat(uint32_t index, VideoFormat& format);
	virtual bool openDevice() { return false; };
	virtual bool closeDevice() { return false; };
	virtual void setDeviceHandle(onEventHandle eventHandle) { onDeviceEvent = eventHandle; };
	virtual void setDataHandle(onReviceHandle eventHandle) { onProcessData = eventHandle; };
	virtual bool bOpen() { return false; }
	virtual CamParametrs getParametrs() { return CamParametrs{}; };
	virtual void setParametrs(CamParametrs parametrs) {};
	virtual int32_t getVideoType() { return type; };
	bool saveCameraParameters(std::wstring path);
	bool loadCameraParameters(std::wstring path);
};

