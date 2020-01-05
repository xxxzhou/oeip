#pragma once
#include "MediaStruct.h"
#include <vector>
#include "ReaderCallback.h"
#include <functional>
#include "MFCaptureDevice.h"
#include "VideoDevice.h"
using namespace std;

struct IMFActivate;
struct IMFMediaSource;
struct IMFSourceReader;

class VideoCaptureDevice : public MFCaptureDevice, public VideoDevice
{
public:
	VideoCaptureDevice();
	virtual ~VideoCaptureDevice();
public:
	virtual bool init(IMFActivate* pActivate, unsigned int num) override;	
	virtual bool openDevice() override;
	virtual bool closeDevice() override;
	virtual bool bOpen() override;
	const OeipVideoType getVideoType(const wchar_t* videoName);
	void addVideoType(const MediaType& mediaType, int index);
	virtual CamParametrs getParametrs() override;
	virtual void setParametrs(CamParametrs parametrs) override;
	virtual wchar_t* getDeviceName()override { return deviceName; };
	virtual wchar_t* getDeviceID() override { return deviceID; };
	virtual void setDeviceHandle(onEventHandle eventHandle) override;
private:
	bool resetFormats();
private:
	CamParametrs preParametrs = {};
	CComPtr<IMFMediaTypeHandler> handle = nullptr;
};

