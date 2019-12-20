#pragma once

#include "MediaStruct.h"
#include "ReaderCallback.h"

class MFCaptureDevice
{
public:
	MFCaptureDevice();
	virtual ~MFCaptureDevice();
	virtual bool init(IMFActivate *pActivate, unsigned int num);
	static bool Init(IMFActivate *pActivate);

	bool Open();
	bool Close();
protected:
	CComPtr<IMFAttributes> pAttributes = nullptr;
	CComPtr<IMFActivate> activate = nullptr;
	CComPtr<IMFMediaSource> source = nullptr;
	CComPtr<IMFSourceReader> sourceReader = nullptr;
	//readerCallback用于MF读取数据 
	CComPtr<ReaderCallback> readerCallback = nullptr;
public:
	unsigned int id = 0;
	wchar_t* deviceName = nullptr;
	wchar_t* deviceID = nullptr;
	bool bOpen = false;
protected:
	unsigned long streamIndex = -1;

};

