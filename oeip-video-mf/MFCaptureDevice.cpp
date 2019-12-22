#include "MFCaptureDevice.h"
#include <functional>

using namespace std;
using namespace std::placeholders;

MFCaptureDevice::MFCaptureDevice() {
	readerCallback = new ReaderCallback();
	auto hr = MFCreateAttributes(&pAttributes, 2);
	//开户格式转换，如mgjp 转yuv2
	if (SUCCEEDED(hr)) {
		hr = pAttributes->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, TRUE);
	}
	// 异步采集方案 
	if (SUCCEEDED(hr)) {
		hr = pAttributes->SetUnknown(MF_SOURCE_READER_ASYNC_CALLBACK, readerCallback);
	}
}

MFCaptureDevice::~MFCaptureDevice() {
	if (source != nullptr) {
		source->Shutdown();
		source = nullptr;
		//safeRelease(source);
	}
}

bool MFCaptureDevice::init(IMFActivate* pActivate, unsigned int num) {
	activate = pActivate;
	id = num;
	auto hr = activate->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &deviceName, nullptr);
	hr = activate->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &deviceID, nullptr);
	//可能会造成设备卡住
	hr = activate->ActivateObject(__uuidof(IMFMediaSource), (void**)&source);
	//很多采集设备可以进这步，但是MF读不了，不需要给出错误信息
	if (FAILED(hr)) {
		logMessage(OEIP_INFO, "create media soure fail.");
		return false;
	}
#if REALSENSE 
	wstring deviceNameStr = deviceName;
	//过滤掉RealSense2
	int index = deviceNameStr.find(L"Intel(R) RealSense(TM)");
	if (index >= 0) {
		return false;
	}
#endif
	hr = MFCreateSourceReaderFromMediaSource(source, pAttributes, &sourceReader);
	readerCallback->setSourceReader(sourceReader, streamIndex);
	//safeRelease(pAttributes);
	if (FAILED(hr)) {
		logMessage(OEIP_ERROR, "create soure reader fail.");
		return false;
	}
	bOpen = true;
	return true;
}

bool MFCaptureDevice::Init(IMFActivate* pActivate) {
	wchar_t* name = nullptr;
	auto hr = pActivate->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &name, nullptr);
	//可能会造成设备卡住
	CComPtr<IMFMediaSource> msource = nullptr;
	hr = pActivate->ActivateObject(__uuidof(IMFMediaSource), (void**)&msource);
	//很多采集设备可以进这步，但是MF读不了，不需要给出错误信息
	if (FAILED(hr)) {
		logMessage(OEIP_INFO, "create media soure fail.");
		return false;
	}
	//#if REALSENSE 
	wstring deviceNameStr = name;
	//过滤掉RealSense2
	int index = deviceNameStr.find(L"Intel(R) RealSense(TM)");
	if (index >= 0) {
		return false;
	}
	//#endif
	return true;
}

bool MFCaptureDevice::Open() {
	//可能会造成设备卡住
	if (!bOpen) {
		auto hr = activate->ActivateObject(__uuidof(IMFMediaSource), (void**)&source);
		hr = MFCreateSourceReaderFromMediaSource(source, pAttributes, &sourceReader);
		readerCallback->setSourceReader(sourceReader, streamIndex);
		//很多采集设备可以进这步，但是MF读不了，不需要给出错误信息
		if (FAILED(hr)) {
			logMessage(OEIP_INFO, "create media soure fail.");
			return false;
		}
		bOpen = true;
	}
	return true;
}

bool MFCaptureDevice::Close() {
	long hr = 0;
	if (bOpen) {
		sourceReader = nullptr;
		//hr = source->Shutdown();
		source = nullptr;
		hr = activate->ShutdownObject();
		bOpen = false;
	}
	return SUCCEEDED(hr);
}

