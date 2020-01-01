#include "MFVideoManager.h"
#include "MFCaptureDevice.h"
#include "VideoCaptureDevice.h"
#include "AudioRecordWin.h"

MFVideoManager::MFVideoManager() {
	CComPtr<IMFAttributes> pAttributes = nullptr;
	auto hr = MFCreateAttributes(&pAttributes, 1);
	if (SUCCEEDED(hr)) {
		hr = pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	}
	if (SUCCEEDED(hr)) {
		IMFActivate** ppDevices = nullptr;
		UINT32 count = -1;
		hr = MFEnumDeviceSources(pAttributes, &ppDevices, &count);
		if (SUCCEEDED(hr)) {
			if (count > 0) {
				for (UINT32 i = 0; i < count; i++) {
					//过滤掉RealSense2					
					if (MFCaptureDevice::Init(ppDevices[i])) {
						VideoCaptureDevice* vc = new VideoCaptureDevice();
						if (vc->init(ppDevices[i], i)) {
							videoList.push_back(vc);
						}
						else {
							delete vc;
						}
					}
					ppDevices[i]->Release();
				}
			}
			CoTaskMemFree(ppDevices);
		}
	}
}


MFVideoManager::~MFVideoManager() {
	HRESULT hr = MFShutdown();
	CoUninitialize();
}

std::vector<VideoDevice*> MFVideoManager::getDeviceList() {
	return videoList;
}

bool bCanLoad() {
	auto hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
	hr &= MFStartup(MF_VERSION);
	hr &= CoInitialize(NULL);
	if (!SUCCEEDED(hr)) {
		logMessage(OEIP_ERROR, "Media foundation not initialized correctly.");
		return false;
	}
	else {
		logMessage(OEIP_INFO, "Media foundation correctly initialized.");
		return true;
	}
}

void registerFactory() {
	registerFactory(new MFVideoManagerFactory(), VideoDeviceType::OEIP_MF, "video mf");
	registerFactory(new AudioRecordWinFactory(), 0, "audio record mf");
}
