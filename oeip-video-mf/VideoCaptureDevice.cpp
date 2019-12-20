#include "VideoCaptureDevice.h"
#include <mfobjects.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <OeipCommon.h>
#include "FormatReader.h"
#include <Strmif.h>
#include <functional>

using namespace std;
using namespace std::placeholders;

VideoCaptureDevice::VideoCaptureDevice()
{
	type = VideoDeviceType::OEIP_MF;
	streamIndex = MF_SOURCE_READER_FIRST_VIDEO_STREAM;
	//把数据交给VideoProcess处理
	auto processData = std::bind(&VideoCaptureDevice::onDataHandle, this, _1, _2);
	readerCallback->setBufferRevice(processData);
}

VideoCaptureDevice::~VideoCaptureDevice()
{
	readerCallback->setBufferRevice(nullptr);
}

bool VideoCaptureDevice::init(IMFActivate* pActivate, unsigned int num)
{
	bool bResult = MFCaptureDevice::init(pActivate, num);
	if (bResult) {
		resetFormats();
		bResult = videoFormats.size() > 0;
		if (bResult) {
			setFormat(0);
		}
	}
	return bResult;
}

void VideoCaptureDevice::setDeviceHandle(onEventHandle eventHandle)
{
	VideoDevice::setDeviceHandle(eventHandle);
	readerCallback->setDeviceEvent(onDeviceEvent);
}

bool VideoCaptureDevice::resetFormats()
{
	bool bExcute = false;
	videoFormats.clear();

	CComPtr<IMFPresentationDescriptor> pd = nullptr;
	CComPtr<IMFStreamDescriptor> sd = nullptr;
	BOOL bSelected = false;
	unsigned long types = 0;
	auto hr = source->CreatePresentationDescriptor(&pd);
	if (FAILED(hr))
		return false;
	hr = pd->GetStreamDescriptorByIndex(0, &bSelected, &sd);
	if (FAILED(hr))
		return false;
	handle.Release();
	hr = sd->GetMediaTypeHandler(&handle);
	if (FAILED(hr))
		return false;
	hr = handle->GetMediaTypeCount(&types);
	if (FAILED(hr))
		return false;
	for (int i = 0; i < types; i++) {
		CComPtr<IMFMediaType> type = nullptr;
		hr = handle->GetMediaTypeByIndex(i, &type);
		if (FAILED(hr))
			return false;
		MediaType mediaType = {};
		if (FormatReader::Read(type, mediaType)) {
			addVideoType(mediaType, i);
		}
	}
	bExcute = true;
	return bExcute;
}

bool VideoCaptureDevice::openDevice()
{
	//先关闭当前正在读的数据流
	bool result = MFCaptureDevice::Open();
	if (!result) {
		logMessage(OEIP_ERROR, "create source reader fail.");
		return false;
	}
	//先关闭在读的流，不然设置不了format
	readerCallback->setPlay(false);
	CComPtr<IMFMediaType> mtype = nullptr;
	CComPtr<IMFMediaType> pType = nullptr;
	//使用新的格式
	auto hr = handle->GetMediaTypeByIndex(videoFormat.index, &mtype);
	if (SUCCEEDED(hr)) {
		//source 应用新的格式		
		hr = handle->SetCurrentMediaType(mtype);
		//source reader得到当前的播放格式，如果是压缩的，输出转码
		hr = sourceReader->GetNativeMediaType(streamIndex, selectIndex, &pType);
		if (FAILED(hr)) {
			logMessage(OEIP_ERROR, "The current index cannot be correctly formatted.");
			return false;
		}
		//MediaType MT = FormatReader::Read(pType);
		GUID majorType = { 0 };
		GUID subtype = { 0 };
		hr = pType->GetGUID(MF_MT_MAJOR_TYPE, &majorType);
		hr = pType->GetGUID(MF_MT_SUBTYPE, &subtype);
		if (majorType != MFMediaType_Video) {
			logMessage(OEIP_ERROR, "createSourceReader current index not video format.");
			return false;
		}
		if (subtype == MFVideoFormat_MJPG) {
			subtype = MFVideoFormat_YUY2;
			hr = pType->SetGUID(MF_MT_SUBTYPE, subtype);
		}
		hr = sourceReader->SetCurrentMediaType(streamIndex, nullptr, pType);
		if (FAILED(hr)) {
			logMessage(OEIP_ERROR, "The encoder cannot set this format.");
			return false;
		}
		//开始读取数据
		readerCallback->setPlay(true);
	}
	else {
		logMessage(OEIP_ERROR, "The current index was not applied correctly.");
	}
	return SUCCEEDED(hr);
}

bool VideoCaptureDevice::closeDevice()
{
	bool result = true;
	if (bOpen()) {
		result = readerCallback->setPlay(false);
		if (result)
			readerCallback->onDeviceHandle(OEIP_DeviceStop, 0);
	}
	MFCaptureDevice::Close();
	return result;
}

bool VideoCaptureDevice::bOpen()
{
	return readerCallback->IsOpen();
}

const OeipVideoType VideoCaptureDevice::getVideoType(const wchar_t* videoName)
{
	static vector<wstring> videoTypeList = { L"MFVideoFormat_NV12" ,L"MFVideoFormat_YUY2" ,L"MFVideoFormat_YVYU",L"MFVideoFormat_UYVY" ,L"MFVideoFormat_MJPG" ,L"MFVideoFormat_RGB24" ,L"MFVideoFormat_ARGB32" };
	int vindex = -1;
	for (int i = 0; i < videoTypeList.size(); i++) {
		if (wcscmp(videoName, videoTypeList[i].c_str()) == 0) {
			vindex = i;
			break;
		}
	}
	OeipVideoType videoType = (OeipVideoType)(vindex + 1);
	return videoType;
}

void VideoCaptureDevice::addVideoType(const MediaType& mediaType, int index)
{
	VideoFormat videoFormat;
	videoFormat.index = index;
	videoFormat.width = mediaType.width;
	videoFormat.height = mediaType.height;
	videoFormat.fps = mediaType.frameRate;
	videoFormat.videoType = getVideoType(mediaType.subtypeName);
	if (videoFormat.videoType != OEIP_VIDEO_OTHER)
		videoFormats.push_back(videoFormat);
}

CamParametrs VideoCaptureDevice::getParametrs()
{
	CamParametrs out = {};
	unsigned int shift = sizeof(Parametr);
	Parametr* pParametr = (Parametr*)(&out);

	CComPtr<IAMVideoProcAmp> pProcAmp = NULL;
	HRESULT hr = source->QueryInterface(IID_PPV_ARGS(&pProcAmp));
	if (SUCCEEDED(hr)) {
		for (unsigned int i = 0; i < 10; i++) {
			Parametr temp;
			hr = pProcAmp->GetRange(VideoProcAmp_Brightness + i, &temp.Min, &temp.Max, &temp.Step, &temp.Default, &temp.Flag);
			if (SUCCEEDED(hr)) {
				long currentValue = temp.Default;
				long flag = temp.Flag;
				hr = pProcAmp->Get(VideoProcAmp_Brightness + i, &currentValue, &flag);
				temp.CurrentValue = currentValue;
				temp.Flag = flag;
				pParametr[i] = temp;
			}
		}
	}
	CComPtr<IAMCameraControl> pProcControl = NULL;
	hr = source->QueryInterface(IID_PPV_ARGS(&pProcControl));
	if (SUCCEEDED(hr)) {
		for (unsigned int i = 0; i < 7; i++) {
			Parametr temp;
			hr = pProcControl->GetRange(CameraControl_Pan + i, &temp.Min, &temp.Max, &temp.Step, &temp.Default, &temp.Flag);
			if (SUCCEEDED(hr)) {
				long currentValue = temp.Default;
				long flag = temp.Flag;
				hr = pProcControl->Get(CameraControl_Pan + i, &currentValue, &flag);
				temp.CurrentValue = currentValue;
				temp.Flag = flag;
				pParametr[10 + i] = temp;
			}
		}
	}
	return out;
}

void VideoCaptureDevice::setParametrs(CamParametrs parametrs)
{
	unsigned int shift = sizeof(Parametr);
	Parametr* pParametr = (Parametr*)(&parametrs);
	CamParametrs curPar = getParametrs();
	Parametr* pPrevParametr = (Parametr*)(&curPar);

	CComPtr<IAMVideoProcAmp> pProcAmp = NULL;
	HRESULT hr = source->QueryInterface(IID_PPV_ARGS(&pProcAmp));
	if (SUCCEEDED(hr)) {
		for (unsigned int i = 0; i < 10; i++) {
			if (pParametr[i].Min == pParametr[i].Max)
				continue;
			if (pPrevParametr[i].CurrentValue != pParametr[i].CurrentValue || pPrevParametr[i].Flag != pParametr[i].Flag) {
				hr = pProcAmp->Set(VideoProcAmp_Brightness + i, pParametr[i].CurrentValue, pParametr[i].Flag);
			}
		}
	}
	CComPtr<IAMCameraControl> pProcControl = NULL;
	hr = source->QueryInterface(IID_PPV_ARGS(&pProcControl));
	if (SUCCEEDED(hr)) {
		for (unsigned int i = 0; i < 7; i++) {
			if (pParametr[10 + i].Min == pParametr[10 + i].Max)
				continue;
			if (pParametr[10 + i].CurrentValue <pParametr[10 + i].Min || pParametr[10 + i].CurrentValue >pParametr[10 + i].Max)
				continue;
			if (pPrevParametr[10 + i].CurrentValue != pParametr[10 + i].CurrentValue || pPrevParametr[10 + i].Flag != pParametr[10 + i].Flag)
				hr = pProcControl->Set(CameraControl_Pan + i, pParametr[10 + i].CurrentValue, pParametr[10 + i].Flag);
		}
	}
	preParametrs = parametrs;
}
