#pragma once
#include <guiddef.h>
#include <OeipCommon.h>
#include <DirectXMath.h>
#include <winerror.h>
#include <vector>
#include <mfobjects.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <atlbase.h>
#include <d3d11.h>
#include <chrono>
#include <string>

using namespace std;

//https://msdn.microsoft.com/en-us/library/windows/desktop/ff819477(v=vs.85).aspx 写文件 
struct IMFMediaType;

struct MediaType
{
	unsigned int frameSize;
	unsigned int height;
	unsigned int width;
	unsigned int yuvMatrix;
	unsigned int videoLighting;
	unsigned int defaultStride;
	unsigned int videoChromaSiting;
	GUID formatType;
	wchar_t* formatTypeName;
	unsigned int fixedSizeSamples;
	unsigned int videoNominalRange;
	unsigned int frameRate;
	unsigned int frameRateLow;
	unsigned int pixelAspectRatio;
	unsigned int pixelAspectRatioLow;
	unsigned int allSamplesIndependent;
	unsigned int frameRateRangeMin;
	unsigned int frameRateRangeMinLow;
	unsigned int sampleSize;
	unsigned int videoPrimaries;
	unsigned int interlaceMode;
	unsigned int frameRateRangeMax;
	unsigned int frameRateRangeMaxLow;
	unsigned int bitRate;
	GUID majorType;
	wchar_t* majorTypeName;
	GUID subtype;
	wchar_t* subtypeName;
};

bool getSourceMediaList(IMFMediaSource* source, std::vector<MediaType>& mediaTypeList);

//void logMessage(int level, const char* message);

//void setLogEvent(logEventHandle logEvent);

//void setEventHandle(onEventHandle eventHandle);

inline bool checkHR(HRESULT hr, const char* message)
{
	if (FAILED(hr))	{
		logMessage(OEIP_ERROR, message);
	}
	return SUCCEEDED(hr);
}

//void getTimeDuration(long long & timeStamp);
