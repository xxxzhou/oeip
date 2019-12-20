#include "MediaStruct.h"
#include <Mfapi.h>
#include "FormatReader.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <string>

bool getSourceMediaList(IMFMediaSource* source, std::vector<MediaType>& mediaTypeList)
{
	bool bExcute = false;
	CComPtr<IMFPresentationDescriptor> pd = nullptr;
	CComPtr<IMFStreamDescriptor> sd = nullptr;
	CComPtr<IMFMediaTypeHandler> handle = nullptr;

	BOOL bSelected = false;
	unsigned long types = 0;
	auto hr = source->CreatePresentationDescriptor(&pd);
	if (FAILED(hr)) {
		return false;
	}
	hr = pd->GetStreamDescriptorByIndex(0, &bSelected, &sd);
	if (FAILED(hr)) {
		return false;
	}
	hr = sd->GetMediaTypeHandler(&handle);
	if (FAILED(hr)) {
		return false;
	}
	hr = handle->GetMediaTypeCount(&types);
	if (FAILED(hr)) {
		return false;
	}
	for (int i = 0; i < types; i++) {
		CComPtr<IMFMediaType> type = nullptr;
		hr = handle->GetMediaTypeByIndex(i, &type);
		if (FAILED(hr)) {
			return false;
		}
		MediaType mediaType = {};
		if (FormatReader::Read(type, mediaType)) {
			mediaTypeList.push_back(mediaType);
		}		
	}
	bExcute = true;
	return bExcute;
}
