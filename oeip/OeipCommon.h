#pragma once
#include "Oeip.h"
#include <string>
#include <vector>

#define OEIP_EMPTYSTR ""

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=nullptr; } } 
#endif
#ifndef SAFE_DELETE
#define SAFE_DELETE(p)      { if (p) { delete (p); (p)=nullptr; } } 
#endif

struct UInt3
{
	uint32_t X = 1;
	uint32_t Y = 1;
	uint32_t Z = 1;

	UInt3(uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) {
		X = x;
		Y = y;
		Z = z;
	}
};

OEIPDLL_EXPORT void logMessage(int level, const char* message);

OEIPDLL_EXPORT void setLogEvent(logEventHandle logEvent);

OEIPDLL_EXPORT std::wstring string2wstring(std::string str);

OEIPDLL_EXPORT std::string wstring2string(std::wstring wstr);

OEIPDLL_EXPORT bool readResouce(std::string modelName, int32_t rcId, std::string rctype, std::string& resouce, uint32_t& dataType);

OEIPDLL_EXPORT void splitString(const std::string& str, std::vector<std::string>& strarray, const std::string& split);

OEIPDLL_EXPORT void copywcharstr(wchar_t* dest, const wchar_t* source, int maxlength);

OEIPDLL_EXPORT void copycharstr(char* dest, const char* source, int maxlength);

OEIPDLL_EXPORT bool loadFile(std::wstring path, std::vector<uint8_t>& data, int length);

OEIPDLL_EXPORT bool saveFile(std::wstring path, void* data, int length);

OEIPDLL_EXPORT int64_t getNowTimestamp();

OEIPDLL_EXPORT int32_t getDataType(OeipVideoType videoType);

OEIPDLL_EXPORT std::string getLayerName(OeipLayerType layerType);

OEIPDLL_EXPORT uint32_t divUp(int32_t x, int32_t y);

template <class T> void safeRelease(T*& ppT)
{
	if (ppT != nullptr) {
		(ppT)->Release();
		ppT = nullptr;
	}
}

template <class T> void safeReleaseAll(T*& ppT)
{
	if (ppT != nullptr) {
		long e = (ppT)->Release();
		while (e) {
			e = (ppT)->Release();
		}
		ppT = nullptr;
	}
}

template <class T> void safeDelete(T*& ppT)
{
	if (ppT != nullptr) {
		delete ppT;
		ppT = nullptr;
	}
}

template<typename  T>
void clearList(std::vector<T*> list)
{
	for (T* t : list) {
		safeDelete(t);
	}
	list.clear();
}