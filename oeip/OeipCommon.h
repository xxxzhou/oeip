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

OEIPDLL_EXPORT std::string getProgramPath();

template <class T>
void safeRelease(T*& ppT) {
	if (ppT != nullptr) {
		(ppT)->Release();
		ppT = nullptr;
	}
}

template <class T>
void safeReleaseAll(T*& ppT) {
	if (ppT != nullptr) {
		long e = (ppT)->Release();
		while (e) {
			e = (ppT)->Release();
		}
		ppT = nullptr;
	}
}

template <class T>
void safeDelete(T*& ppT) {
	if (ppT != nullptr) {
		delete ppT;
		ppT = nullptr;
	}
}

template<typename  T>
void clearList(std::vector<T*> list) {
	for (T* t : list) {
		safeDelete(t);
	}
	list.clear();
}

template<int32_t index, typename... Types>
struct LayerParamet;

template<int32_t index, typename First, typename... Types>
struct LayerParamet<index, First, Types...>
{
	using ParametType = typename LayerParamet<index - 1, Types...>::ParametType;
};

template<typename T, typename... Types>
struct LayerParamet<0, T, Types...>
{
	using ParametType = T;
};

template<typename T, typename... Types>
struct LayerParamet<-1, T, Types...>
{
	using ParametType = int32_t;
};

//前向声明
template < typename T, typename... List >
struct IndexOf;

//一般形式
template < typename T, typename Head, typename... Rest >
struct IndexOf<T, Head, Rest...>
{
	static const int32_t value = IndexOf<T, Rest...>::value + 1;
};

//递归终止一 查找到相等(T=T向前++)
template < typename T, typename... Rest >
struct IndexOf<T, T, Rest...>
{
	static const int32_t value = 0;
};

//递归终止二 没找到
template < typename T >
struct IndexOf<T>
{
	static const int32_t value = -1;
};