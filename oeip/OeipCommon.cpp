#include "OeipCommon.h"
#include <atlcomcli.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>

static logEventHandle logHandle = nullptr;

void logMessage(int level, const char* message)
{
	if (logHandle != nullptr)
	{
		//GBK->UTF8 后面全改为带bom的utf8
		//std::string str = message;
		//const char* GBK_LOCALE_NAME = ".936";
		//wstring_convert<codecvt_byname<wchar_t, char, mbstate_t>> cv1(new codecvt_byname<wchar_t, char, mbstate_t>(GBK_LOCALE_NAME));
		//wstring tmp_wstr = cv1.from_bytes(str);
		//wstring_convert<codecvt_utf8<wchar_t>> cv2;
		//string utf8_str = cv2.to_bytes(tmp_wstr);

		//logHandle(level, utf8_str._Myptr());
		//中文 ExecutionEngineException: String conversion error: Illegal byte sequence encounted in the input.
		logHandle(level, message);
	}
	else if (message)
	{
		auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		struct tm t;   //tm结构指针
		localtime_s(&t, &now);   //获取当地日期和时间
		//用std::cout会导致UE4烘陪失败,记录下
		std::wstring wmessage = string2wstring(message);
		std::wcout << std::put_time(&t, L"%Y-%m-%d %X") << L" Level: " << level << L" " << wmessage << std::endl;
	}
}

void setLogEvent(logEventHandle logEvent)
{
	logHandle = logEvent;
}

std::wstring string2wstring(std::string str)
{
	std::wstring result;
	//获取缓冲区大小，并申请空间，缓冲区大小按字符计算  
	int len = MultiByteToWideChar(CP_ACP, 0, str.c_str(), str.size(), NULL, 0);
	TCHAR* buffer = new TCHAR[len + 1];
	//多字节编码转换成宽字节编码  
	MultiByteToWideChar(CP_ACP, 0, str.c_str(), str.size(), buffer, len);
	buffer[len] = '\0';             //添加字符串结尾  
	//删除缓冲区并返回值  
	result.append(buffer);
	delete[] buffer;
	return result;
}

//将wstring转换成string  
std::string wstring2string(std::wstring wstr)
{
	std::string result;
	//获取缓冲区大小，并申请空间，缓冲区大小事按字节计算的  
	int len = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), wstr.size(), NULL, 0, NULL, NULL);
	char* buffer = new char[len + 1];
	//宽字节编码转换成多字节编码  
	WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), wstr.size(), buffer, len, NULL, NULL);
	buffer[len] = '\0';
	//删除缓冲区并返回值  
	result.append(buffer);
	delete[] buffer;
	return result;
}

bool readResouce(std::string modelName, int32_t rcId, std::string rctype, std::string& resouce, uint32_t& dataType)
{
	HMODULE hinstance = ::GetModuleHandleA(modelName.c_str());
	HRSRC hRsrc = FindResourceA(hinstance, MAKEINTRESOURCEA(rcId), rctype.c_str());

	if (nullptr == hRsrc)
		return false;
	//获取资源的大小
	DWORD dwSize = SizeofResource(hinstance, hRsrc);
	if (0 == dwSize)
		return false;
	//加载资源
	HGLOBAL hGlobal = LoadResource(hinstance, hRsrc);
	if (nullptr == hGlobal)
		return false;
	//锁定资源
	LPVOID pBuffer = LockResource(hGlobal);
	if (nullptr == pBuffer)
		return false;

	resouce = (char*&)pBuffer;// LPVOID  ---> String        
	resouce = resouce.substr(0, dwSize); //去除各种文件的附加信息。
	dataType = resouce.size();
}

void splitString(const std::string& str, std::vector<std::string>& strarray, const std::string& split)
{
	std::string::size_type pos1, pos2;
	pos2 = str.find(split);
	pos1 = 0;
	while (std::string::npos != pos2) {
		strarray.push_back(str.substr(pos1, pos2 - pos1));
		pos1 = pos2 + split.size();
		pos2 = str.find(split, pos1);
	}
	if (pos1 != str.length())
		strarray.push_back(str.substr(pos1));
}

void copywcharstr(wchar_t* dest, const wchar_t* source, int maxlength)
{
	int length = sizeof(wchar_t) * (wcslen(source) + 1);
	memcpy(dest, source, min(length, maxlength));
}

void copycharstr(char* dest, const char* source, int maxlength)
{
	int length = sizeof(char) * (strlen(source) + 1);
	memcpy(dest, source, min(length, maxlength));
}

bool loadFile(std::wstring path, std::vector<uint8_t>& data, int length)
{
	bool bIn = std::tr2::sys::exists(path);
	if (!bIn)
	{
		std::string message = "path not exist:" + wstring2string(path);
		logMessage(OEIP_WARN, message.c_str());
		return false;
	}
	try
	{
		//文件数据输入到内存
		auto fileMask = (std::ios::binary | std::ios::in);
		std::ifstream fileStream;
		fileStream.open(path, fileMask);
		fileStream.read((char*)data.data(), length);
		fileStream.close();
		return true;
	}
	catch (const std::exception&)
	{
		std::string message = "load file path:" + wstring2string(path) + " fail.";
		logMessage(OEIP_ERROR, message.c_str());
		return false;
	}
}

bool saveFile(std::wstring path, void* data, int length)
{
	try
	{
		//内存数据写入到文件
		auto fileMask = (std::ios::binary | std::ios::out);
		std::ofstream fileStream(path, fileMask);
		fileStream.write((char*)data, length);
		fileStream.close();
		return true;
	}
	catch (const std::exception&)
	{
		std::string message = "save file path:" + wstring2string(path) + " fail.";
		logMessage(OEIP_ERROR, message.c_str());
		return false;
	}
}

int64_t getNowTimestamp()
{
	std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
	auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
	time_t timestamp = tmp.count();
	return timestamp;
}

int32_t getDataType(OeipVideoType videoType)
{
	int32_t imageType = -1;
	switch (videoType)
	{
	case OEIP_VIDEO_OTHER:
		break;
	case OEIP_VIDEO_NV12:
		imageType = OEIP_CV_8UC1;
		break;
	case OEIP_VIDEO_YUY2:
		imageType = OEIP_CV_8UC1;
		break;
	case OEIP_VIDEO_YVYU:
		imageType = OEIP_CV_8UC1;
		break;
	case OEIP_VIDEO_UYVY:
		imageType = OEIP_CV_8UC1;
		break;
	case OEIP_VIDEO_MJPG:
		imageType = OEIP_CV_8UC1;
		break;
	case OEIP_VIDEO_RGB24:
		imageType = OEIP_CV_8UC3;
		break;
	case OEIP_VIDEO_ARGB32:
		imageType = OEIP_CV_8UC4;
		break;
	case OEIP_VIDEO_RGBA32:
		imageType = OEIP_CV_8UC4;
		break;
	case OEIP_VIDEO_DEPTH:
		imageType = OEIP_CV_16UC1;
		break;
	default:
		break;
	}
	return imageType;
}

std::string getLayerName(OeipLayerType layerType)
{
	std::string name = "no name layer";
	switch (layerType)
	{
	case OEIP_NONE_LAYER:
		break;
	case OEIP_INPUT_LAYER:
		name = "input layer";
		break;
	case OEIP_YUV2RGBA_LAYER:
		name = "yuv to rgba layer";
		break;
	case OEIP_MAPCHANNEL_LAYER:
		break;
	case OEIP_RGBA2YUV_LAYER:
		break;
	case OEIP_OUTPUT_LAYER:
		name = "output layer";
		break;
	case OEIP_MAX_LAYER:
		break;
	default:
		break;
	}
	return name;
}

uint32_t divUp(int32_t x, int32_t y)
{
	return (x + y - 1) / y;
}
