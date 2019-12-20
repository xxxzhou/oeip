#include "VideoDevice.h"
#include <filesystem>
#include <fstream>
#include <regex>

VideoDevice::VideoDevice()
{
}

VideoDevice::~VideoDevice()
{
}

bool VideoDevice::setFormat(uint32_t index)
{
	if (index < 0 || index >= videoFormats.size())
		index = 0;
	bool open = bOpen();
	if (open && selectIndex != index) {
		closeDevice();
	}
	selectIndex = index;
	videoFormat = videoFormats[selectIndex];
	if (open) {
		openDevice();
	}
	return true;
}

void VideoDevice::getFormat(uint32_t index, VideoFormat& format)
{
	if (index >= videoFormats.size()) {
		logMessage(OEIP_WARN, "video capture getformat incorrect index");
		index = 0;
	}
	format = videoFormats[index];
}

bool VideoDevice::saveCameraParameters(std::wstring path)
{
	if (path.empty()) {
		return false;
	}
	bool bIn = std::tr2::sys::exists(path);
	if (!bIn) {
		std::tr2::sys::create_directories(path);
	}
	//普通的ID有特殊字符,故先用设备名字
	std::wstring fileName = getDeviceID();
	std::wregex reg(L"[^a-zA-Z0-9]+");
	std::wstring fname = regex_replace(fileName, reg, L"");
	std::wstring filePath = path + L"//" + fname + L".pam";
	try {
		auto fileMask = (std::ios::binary | std::ios::out);
		std::ofstream fileStream(filePath, fileMask);
		auto param = getParametrs();
		fileStream.write((char*)&param, sizeof(CamParametrs));
		fileStream.close();
		return true;
	}
	catch (const std::exception&) {
		std::string message = "save camera parameters fail.";
		logMessage(OEIP_ERROR, message.c_str());
		return false;
	}
}

bool VideoDevice::loadCameraParameters(std::wstring path)
{
	std::wstring fileName = getDeviceID();
	std::wregex reg(L"[^a-zA-Z0-9]+");
	std::wstring fname = regex_replace(fileName, reg, L"");
	std::wstring filePath = path + L"//" + fname + L".pam";
	bool bIn = std::tr2::sys::exists(filePath);
	if (!bIn)
		return false;
	try {
		CamParametrs param = {};
		//文件数据输入到内存
		auto fileMask = (std::ios::binary | std::ios::in);
		std::ifstream fileStream;
		fileStream.open(filePath, fileMask);
		fileStream.read((char*)&param, sizeof(CamParametrs));
		fileStream.close();
		//应用保存的摄像机参数
		setParametrs(param);
		return true;
	}
	catch (const std::exception&) {
		std::string message = "load camera parameters fail.";
		logMessage(OEIP_ERROR, message.c_str());
		return false;
	}
}

void VideoDevice::onDeviceHandle(OeipDeviceEventType eventType, int32_t data)
{
	if (onDeviceEvent) {
		onDeviceEvent(eventType, data);
	}
}

void VideoDevice::onDataHandle(unsigned long lenght, uint8_t* data)
{
	if (onProcessData) {
		//OeipImageType imageType = getImageType(videoFormat.videoType);
		//int32_t elementCount = (int32_t)imageType;
		onProcessData(data, videoFormat.width, videoFormat.height);
	}
}
