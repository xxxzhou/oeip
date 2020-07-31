#include "VideoDevice.h"
#include <filesystem>
#include <fstream>
#include <regex>

#if MSVC_PLATFORM_TOOLSET > 140
namespace stdfs = std::filesystem;
#else
namespace stdfs = std::tr2::sys;
#endif

VideoDevice::VideoDevice() {
}

VideoDevice::~VideoDevice() {
}

int32_t VideoDevice::findFormatIndex(int32_t width, int32_t height, int32_t fps) {
	int32_t index = 0;
	if (videoFormats.size() < 0)
		return -1;
	bool bFind = false;
	int32_t first = -1;
	int32_t second = -1;
	int32_t three = 0;
	VideoFormat preFormat = videoFormats[0];
	for (const VideoFormat& format : videoFormats) {
		if (format.width == width && format.height == height && format.fps == fps) {
			bFind = true;
			//尽量不选MJPG,多了解码的消耗
			if (format.videoType != OEIP_VIDEO_MJPG)
				first = index;
			else
				second = index;
		}
		//选一个分辨率最大的
		if (format.height >= preFormat.height && format.width >= preFormat.height && format.fps >= 20 && format.fps <= 60) {
			//桢优先然后是格式
			if (format.fps > preFormat.fps || (format.fps == preFormat.fps && format.videoType != OEIP_VIDEO_MJPG)) {
				three = index;
				preFormat = format;
			}
		}
		index++;
	}
	if (bFind) {
		return first >= 0 ? first : second;
	}
	return three;
}

bool VideoDevice::setFormat(uint32_t index) {
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

void VideoDevice::getFormat(uint32_t index, VideoFormat& format) {
	if (index >= videoFormats.size()) {
		logMessage(OEIP_WARN, "video capture getformat incorrect index");
		index = 0;
	}
	format = videoFormats[index];
}

bool VideoDevice::saveCameraParameters(std::wstring path) {
	if (path.empty()) {
		return false;
	}
	bool bIn = stdfs::exists(path);
	if (!bIn) {
		stdfs::create_directories(path);
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

bool VideoDevice::loadCameraParameters(std::wstring path) {
	std::wstring fileName = getDeviceID();
	std::wregex reg(L"[^a-zA-Z0-9]+");
	std::wstring fname = regex_replace(fileName, reg, L"");
	std::wstring filePath = path + L"//" + fname + L".pam";

	bool bIn = stdfs::exists(filePath);// filesystem
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

void VideoDevice::onDeviceHandle(OeipDeviceEventType eventType, int32_t data) {
	if (onDeviceEvent) {
		onDeviceEvent(eventType, data);
	}
}

void VideoDevice::onDataHandle(unsigned long lenght, uint8_t* data) {
	if (onProcessData) {
		//OeipImageType imageType = getImageType(videoFormat.videoType);
		//int32_t elementCount = (int32_t)imageType;
		onProcessData(data, videoFormat.width, videoFormat.height);
	}
}
