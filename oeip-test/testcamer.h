#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <OeipExport.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <thread>
#include <Windows.h>
#include <fstream>

#include <../oeip/BaseLayer.h>

using namespace std;
using namespace cv;

namespace OeipCamera
{
	//template<int32_t inSize, int32_t outSize>

	cv::Mat* show = nullptr;
	int32_t devicdIndex = 0;
	int32_t formatIndex = 0;
	int32_t pipeId = 0;
	int32_t inputLayerIndex = 0;
	int32_t width = 1920;
	int32_t height = 1080;
	YUV2RGBAParamet yuip = {};

	void dataRecive(uint8_t* data, int32_t width, int32_t height) {
		//std::cout << width << height << std::endl;
		updatePipeInput(pipeId, inputLayerIndex, data);
		runPipe(pipeId);
	}

	void onPipeData(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t outputIndex) {
		//std::cout << width << height << std::endl;
		memcpy(show->ptr<char>(0), data, width * height * 4);
	}

	void testCamera() {
		initOeip();

		pipeId = initPipe(OEIP_DX11);
		setPipeDataAction(pipeId, onPipeData);

		int32_t deviceCount = getDeviceCount();
		std::vector<OeipDeviceInfo> devices;
		devices.resize(deviceCount);
		getDeviceList(devices.data(), deviceCount);
		setDeviceDataAction(devicdIndex, dataRecive);

		int32_t formatCount = getFormatCount(devicdIndex);
		std::vector<VideoFormat> formats;
		formats.resize(formatCount);
		getFormatList(devicdIndex, formats.data(), formatCount);

		setFormat(devicdIndex, formatIndex);
		openDevice(devicdIndex);

		InputParamet ip = {};
		yuip.yuvType = getVideoYUV(formats[formatIndex].videoType);
		updatePipeParamet(pipeId, 0, (void*)&ip);
		updatePipeParamet(pipeId, 1, &yuip);
		if (yuip.yuvType == OEIP_VIDEO_NV12)
			setPipeInput(pipeId, inputLayerIndex, width, height * 3 / 2, OEIP_CV_8UC1);
		else
			setPipeInput(pipeId, inputLayerIndex, width / 2, height, OEIP_CV_8UC4);
		show = new cv::Mat(height, width, CV_8UC4);
		//const char* window_name = "vvvvvvvv";
		while (int key = cv::waitKey(1)) {
			cv::imshow("a", *show);
			switch (key)
			{
			case 'q':
				updatePipeParamet(pipeId, 0, &ip);
				break;
			default:
				break;
			}
		}
	}
}
