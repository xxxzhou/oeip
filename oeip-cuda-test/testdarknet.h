#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#include "../oeip/OeipExport.h"
#include "../oeip/BaseLayer.h"
#include "../oeip/VideoPipe.h"

namespace DarknetPerson
{
	cv::Mat* show = nullptr;
	int32_t devicdIndex = 0;
	int32_t formatIndex = 0;
	int32_t width = 1920;
	int32_t height = 1080;
	OeipVideoType videoType = OEIP_VIDEO_OTHER;
	VideoPipe* vpipe = nullptr;

	void dataRecive(uint8_t* data, int32_t width, int32_t height) {
		//std::cout << width << height << std::endl;
		vpipe->runVideoPipe(0, data);
	}

	void onPipeData(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t outputIndex) {
		if (vpipe->getOutputId() == layerIndex)
			memcpy(show->ptr<char>(0), data, width * height * 4);
	}

	void testDarknetPerson() {
		initOeip();

		vpipe = new VideoPipe(OEIP_CUDA);
		int32_t darknet = addPiepLayer(vpipe->getPipeId(), "darknet", OEIP_DARKNET_LAYER);
		DarknetParamet darknetParamet = {};
		darknetParamet.bLoad = 1;
		copycharstr(&darknetParamet.confile[0], "../../ThirdParty/yolov3-tiny-test.cfg", 512);
		copycharstr(&darknetParamet.weightfile[0], "../../ThirdParty/yolov3-tiny_745000.weights", 512);
		darknetParamet.thresh = 0.3f;
		darknetParamet.nms = 0.4f;
		darknetParamet.bDraw = 1;
		darknetParamet.drawColor = getColor(1.0f, 0.1f, 0.1f, 0.5f);
		setPipeDataAction(vpipe->getPipeId(), onPipeData);
		connectLayerIndex(vpipe->getPipeId(), darknet, vpipe->getResizeId());
		updatePipeParamet(vpipe->getPipeId(), darknet, &darknetParamet);

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

		videoType = formats[formatIndex].videoType;
		vpipe->setVideoFormat(videoType, width, height);
		show = new cv::Mat(height, width, CV_8UC4);
		//const char* window_name = "vvvvvvvv";
		while (int key = cv::waitKey(20)) {
			cv::imshow("a", *show);
			if (key == 'q')
				break;
		}
		shutdownOeip();
	}
}