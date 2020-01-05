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
#include <../oeip/VideoPipe.h>
#include <../oeip-ffmpeg/FH264Encoder.h>

using namespace std;
using namespace cv;

namespace OeipEncoderVideo
{
	//template<int32_t inSize, int32_t outSize>
	cv::Mat* show = nullptr;
	int32_t devicdIndex = 0;
	int32_t formatIndex = 0;
	int32_t inputLayerIndex = 0;
	int32_t width = 1920;
	int32_t height = 1080;
	OeipVideoType videoType = OEIP_VIDEO_OTHER;
	VideoPipe* vpipe = nullptr;
	std::unique_ptr< FH264Encoder> encoder;
	void dataRecive(uint8_t* data, int32_t width, int32_t height) {
		//std::cout << width << height << std::endl;
		vpipe->runVideoPipe(0, data);
	}

	void onPipeData(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t outputIndex) {
		if (vpipe->getOutputId() == layerIndex) {
			memcpy(show->ptr<char>(0), data, width * height * 4);
		}
		if (vpipe->getOutYuvId() == layerIndex) {
			int64_t timestamp = getNowTimestamp();
			int ret = encoder->encoder(data, width * height, timestamp);
			if (ret != 0) {
				return;
			}
			uint64_t xtimestmap = 0;
			int outLen = OEIP_H264_BUFFER_MAX_SIZE;
			vector<uint8_t> buffer(OEIP_H264_BUFFER_MAX_SIZE);
			while (1) {
				outLen = OEIP_H264_BUFFER_MAX_SIZE;
				ret = encoder->readPacket(buffer.data(), outLen, xtimestmap);
				if (ret == 0) {
					static FILE* pf = fopen("D:\\test.h264", "wb");
					if (pf)
						fwrite(buffer.data(), 1, outLen, pf);
					//fclose(pf);
				}
				else {
					break;
				}
			}
		}
	}

	void test() {
		initOeip();

		vpipe = new VideoPipe(OEIP_CUDA);
		setPipeDataAction(vpipe->getPipeId(), onPipeData);

		int32_t deviceCount = getDeviceCount();
		std::vector<OeipDeviceInfo> devices;
		devices.resize(deviceCount);
		getDeviceList(devices.data(), deviceCount);
		setDeviceDataAction(devicdIndex, dataRecive);

		int32_t formatCount = getFormatCount(devicdIndex);
		std::vector<VideoFormat> formats;
		formats.resize(formatCount);
		getFormatList(devicdIndex, formats.data(), formatCount);

		OeipVideoEncoder vencoder = {};
		vencoder.width = width;
		vencoder.height = height;
		vencoder.fps = 30;
		vencoder.bitrate = 4000000;
		vencoder.yuvType = OEIP_YUVFMT_YUV420P;
		encoder = std::make_unique< FH264Encoder>(vencoder);

		setFormat(devicdIndex, formatIndex);
		openDevice(devicdIndex);

		videoType = formats[formatIndex].videoType;
		vpipe->setVideoFormat(videoType, width, height);
		show = new cv::Mat(height, width, CV_8UC4);

		while (int key = cv::waitKey(20)) {
			cv::imshow("a", *show);
			if (key == 'q')
				break;
		}
		shutdownOeip();
	}
}
