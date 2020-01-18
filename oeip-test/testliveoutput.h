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
#include <../oeip/LivePipe.h>
#include <../oeip-ffmpeg/FLiveOutput.h>
#include <../oeip-ffmpeg/FLiveInput.h>

using namespace std;
using namespace cv;

namespace OeipLiveOutput
{
	//template<int32_t inSize, int32_t outSize>
	cv::Mat* show = nullptr;
	cv::Mat* showLive = nullptr;
	int32_t devicdIndex = 0;
	int32_t formatIndex = 0;
	int32_t inputLayerIndex = 0;
	//VideoPipe* vpipe = nullptr;
	std::unique_ptr<VideoPipe> vpipe = nullptr;
	std::unique_ptr<LivePipe> lpipe = nullptr;
	bool bPush = true;
	bool bPull = true;

	bool bRecord = false;
	VideoFormat videoFormat = {};
	std::unique_ptr<FLiveOutput> liveOut = nullptr;
	std::unique_ptr<FLiveInput> liveIn = nullptr;
	OeipAudioDesc cdesc = {};
	int32_t sampleRate = 8000;
	OeipYUVFMT fmt = OEIP_YUVFMT_YUV420P;
	std::vector<uint8_t> frameData;
	void dataRecive(uint8_t* data, int32_t width, int32_t height) {
		//std::cout << width << height << std::endl;
		vpipe->runVideoPipe(0, data);
	}

	void onPullData(OeipVideoFrame frame) {
		int32_t width = frame.width;
		int32_t height = frame.height;
		if (fmt == OEIP_YUVFMT_YUY2P) {
			memcpy(frameData.data(), frame.data[0], width * height);
			memcpy(frameData.data() + width * height, frame.data[1], width * height / 2);
			memcpy(frameData.data() + width * height * 3 / 2, frame.data[2], width * height / 2);
		}
		else if (fmt == OEIP_YUVFMT_YUV420P) {
			memcpy(frameData.data(), frame.data[0], width * height);
			memcpy(frameData.data() + width * height, frame.data[1], width * height / 4);
			memcpy(frameData.data() + width * height * 5 / 4, frame.data[2], width * height / 4);
		}
		lpipe->runVideoPipe(0, frameData.data());
	}

	void onPipeData(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t outputIndex) {
		if (vpipe->getOutputId() == layerIndex) {
			memcpy(show->ptr<char>(0), data, width * height * 4);
		}
		if (vpipe->getOutYuvId() == layerIndex) {
			if (bPush && bRecord && liveOut) {
				int32_t width = videoFormat.width;
				int32_t height = videoFormat.height;
				OeipVideoFrame vf = {};
				if (fmt == OEIP_YUVFMT_YUY2P) {
					vf.data[0] = data;
					vf.data[1] = data + width * height;
					vf.data[2] = data + width * height * 3 / 2;
				}
				else if (fmt == OEIP_YUVFMT_YUV420P) {
					vf.data[0] = data;
					vf.data[1] = data + width * height;
					vf.data[2] = data + width * height * 5 / 4;
				}
				vf.dataSize = width * height;
				vf.fmt = fmt;
				vf.height = videoFormat.height;
				vf.width = videoFormat.width;
				vf.timestamp = (uint32_t)getNowTimestamp();//av_gettime
				liveOut->pushVideo(vf);
			}
		}
	}

	void onPipeDataLive(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t outputIndex) {
		if (lpipe->getOutputId() == layerIndex) {
			memcpy(showLive->ptr<char>(0), data, width * height * 4);
		}
	}

	void onMixData(uint8_t* data, int32_t lenght) {
		if (bPush && bRecord && liveOut) {
			OeipAudioFrame af = {};
			af.channels = 1;
			af.data = (uint8_t*)data;
			af.dataSize = lenght;
			af.timestamp = (uint32_t)getNowTimestamp();
			af.sampleRate = sampleRate;
			liveOut->pushAudio(af);
		}
	}

	void test() {
		initOeip();

		OeipGpgpuType runType = OEIP_CUDA;
		vpipe = std::make_unique<VideoPipe>(runType);
		lpipe = std::make_unique<LivePipe>(runType);
		setPipeDataAction(vpipe->getPipeId(), onPipeData);
		setPipeDataAction(lpipe->getPipeId(), onPipeDataLive);

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

		videoFormat = formats[formatIndex];

		vpipe->setVideoFormat(videoFormat.videoType, videoFormat.width, videoFormat.height);
		show = new cv::Mat(videoFormat.height, videoFormat.width, CV_8UC4);
		showLive = new cv::Mat(videoFormat.height, videoFormat.width, CV_8UC4);

		if (fmt == OEIP_YUVFMT_YUV420P) {
			frameData.resize(videoFormat.height * videoFormat.width * 3 / 2);
		}
		else if (fmt == OEIP_YUVFMT_YUY2P) {
			frameData.resize(videoFormat.height * videoFormat.width * 2);
		}
		//const char* window_name = "vvvvvvvv";
		while (int key = cv::waitKey(20)) {
			cv::imshow("a", *show);
			cv::imshow("b", *showLive);
			if (key == 'q')
				break;
			if (key == 'r') {
				const char* rtmpUrl = "rtmp://192.168.0.140:8011/live/ivroom1_2_0";
				bRecord = !bRecord;
				if (bRecord) {
					cdesc.channel = 1;
					cdesc.sampleRate = sampleRate;
					cdesc.bitSize = 16;
					startAudioOutput(true, true, cdesc, onMixData);
					if (bPush) {
						liveOut = std::make_unique< FLiveOutput>();
						liveOut->open(rtmpUrl);
						lpipe->setVideoFormat(fmt, videoFormat.width, videoFormat.height);
					}
					if (bPull) {
						liveIn = std::make_unique<FLiveInput>();
						liveIn->setVideoDataEvent(onPullData);
						liveIn->open(rtmpUrl);
					}
					logMessage(OEIP_INFO, "live start");
				}
				else {
					closeAudioOutput();
					if (bPull) {
						liveIn->close();
					}
					if (bPush) {
						liveOut->close();
					}
					logMessage(OEIP_INFO, "live stop");
				}
			}
		}
		shutdownOeip();
	}
}
