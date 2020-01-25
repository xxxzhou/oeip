#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <OeipExport.h>
#include <OeipCommon.h>
#include <../oeip/VideoPipe.h>
#include <../oeip/LivePipe.h>
#include "../oeip-live/OeipLiveExport.h"
#include <memory>

using namespace std;
using namespace cv;

namespace OeipTestLiveRoom
{
	class FTLiveBack :public OeipLiveBack
	{
	public:
		FTLiveBack() {};
		~FTLiveBack() {};
	private:
		// 通过 OeipLiveBack 继承
		virtual void onInitRoom(int32_t code) override {};

		virtual void onLoginRoom(int32_t code, int32_t userId) override {};

		virtual void onUserChange(int32_t userId, bool bAdd) override {};

		virtual void onStreamUpdate(int32_t userId, int32_t index, bool bAdd) override {};

		virtual void onLogoutRoom(int32_t code) override {};

		virtual void onOperateResult(int32_t operate, int32_t code, std::string message) override {};

		virtual void onPushStream(int32_t index, int32_t code) override {};

		virtual void onPullStream(int32_t userId, int32_t index, int32_t code) override {};

	};

	cv::Mat* show = nullptr;
	cv::Mat* showLive = nullptr;
	int32_t devicdIndex = 0;
	int32_t formatIndex = 0;
	int32_t inputLayerIndex = 0;
	VideoFormat videoFormat = {};
	std::unique_ptr<VideoPipe> vpipe = nullptr;

	void dataRecive(uint8_t* data, int32_t width, int32_t height) {
		//std::cout << width << height << std::endl;
		vpipe->runVideoPipe(0, data);
	}

	void onPipeData(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t outputIndex) {
		if (vpipe->getOutputId() == layerIndex) {
			memcpy(show->ptr<char>(0), data, width * height * 4);
		}
		if (vpipe->getOutYuvId() == layerIndex) {
		}
	}

	inline void test() {
		//初始仳
		initOeip();
		initOeipLive();
		OeipGpgpuType runType = OEIP_DX11;
		vpipe = std::make_unique<VideoPipe>(runType);
		setPipeDataAction(vpipe->getPipeId(), onPipeData);
		//得到设备
		int32_t deviceCount = getDeviceCount();
		std::vector<OeipDeviceInfo> devices;
		devices.resize(deviceCount);
		getDeviceList(devices.data(), deviceCount);
		setDeviceDataAction(devicdIndex, dataRecive);
		//得到设备格式
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

		std::unique_ptr<FTLiveBack> liveBack = std::make_unique< FTLiveBack>();
		OeipLiveContext ctx = {};
		copycharstr(ctx.liveServer, "http://localhost:6110", sizeof(ctx.liveServer));
		initLiveRoom(ctx, liveBack.get());
		loginRoom("111", 21);
		while (int key = cv::waitKey(20)) {
			cv::imshow("a", *show);
			cv::imshow("b", *showLive);
			if (key == 'q') {
				shutdownOeipLive();
				break;
			}
		}
	}
}
