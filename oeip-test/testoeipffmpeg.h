#pragma once

#include <../oeip-ffmpeg/FRtmpInput.h>
#include <../oeip-ffmpeg/FAudioPlay.h>
#include <../oeip-ffmpeg/FMuxing.h>
#include <../oeip/LivePipe.h>

namespace OeipFFmpeg
{
	std::unique_ptr<FRtmpInput> inputMF;
	//std::unique_ptr<FAudioPlay> audioPlay;
	std::unique_ptr<LivePipe> lpipe = nullptr;
	std::unique_ptr<FMuxing> muxing = nullptr;

	const char* src_filename = "D:\\tnmil3.flv";//tnmil3.flv 2.mp4
	const char* dst_filename = "D:\\tnmil3d.mp4";
	bool bClose = false;

	FILE* videoDestFile = nullptr;
	FILE* audioDestFile = nullptr;
	OeipAudioDesc audioDesc = {};
	vector<uint8_t> mixData;
	std::vector<uint8_t> frameData;

	cv::Mat* show = nullptr;
	OeipVideoEncoder videoInfo;

	void onAudioFrame(OeipAudioFrame frame) {
		mixData.insert(mixData.end(), frame.data, frame.data + frame.dataSize);
		//audioPlay->playAudioData(frame.data, frame.dataSize);
		muxing->pushAudio(frame);
	}

	void onVideoFrame(OeipVideoFrame frame) {
		//把align数据变成连续不分隔的	
		//fillFFmpegFrame(frameData.data(), frame);		
		getVideoFrameData(frameData.data(), frame);
		lpipe->runVideoPipe(0, frameData.data());
		fillVideoFrame(frameData.data(), frame);
		muxing->pushVideo(frame);
	}

	void onReadEvent(int type, int code) {
		if (type == OEIP_DECODER_OPEN) {
			bClose = false;
			audioDestFile = fopen("D:\\3.wav", "wb");
			OeipAudioEncoder audioInfo;
			inputMF->getAudioInfo(audioInfo);
			audioDesc.bitSize = 16;
			audioDesc.channel = audioInfo.channel;
			audioDesc.sampleRate = audioInfo.frequency;

			//audioPlay->openDevice(audioDesc);
			inputMF->getVideoInfo(videoInfo);
			if (videoInfo.yuvType == OEIP_YUVFMT_YUV420P) {
				frameData.resize(videoInfo.height * videoInfo.width * 3 / 2);
			}
			else if (videoInfo.yuvType == OEIP_YUVFMT_YUY2P) {
				frameData.resize(videoInfo.height * videoInfo.width * 2);
			}
			lpipe->setVideoFormat(videoInfo.yuvType, videoInfo.width, videoInfo.height);
			//保存音视频
			muxing->setAudioEncoder(audioInfo);
			muxing->setVideoEncoder(videoInfo);
			muxing->open(dst_filename, true, true);


		}
		else if (type == OEIP_DECODER_READ) {
			bClose = true;
			//audioPlay->closeDevice();
			muxing->close();
		}
		else if (type == OEIP_DECODER_CLOSE) {
			vector<uint8_t> chead;
			getWavHeader(chead, mixData.size(), audioDesc);
			mixData.insert(mixData.begin(), chead.begin(), chead.end());
			if (audioDestFile)
				fwrite(mixData.data(), 1, mixData.size(), audioDestFile);
		}
	}

	void onPipeData(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t outputIndex) {
		if (lpipe->getOutputId() == layerIndex) {
			memcpy(show->ptr<char>(0), data, width * height * 4);
		}
	}

	void test() {
		initOeip();
		lpipe = std::make_unique<LivePipe>(OEIP_CUDA);
		setPipeDataAction(lpipe->getPipeId(), onPipeData);

		muxing = std::make_unique< FMuxing>();
		//播放音频
		//audioPlay = std::make_unique<FAudioPlay>();
		inputMF = std::make_unique<FRtmpInput>();
		inputMF->setAudioDataEvent(onAudioFrame);
		inputMF->setVideoDataEvent(onVideoFrame);
		inputMF->setOperateEvent(onReadEvent);
		inputMF->open(src_filename, true, true);
		show = new cv::Mat(videoInfo.height, videoInfo.width, CV_8UC4);		
		while (int key = cv::waitKey(5)) {
			if (show == nullptr) {
				continue;
			}
			cv::imshow("a", *show);
			if (key == 'q')
				break;
			if (bClose)
				break;
		}
		//while (!bClose) {
		//	this_thread::sleep_for(std::chrono::milliseconds(1));
		//}
		inputMF->close();
		frameData.clear();
	}
}
