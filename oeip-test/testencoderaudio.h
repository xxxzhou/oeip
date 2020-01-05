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

#include <OeipExport.h>
#include "../oeip/AudioOutput.h"
#include <../oeip/BaseLayer.h>
#include <../oeip/VideoPipe.h>
#include <../oeip-ffmpeg/FAACEncoder.h>

namespace OeipEncoderAudio
{
	using namespace std;
	using namespace cv;
	vector<uint8_t> micData;
	vector<uint8_t> loopData;
	vector<uint8_t> mixData;
	OeipAudioDesc cdesc = {};
	std::unique_ptr< FAACEncoder> encoder;
	bool bRecord = false;

	void onMixData(uint8_t* data, int32_t lenght) {
		mixData.insert(mixData.end(), data, data + lenght);

		int64_t timestamp = getNowTimestamp();
		int ret = encoder->encoder(data, lenght, timestamp);
		if (ret != 0) {
			return;
		}
		uint64_t xtimestmap = 0;
		int outLen = OEIP_AAC_BUFFER_MAX_SIZE;
		vector<uint8_t> buffer(OEIP_AAC_BUFFER_MAX_SIZE);
		while (1) {
			outLen = OEIP_AAC_BUFFER_MAX_SIZE;
			ret = encoder->readPacket(buffer.data(), outLen, xtimestmap);
			if (ret == 0) {
				static FILE* pf = fopen("D:\\test.acc", "wb");
				if (pf)
					fwrite(buffer.data(), 1, outLen, pf);
				//fclose(pf);
			}
			else {
				break;
			}
		}
	}

	void audioRecord(bool bMic, uint8_t* udata, int dataLen, OeipAudioDataType type) {
		if (type == OEIP_Audio_Data || type == OEIP_AudioData_None) {
			//std::string message = "record:" + std::to_string(dataLen);
			//logMessage(zmf_info, message.c_str());
			if (bMic)
				micData.insert(micData.end(), udata, udata + dataLen);
			else
				loopData.insert(loopData.end(), udata, udata + dataLen);
		}
		else if (type == OEIP_Audio_WavHeader) {
			vector<uint8_t> chead;
			if (bMic) {
				micData.insert(micData.begin(), udata, udata + dataLen);
				static FILE* pf = fopen("D:\\1.wav", "wb");
				if (pf)
					fwrite(micData.data(), 1, micData.size(), pf);
			}
			else {
				loopData.insert(loopData.begin(), udata, udata + dataLen);
				static FILE* pf = fopen("D:\\2.wav", "wb");
				if (pf)
					fwrite(loopData.data(), 1, loopData.size(), pf);
			}
		}
	}

	void test() {
		initOeip();
		cv::Mat mat(240, 320, CV_8U);
		cv::imshow("a", mat);
		while (int key = cv::waitKey(30)) {
			if (key == 'q')
				break;
			if (key == 'r') {
				bRecord = !bRecord;
				if (bRecord) {
					cdesc.channel = 1;
					cdesc.sampleRate = 8000;
					cdesc.bitSize = 16;	

					OeipAudioEncoder vencoder = {};
					vencoder.channel = 1;
					vencoder.frequency = 8000;
					encoder = std::make_unique< FAACEncoder>(vencoder);

					setAudioOutputHandle(audioRecord);
					startAudioOutput(true, true, cdesc, onMixData);

					//vencoder.bitrate = 
					logMessage(OEIP_INFO, "record audio");
				}
				else {
					closeAudioOutput();
					vector<uint8_t> chead;
					getWavHeader(chead, mixData.size(), cdesc);
					mixData.insert(mixData.begin(), chead.begin(), chead.end());
					static FILE* pf = fopen("D:\\3.wav", "wb");
					if (pf)
						fwrite(mixData.data(), 1, mixData.size(), pf);
					logMessage(OEIP_INFO, "stop audio");
				}
			}
		}
	}
}
