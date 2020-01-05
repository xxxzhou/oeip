#pragma once
#include <OeipExport.h>
#include "../oeip/AudioOutput.h"
#include <opencv2/highgui/highgui.hpp>

using namespace std;
namespace OeipAudio
{
	//std::unique_ptr<FILE, int(*)(FILE*)> f{ fopen("foo.txt", "r"), &fclose };

	vector<uint8_t> micData;
	vector<uint8_t> loopData;
	vector<uint8_t> mixData;
	OeipAudioDesc cdesc = {};

	bool bRecord = false;

	void onMixData(uint8_t* data, int32_t lenght) {
		mixData.insert(mixData.end(), data, data + lenght);
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
					setAudioOutputHandle(audioRecord);
					startAudioOutput(true, true, cdesc, onMixData);
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