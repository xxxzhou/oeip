#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include "libavutil/opt.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
#include "libswresample/swresample.h"

#ifdef __cplusplus
}
#endif

#include "../oeip/OeipCommon.h"
#include "../oeip-live/OeipLive.h"
#include <string>
#include <memory>

#ifdef OEIPFMEDIA_EXPORT
#define OEIPFMDLL_EXPORT __declspec(dllexport)
#else
#define OEIPFMDLL_EXPORT __declspec(dllimport)
#endif


enum OeipFAVFormat : int32_t
{
	OEIP_AVFORMAT_OTHER,
	OEIP_AVFORMAT_RTMP,
	OEIP_AVFORMAT_HTTP,
	OEIP_AVFORMAT_RTSP,
};

class StreamFrame
{
public:
	StreamFrame(uint8_t* data, int size, uint64_t ts) {
		this->size = size;
		this->ts = ts;
		this->data = (uint8_t*)malloc(size);
		memcpy(this->data, data, size);
	}
	~StreamFrame() {
		if (data) {
			free(data);
		}
	}
public:
	uint8_t* data;
	int size;
	uint64_t ts;
};

inline OeipFAVFormat getAvformat(std::string uri) {
	OeipFAVFormat format = OEIP_AVFORMAT_OTHER;
	if (uri.find("rtmp://") == 0) {
		format = OEIP_AVFORMAT_RTMP;
	}
	else if (uri.find("http://") == 0 || uri.find("udp://") == 0) {
		format = OEIP_AVFORMAT_HTTP;
	}
	else if (uri.find("rtsp://") == 0) {
		format = OEIP_AVFORMAT_RTSP;
	}
	return format;
}

inline std::string getAvformatName(OeipFAVFormat format) {
	std::string name = "";
	switch (format) {
	case OEIP_AVFORMAT_OTHER:
		break;
	case OEIP_AVFORMAT_RTMP:
		name = "flv";
		break;
	case OEIP_AVFORMAT_HTTP:
		name = "mpegts";
		break;
	case OEIP_AVFORMAT_RTSP:
		name = "rtsp";
		break;
	default:
		break;
	}
	return name;
}

inline void checkRet(std::string meg, int32_t ret) {
	char error_char[AV_ERROR_MAX_STRING_SIZE];
	std::string message = meg + " ret: " + av_make_error_string(error_char, AV_ERROR_MAX_STRING_SIZE, ret);
	logMessage(OEIP_ERROR, message.c_str());
}

inline int get_sr_index(unsigned int sampling_frequency)
{
	switch (sampling_frequency) {
	case 96000: return 0;
	case 88200: return 1;
	case 64000: return 2;
	case 48000: return 3;
	case 44100: return 4;
	case 32000: return 5;
	case 24000: return 6;
	case 22050: return 7;
	case 16000: return 8;
	case 12000: return 9;
	case 11025: return 10;
	case 8000:  return 11;
	case 7350:  return 12;
	default:    return 0;
	}
	return 0;
}

typedef std::function<void(int32_t operate, int32_t code)> onOperateHandle;
typedef std::function<void(OeipVideoFrame)> onVideoDataHandle;
//inline int enumVideoCodec() {
//	AVCodec* codec = NULL;
//	int ret = -1;	
//	while (codec = av_codec_next(codec)) {
//		if (av_codec_is_encoder(codec)) {
//			if (codec->type == AVMEDIA_TYPE_VIDEO) {
//				std::string msg = "encoder video:" + std::string(codec->name) + " have.";
//				logMessage(OEIP_INFO, msg.c_str());
//				ret = 1;
//			}
//		}
//	}
//	return ret;
//}