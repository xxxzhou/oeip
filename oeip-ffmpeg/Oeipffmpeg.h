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

#include <OeipCommon.h>
//#include "../oeip-live/OeipLive.h"
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
	OEIP_AVFORMAT_MP4,
	OEIP_AVFORMAT_FLV,
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
	else if (uri.find(".mp4") != std::string::npos) {
		format = OEIP_AVFORMAT_MP4;
	}
	else {
		format = OEIP_AVFORMAT_FLV;
	}
	return format;
}

inline std::string getAvformatName(OeipFAVFormat format) {
	std::string name = "";
	switch (format) {
	case OEIP_AVFORMAT_FLV:
		name = "flv";
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
	case OEIP_AVFORMAT_MP4:
		name = "mp4";
		break;
	default:
		break;
	}
	return name;
}

inline bool isfile(OeipFAVFormat format) {
	return format == OEIP_AVFORMAT_MP4 || format == OEIP_AVFORMAT_FLV;
}

inline void checkRet(std::string meg, int32_t ret) {
	char error_char[AV_ERROR_MAX_STRING_SIZE];
	std::string message = meg + " ret: " + av_make_error_string(error_char, AV_ERROR_MAX_STRING_SIZE, ret);
	logMessage(OEIP_ERROR, message.c_str());
}

inline int get_sr_index(unsigned int sampling_frequency) {
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

inline int get_format_from_sample_fmt(const char** fmt, enum AVSampleFormat sample_fmt) {
	int i;
	struct sample_fmt_entry {
		enum AVSampleFormat sample_fmt; const char* fmt_be, * fmt_le;
	} sample_fmt_entries[] = {
		{ AV_SAMPLE_FMT_U8,  "u8",    "u8"    },
		{ AV_SAMPLE_FMT_S16, "s16be", "s16le" },
		{ AV_SAMPLE_FMT_S32, "s32be", "s32le" },
		{ AV_SAMPLE_FMT_FLT, "f32be", "f32le" },
		{ AV_SAMPLE_FMT_DBL, "f64be", "f64le" },
	};
	*fmt = NULL;

	for (i = 0; i < FF_ARRAY_ELEMS(sample_fmt_entries); i++) {
		struct sample_fmt_entry* entry = &sample_fmt_entries[i];
		if (sample_fmt == entry->sample_fmt) {
			*fmt = AV_NE(entry->fmt_be, entry->fmt_le);
			return 0;
		}
	}

	fprintf(stderr,
		"Sample format %s not supported as output format\n",
		av_get_sample_fmt_name(sample_fmt));
	return AVERROR(EINVAL);
}

inline bool check_sample_fmt(AVCodec* codec, enum AVSampleFormat sample_fmt) {
	const enum AVSampleFormat* p = codec->sample_fmts;
	int i = 0;
	while (p[i] != AV_SAMPLE_FMT_NONE) {
		if (p[i] == sample_fmt) {
			return true;
		}
		i++;
	}
	return false;
}

inline void buildAdts(int size, uint8_t* buffer, int samplerate, int channels) {
	char* padts = (char*)buffer;
	int profile = 2;                                            //AAC LC
	int freqIdx = get_sr_index(samplerate);                     //44.1KHz
	int chanCfg = channels;            //MPEG-4 Audio Channel Configuration. 1 Channel front-center
	padts[0] = (char)0xFF;      // 11111111     = syncword
	padts[1] = (char)0xF1;      // 1111 1 00 1  = syncword MPEG-2 Layer CRC
	padts[2] = (char)(((profile - 1) << 6) + (freqIdx << 2) + (chanCfg >> 2));
	padts[6] = (char)0xFC;
	padts[3] = (char)(((chanCfg & 3) << 6) + ((7 + size) >> 11));
	padts[4] = (char)(((7 + size) & 0x7FF) >> 3);
	padts[5] = (char)((((7 + size) & 7) << 5) + 0x1F);
}

//av_image_copy_to_buffer frame->data 	av_image_fill_arrays data->frame					
inline bool fillFFmpegFrame(uint8_t* data, const OeipVideoFrame& videoFrame) {
	if (videoFrame.fmt != OEIP_YUVFMT_YUY2P && videoFrame.fmt != OEIP_YUVFMT_YUV420P)
		return false;
	AVPixelFormat pformat = AV_PIX_FMT_YUV420P;
	int32_t dataSize = videoFrame.width * videoFrame.height * 3 / 2;
	if (videoFrame.fmt == OEIP_YUVFMT_YUY2P) {
		pformat = AV_PIX_FMT_YUV422P;
		dataSize = videoFrame.width * videoFrame.height * 2;
	}
	int ret = av_image_copy_to_buffer(data, dataSize, videoFrame.data, videoFrame.linesize,
		pformat, videoFrame.width, videoFrame.height, 1);
	return ret >= 0;
}

inline void make_dsi(int frequencyInHz, int channelCount, uint8_t* dsi) {
	int sampling_frequency_index = get_sr_index(frequencyInHz);
	unsigned int object_type = 2; // AAC LC by default
	dsi[0] = (object_type << 3) | (sampling_frequency_index >> 1);
	dsi[1] = ((sampling_frequency_index & 1) << 7) | (channelCount << 3);
}



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