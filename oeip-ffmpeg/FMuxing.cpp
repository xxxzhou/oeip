#include "FMuxing.h"
#include "FAACEncoder.h"
#include "FH264Encoder.h"
#include "FAudioOutput.h"
#include "FRtmpInput.h"

#ifdef __cplusplus
extern "C" {
#endif
#include <SDL.h>
#include <SDL_audio.h>
#ifdef __cplusplus
}
#endif

FMuxing::FMuxing() {
	videoBuffer.resize(OEIP_H264_BUFFER_MAX_SIZE);
	audioBuffer.resize(OEIP_AAC_BUFFER_MAX_SIZE);
}

FMuxing::~FMuxing() {
}

void FMuxing::setVideoEncoder(OeipVideoEncoder vEncoder) {
	std::unique_lock <std::mutex> lck(mtx);
	//如果在推流，返回false
	videoInfo = vEncoder;
	//videoEncoder = std::unique_ptr<FH264Encoder>(new FH264Encoder(vEncoder));
	videoEncoder = std::make_unique< FH264Encoder>(vEncoder);
}

void FMuxing::setAudioEncoder(OeipAudioEncoder aEncoder) {
	std::unique_lock <std::mutex> lck(mtx);
	audioInfo = aEncoder;
	//audioEncoder = std::unique_ptr<FAACEncoder>(new FAACEncoder(aEncoder));
	audioEncoder = std::make_unique< FAACEncoder>(aEncoder);
}

int32_t FMuxing::open(const char* curl, bool bVideo, bool bAudio) {
	std::unique_lock <std::mutex> lck(mtx);
	this->bAudio = bAudio;
	this->bVideo = bVideo;
	url = curl;
	OeipFAVFormat oformat = getAvformat(url);
	std::string format_name = getAvformatName(oformat);
	bRtmp = oformat == OEIP_AVFORMAT_RTMP;
	int32_t ret = 0;
	//打开一个format_name类型的FormatContext
	AVFormatContext* tempOut = nullptr;
	ret = avformat_alloc_output_context2(&tempOut, nullptr, format_name.c_str(), url.c_str());//"mp4" format_name.c_str()
	if (ret < 0) {
		std::string msg = "url:" + url + "could not open";
		checkRet(msg, ret);
		return ret;
	}
	fmtCtx = getUniquePtr(tempOut);
	AVStream* stream = nullptr;
	if (bVideo) {
		stream = avformat_new_stream(fmtCtx.get(), videoEncoder->getCodecCtx()->codec);
		if (stream) {
			avcodec_parameters_from_context(stream->codecpar, videoEncoder->getCodecCtx());
			stream->r_frame_rate = av_make_q(videoInfo.fps, 1);
			stream->codecpar->codec_tag = 0;
			videoIndex = stream->index;
		}
	}
	if (bAudio) {
		stream = avformat_new_stream(fmtCtx.get(), audioEncoder->getCodecCtx()->codec);
		if (stream) {
			avcodec_parameters_from_context(stream->codecpar, audioEncoder->getCodecCtx());
			if (stream->codecpar->extradata_size == 0) {
				uint8_t* dsi2 = (uint8_t*)av_malloc(2);
				make_dsi(stream->codecpar->sample_rate, stream->codecpar->channels, dsi2);
				stream->codecpar->extradata_size = 2;
				stream->codecpar->extradata = dsi2;
			}
			stream->codecpar->codec_tag = 0;
			audioIndex = stream->index;

		}
	}
	av_dump_format(fmtCtx.get(), 0, url.c_str(), 1);
	//如果是文件，需要这步
	if (!(fmtCtx->oformat->flags & AVFMT_NOFILE)) {
		ret = avio_open(&fmtCtx->pb, url.c_str(), AVIO_FLAG_WRITE);
		if (ret < 0) {
			checkRet("avio_open fail:", ret);
			return ret;
		}
	}
	AVDictionary* dict = nullptr;
	av_dict_set(&dict, "rtsp_transport", "tcp", 0);
	av_dict_set(&dict, "muxdelay", "0.0", 0);
	ret = avformat_write_header(fmtCtx.get(), &dict);
	if (ret != 0) {
		checkRet("could not write header", ret);
		return ret;
	}
	bOpenPush = true;
	logMessage(OEIP_INFO, "push success.");
	return 0;
}

void FMuxing::close() {
	std::unique_lock <std::mutex> lck(mtx);
	int ret = 0;
	if (fmtCtx && bOpenPush) {
		ret = av_write_trailer(fmtCtx.get());
	}
	fmtCtx.reset();
	videoIndex = -1;
	audioIndex = -1;
	bOpenPush = false;
}

int32_t FMuxing::pushVideo(const OeipVideoFrame& videoFrame) {
	std::unique_lock <std::mutex> lck(mtx);
	if (!videoEncoder || videoIndex < 0)
		return -1;
	//传入的时间基限定是毫秒
	int64_t timestamp = videoFrame.timestamp;// av_rescale_q(videoFrame.timestamp, av_make_q(1, 1000), fmtCtx->streams[videoIndex]->time_base);
	int32_t ret = videoEncoder->encoder((uint8_t**)videoFrame.data, videoFrame.dataSize, timestamp);
	if (ret < 0) {
		return ret;
	}
	int outLen = OEIP_H264_BUFFER_MAX_SIZE;
	uint64_t timestmap = 0;
	while (true) {
		outLen = OEIP_H264_BUFFER_MAX_SIZE;
		ret = videoEncoder->readPacket(videoBuffer.data(), outLen, timestmap);
		//在这timestmap需要转换下?
		if (ret < 0)
			break;
		bool bKeyFrame = (videoBuffer.data()[4] & 0x1F) == 7 || !bRtmp ? true : false;//
		AVPacket pkt;
		av_init_packet(&pkt);
		pkt.data = videoBuffer.data();
		pkt.size = outLen;
		pkt.pts = timestmap;
		pkt.dts = pkt.pts;
		if (videoInfo.fps != 0)
			pkt.duration = 1000 / videoInfo.fps;// av_rescale_q(1000, av_make_q(1, 1000), fmtCtx->streams[videoIndex]->time_base) / videoInfo.fps;
		pkt.stream_index = videoIndex;
		if (bRtmp)
			pkt.flags = bKeyFrame ? AV_PKT_FLAG_KEY : 0;
		//限定输入的时间以毫秒为单位
		av_packet_rescale_ts(&pkt, av_make_q(1, 1000), fmtCtx->streams[videoIndex]->time_base);
		ret = av_interleaved_write_frame(fmtCtx.get(), &pkt);
		if (ret < 0) {
			checkRet("error write audio frame", ret);
		}
		av_packet_unref(&pkt);
	}
	return 0;
}

int32_t FMuxing::pushAudio(const OeipAudioFrame& audioFrame) {
	std::unique_lock <std::mutex> lck(mtx);
	if (!audioEncoder || audioIndex < 0)
		return -1;
	int64_t timestamp = audioFrame.timestamp;//  av_rescale_q(audioFrame.timestamp, av_make_q(1, 1000), fmtCtx->streams[audioIndex]->time_base);
	int32_t ret = audioEncoder->encoder((uint8_t**)&audioFrame.data, audioFrame.dataSize, timestamp);
	if (ret < 0) {
		return ret;
	}
	int outLen = OEIP_AAC_BUFFER_MAX_SIZE;
	uint64_t timestmap = 0;
	while (true) {
		outLen = OEIP_AAC_BUFFER_MAX_SIZE;
		ret = audioEncoder->readPacket(audioBuffer.data(), outLen, timestmap);
		if (ret < 0)
			break;
		AVPacket pkt;
		av_init_packet(&pkt);
		//if (bRtmp) {
		pkt.data = audioBuffer.data() + 7;
		pkt.size = outLen - 7;
		//}
		//else {
		//	pkt.data = audioBuffer.data();
		//	pkt.size = outLen;
		//}
		pkt.pts = timestmap;
		pkt.dts = pkt.pts;
		pkt.duration = audioEncoder->getCodecCtx()->frame_size;// 1000 * 1024.f / audioInfo.frequency;
		pkt.pos = -1;
		pkt.stream_index = audioIndex;
		////限定输入的时间以毫秒为单位
		av_packet_rescale_ts(&pkt, av_make_q(1, 1000), fmtCtx->streams[audioIndex]->time_base);
		ret = av_interleaved_write_frame(fmtCtx.get(), &pkt);
		if (ret < 0) {
			checkRet("error write audio frame", ret);
		}
		av_packet_unref(&pkt);
	}
	return 0;
}

bool bCanLoad() {
	auto version = avformat_version();
	//用SDL来播放音频
	int ret = SDL_Init(SDL_INIT_AUDIO);//SDL_Quit();
	return version > 0;
}

void registerFactory() {
	registerFactory(new FAudioOutputFactory(), 0, "ffmpeg output");
	registerFactory(new FRtmpInputFactory(), 0, "media play");
	registerFactory(new FMuxingFactory(), 0, "write media");
}