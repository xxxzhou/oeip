#include "FRtmpOutput.h"

#define  ioVideoFileSize   1024*512
#define  ioAudioFileSize   1024*16

static const int avf_time_base = 1000000;
static const AVRational avf_time_base_q = { 1, avf_time_base };

void make_dsi(int frequencyInHz, int channelCount, uint8_t* dsi)
{
	int sampling_frequency_index = get_sr_index(frequencyInHz);
	unsigned int object_type = 2; // AAC LC by default
	dsi[0] = (object_type << 3) | (sampling_frequency_index >> 1);
	dsi[1] = ((sampling_frequency_index & 1) << 7) | (channelCount << 3);
}

int32_t readAudioIO(void* obj, uint8_t* buf, int32_t size) {
	FRtmpOutput* rtmp = (FRtmpOutput*)obj;
	if (rtmp) {
		return rtmp->audioIO(buf, size);
	}
	return -1;
}

int32_t readVideoIO(void* obj, uint8_t* buf, int32_t size) {
	FRtmpOutput* rtmp = (FRtmpOutput*)obj;
	if (rtmp) {
		return rtmp->videoIO(buf, size);
	}
	return -1;
}

FRtmpOutput::FRtmpOutput() {
}

FRtmpOutput::~FRtmpOutput() {
	audioPack.clear();
	videoPack.clear();
	videoCacheFrame.clear();
}

int32_t FRtmpOutput::startPush() {
	OeipFAVFormat oformat = getAvformat(url);
	std::string format_name = getAvformatName(oformat);
	bRtmp = oformat == OEIP_AVFORMAT_RTMP;

	int32_t ret = 0;
	AVDictionary* dict = nullptr;
	if (bVideo) {
		AVFormatContext* temp = avformat_alloc_context();
		uint8_t* videoBuffer = (uint8_t*)av_malloc(ioVideoFileSize);
		//avio_alloc_context 让FFmpeg从内存中读取数据(readVideoIO需要有数据读出，否则avformat_open_input不成功 )
		AVIOContext* tempIOC = avio_alloc_context(videoBuffer, ioVideoFileSize, 0, this, readVideoIO, nullptr, nullptr);
		//出了这个作用域，自动删除,包含如上videoBuffer
		OAVIOContext videoIOC = getUniquePtr(tempIOC);
		temp->pb = videoIOC.get();
		ret = avformat_open_input(&temp, "", nullptr, nullptr);
		if (ret < 0) {
			checkRet("could not open input video info.", ret);
			return ret;
		}
		//正确打开后保存AVFormatContext指针
		videoFmtCtx = getUniquePtr(temp);
		ret = avformat_find_stream_info(videoFmtCtx.get(), 0);
		if (ret < 0) {
			checkRet("failed to retrieve input video stream information.", ret);
			return ret;
		}
	}
	if (bAudio) {
		AVFormatContext* temp = avformat_alloc_context();
		uint8_t* audioBuffer = (uint8_t*)av_malloc(ioAudioFileSize);
		AVIOContext* tempIOC = avio_alloc_context(audioBuffer, ioAudioFileSize, 0, this, readAudioIO, nullptr, nullptr);
		//出了这个作用域，自动删除
		OAVIOContext audioIOC = getUniquePtr(tempIOC);
		temp->pb = audioIOC.get();
		ret = avformat_open_input(&temp, "", nullptr, nullptr);
		if (ret < 0) {
			checkRet("could not open input audio info.", ret);
			return ret;
		}
		//打开后保存AVFormatContext指针
		audioFmtCtx = getUniquePtr(temp);
		ret = avformat_find_stream_info(audioFmtCtx.get(), 0);
		if (ret < 0) {
			checkRet("failed to retrieve input audio stream information.", ret);
			return ret;
		}
	}
	//打开一个format_name类型的FormatContext
	AVFormatContext* tempOut = nullptr;
	ret = avformat_alloc_output_context2(&tempOut, nullptr, format_name.c_str(), url.c_str());
	if (ret < 0) {
		std::string msg = "url:" + url + "could not open";
		checkRet(msg, ret);
		return ret;
	}
	fmtCtx = getUniquePtr(tempOut);
	if (bVideo) {
		for (int i = 0; i < videoFmtCtx->nb_streams; i++) {
			AVStream* instream = videoFmtCtx->streams[i];
			AVStream* outstream = avformat_new_stream(fmtCtx.get(), avcodec_find_encoder(AV_CODEC_ID_H264));
			outstream->time_base = av_make_q(1, 90000);
			if (instream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
				ret = avcodec_parameters_copy(outstream->codecpar, instream->codecpar);
				if (ret < 0) {
					checkRet("failed to copy context from video input to output stream codec context", ret);
					return ret;
				}
				videoIndex = outstream->index;
				outstream->codecpar->codec_tag = 0;
				//if (fmtCtx->oformat->flags & AVFMT_GLOBALHEADER)
				//	outstream->codecpar->codec_tag |= AV_CODEC_FLAG_GLOBAL_HEADER;
			}
		}
	}
	if (bAudio) {
		for (int i = 0; i < audioFmtCtx->nb_streams; i++) {
			AVStream* instream = audioFmtCtx->streams[i];
			AVStream* outstream = avformat_new_stream(fmtCtx.get(), avcodec_find_encoder(AV_CODEC_ID_AAC));
			if (instream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
				ret = avcodec_parameters_copy(outstream->codecpar, instream->codecpar);
				if (ret < 0) {
					checkRet("failed to copy context from audio input to output stream codec context", ret);
					return ret;
				}
				if (outstream->codecpar->extradata_size == 0) {
					uint8_t* dsi2 = (uint8_t*)av_malloc(2);
					make_dsi(outstream->codecpar->sample_rate, outstream->codecpar->channels, dsi2);
					outstream->codecpar->extradata_size = 2;
					outstream->codecpar->extradata = dsi2;
				}
				audioIndex = outstream->index;
				outstream->codecpar->codec_tag = 0;
			}
		}
	}
	av_dump_format(fmtCtx.get(), 0, url.c_str(), 1);
	if (!(fmtCtx->oformat->flags & AVFMT_NOFILE)) {
		ret = avio_open(&fmtCtx->pb, url.c_str(), AVIO_FLAG_WRITE);
		if (ret < 0) {
			checkRet("open output url error.", ret);
			return ret;
		}
	}
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

int32_t FRtmpOutput::audioIO(uint8_t* buffer, int32_t bufsize) {
	int32_t size = 0;
	if (!bOpenPush) {
		size = audioPack.size();
		if (size > 0 && size < bufsize) {
			memcpy(buffer, &audioPack[0], size);
			audioPack.clear();
			return size;
		}
	}
	return 0;
}

int32_t FRtmpOutput::videoIO(uint8_t* buffer, int32_t bufsize) {
	int32_t size = 0;
	if (!bOpenPush) {
		size = videoPack.size();
		if (size > 0 && size < bufsize) {
			memcpy(buffer, &videoPack[0], size);
			videoPack.clear();
			return size;
		}
	}
	return 0;
}

int32_t FRtmpOutput::openURL(const char* url, bool bVideo, bool bAudio) {
	this->url = url;
	this->bAudio = bAudio;
	this->bVideo = bVideo;
	return 0;
}

void FRtmpOutput::close() {
	//调用过avformat_write_header需要调用下句
	if (fmtCtx && bOpenPush) {
		av_write_trailer(fmtCtx.get());
	}
	bOpenPush = false;
	audioFmtCtx.reset();
	videoFmtCtx.reset();
	fmtCtx.reset();
	bIFrameFirst = false;
	bAACFirst = false;
	audioPack.clear();
	videoPack.clear();
	onOperateAction(OEIP_LIVE_OPERATE_CLOSE, 0);
}

int32_t FRtmpOutput::pushVideo(uint8_t* data, int32_t size, uint64_t timestamp) {
	if (!bVideo)
		return 0;
	int32_t ret = 0;
	bool bKeyFrame = (data[4] & 0x1F) == 7 || !bRtmp ? true : false;//
	//第一个P桢放入IO，供推流IO模式读取相应的信息
	if (!bIFrameFirst) {
		if (!bKeyFrame)
			return 0;
		videoPack.resize(size);
		memcpy(&videoPack[0], data, size);
		bIFrameFirst = true;
	}
	//没有开始推流，同步音视频关键桢
	if (!bOpenPush && bKeyFrame) {
		if ((bAudio && bAACFirst) || (!bAudio && bVideo)) {
			ret = startPush();
			onOperateAction(OEIP_LIVE_OPERATE_OPEN, ret);
			if (ret < 0)
				return ret;			
		}
	}
	//缓存还没打开推流的那段时间,存入最新的一个GOP数据,GOP(M=3, N=12,IBBPBBPBBPBBI)
	if (!bOpenPush) {
		if (bKeyFrame)
			videoCacheFrame.clear();
		std::shared_ptr<StreamFrame> frame(new StreamFrame(data, size, timestamp));
		videoCacheFrame.push_back(frame);
		return 0;
	}
	else {
		//把缓存的数据GOP非I桢数据推入
		static bool bCacheClean = false;
		if (!bCacheClean && !videoCacheFrame.empty()) {
			bCacheClean = true;
			if (!bKeyFrame) {
				for (int i = 0; i < videoCacheFrame.size(); i++) {
					auto frame = videoCacheFrame[i];
					pushVideo(frame->data, frame->size, frame->ts);
				}
			}
			videoCacheFrame.clear();
			bCacheClean = false;
		}
	}
	if (bRtmp) {
		int32_t key = (data[4] & 0x1F);
		if (!((data[0] == 0 && data[1] == 0) && (key == 7 || key == 1 || key == 5))) {
			logMessage(OEIP_ERROR, "error video frame.");
			return -2;
		}
	}
	//发送packet包
	AVStream* instream = videoFmtCtx->streams[0];
	AVStream* outstream = fmtCtx->streams[videoIndex];
	AVPacket pkt;
	pkt.buf = nullptr;
	pkt.data = data;
	pkt.size = size;
	if (!videoTimestamp || videoTimestamp > timestamp)
		videoTimestamp = timestamp;
	pkt.pts = timestamp;
	pkt.dts = pkt.pts;
	pkt.duration = (int64_t)(1000 / av_q2d(instream->r_frame_rate));
	pkt.pos = -1;
	pkt.stream_index = outstream->index;
	pkt.side_data = nullptr;
	pkt.side_data_elems = 0;
	pkt.flags = bKeyFrame ? AV_PKT_FLAG_KEY : 0;
	ret = av_interleaved_write_frame(fmtCtx.get(), &pkt);
	if (ret < 0) {
		checkRet("error write video frame", ret);
	}
	av_packet_unref(&pkt);
	return ret;
}

int32_t FRtmpOutput::pushAudio(uint8_t* data, int32_t size, uint64_t timestamp) {
	if (!bAudio)
		return 0;
	int32_t ret = 0;
	if (!bAACFirst) {
		audioPack.resize(size);
		memcpy(&audioPack[0], data, size);
		bAACFirst = true;
		//如果需要推视频，保证已经有视频信息，则开始推流，否则等视频那边开始推
		if ((bVideo && bIFrameFirst) || (!bVideo && bAudio)) {
			ret = startPush();
			onOperateAction(OEIP_LIVE_OPERATE_OPEN, ret);
			if (ret < 0)
				return ret;			
		}
	}
	//音频数据不缓存
	if (!bOpenPush)
		return 0;
	timestamp = (uint32_t)timestamp;
	if (!audioTimestamp || audioTimestamp > timestamp)
		audioTimestamp = timestamp;
	AVStream* instream = audioFmtCtx->streams[0];
	AVStream* outstream = fmtCtx->streams[audioIndex];
	AVPacket pkt;
	av_init_packet(&pkt);
	if (bRtmp) {
		pkt.data = (uint8_t*)(data + 7);
		pkt.size = size - 7;
	}
	else {
		pkt.data = (uint8_t*)(data);
		pkt.size = size;
	}
	pkt.pts = timestamp;
	pkt.dts = pkt.pts;
	pkt.duration = 1000 * 1024.f / instream->codecpar->sample_rate;
	pkt.pos = -1;
	pkt.stream_index = outstream->index;
	ret = av_interleaved_write_frame(fmtCtx.get(), &pkt);
	if (ret < 0) {
		checkRet("error write audio frame", ret);
	}
	av_packet_unref(&pkt);
	return ret;
}
