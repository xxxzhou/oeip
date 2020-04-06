#include "FRtmpInput.h"
#include <climits>

int decode_interrupt_cb(void* ctx) {
	// abort blocking operation
	FRtmpInput* rInput = (FRtmpInput*)(ctx);
	//logMessage(OEIP_INFO, rInput->bTempOpen ? "OEPN." : "CLOSE");
	if (!rInput || !rInput->bTempOpen)
		return 1;// abort
	return 0;//继续
}

FRtmpInput::FRtmpInput() {
	audioPlay = std::make_unique<FAudioPlay>();
}

FRtmpInput::~FRtmpInput() {
	if (bOpenPull) {
		close();
	}
}

void FRtmpInput::readPack() {
	int ret = 0;
	//声明一桢数据	
	frame = getUniquePtr(av_frame_alloc());
	//如果有声音，需要转码
	OAVFrame aframe = getUniquePtr(av_frame_alloc());
	if (bAudio && audioIndex >= 0) {
		aframe->nb_samples = audioCtx->frame_size;
		aframe->channel_layout = AV_CH_LAYOUT_MONO;
		aframe->format = outSampleFormat;
		aframe->sample_rate = audioCtx->sample_rate;
		if ((ret = av_frame_get_buffer(aframe.get(), 0)) < 0) {
			checkRet("error allocating an audio buffer.", ret);
			onOperateAction(OEIP_DECODER_READ, ret);
			return;
		}
	}
	//最小的PTS时间
	int64_t minPts = _I64_MAX;
	int64_t minNowMs = 0;
	while (bOpenPull) {
		AVPacket packet;
		av_init_packet(&packet);
		//av_read_frame本身会阻塞当前线程(可能以分钟记),故以等待加状态改变在这无用
		if ((ret = av_read_frame(fmtCtx.get(), &packet)) < 0)
			break;
		if (packet.stream_index == videoIndex) {
			ret = avcodec_send_packet(videoCtx.get(), &packet);
			if (ret < 0) {
				checkRet("input avcodec_send_packet fail", ret);
				av_packet_unref(&packet);
				break;
			}
			while (ret >= 0) {
				ret = avcodec_receive_frame(videoCtx.get(), frame.get());
				if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
					break;
				}
				else if (ret < 0) {
					break;
				}
				//frame->pts = frame->best_effort_timestamp;
				frame->pts = av_rescale_q(frame->best_effort_timestamp, fmtCtx->streams[packet.stream_index]->time_base, av_make_q(1, 1000));
				if (onVideoDataEvent) {
					//av_image_copy_to_buffer frame->data 	av_image_fill_arrays data->frame	
					OeipVideoFrame videoFrame = {};
					int width = frame->width;
					int height = frame->height;
					//frame
					if (videoEncoder.yuvType == OEIP_YUVFMT_YUY2P) {
						videoFrame.dataSize = width * height * 2;
					}
					else if (videoEncoder.yuvType == OEIP_YUVFMT_YUV420P) {
						videoFrame.dataSize = width * height * 3 / 2;
					}
					if (videoEncoder.yuvType == OEIP_YUVFMT_YUY2P || videoEncoder.yuvType == OEIP_YUVFMT_YUV420P) {
						videoFrame.data[0] = frame->data[0];
						videoFrame.data[1] = frame->data[1];
						videoFrame.data[2] = frame->data[2];
					}
					videoFrame.width = width;
					videoFrame.height = height;
					videoFrame.fmt = videoEncoder.yuvType;
					videoFrame.timestamp = frame->pts;
					memcpy(videoFrame.linesize, frame->linesize, sizeof(int) * 4);
					onVideoDataEvent(videoFrame);
				}
				//记录最小时间
				if (frame->pts < minPts) {
					minPts = frame->pts;
					minNowMs = getNowTimestamp();
				}
				int64_t sc = (frame->pts - minPts) - (getNowTimestamp() - minNowMs);
				if (isfile(oformat) && sc > 0)
					std::this_thread::sleep_for(std::chrono::milliseconds(sc));
			}
			av_packet_unref(&packet);
			//std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		else if (packet.stream_index == audioIndex) {
			ret = avcodec_send_packet(audioCtx.get(), &packet);
			if (ret < 0) {
				checkRet("input rtmp avcodec_send_packet fail", ret);
				av_packet_unref(&packet);
				break;
			}
			while (ret >= 0) {
				ret = avcodec_receive_frame(audioCtx.get(), frame.get());
				if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
					break;
				}
				else if (ret < 0) {
					break;
				}
				frame->pts = av_rescale_q(frame->best_effort_timestamp, fmtCtx->streams[packet.stream_index]->time_base, av_make_q(1, 1000));
				int32_t dataSize = aframe->nb_samples * av_get_channel_layout_nb_channels(AV_CH_LAYOUT_MONO) * av_get_bytes_per_sample(outSampleFormat);
				if (onAudioFrameEvent) {
					swr_convert(swrCtx.get(), (uint8_t**)aframe->data, aframe->nb_samples,
						(const uint8_t**)frame->data, frame->nb_samples);
					OeipAudioFrame audioFrame = {};
					audioFrame.bitDepth = av_get_bytes_per_sample(outSampleFormat) * 8;// outSampleFormat
					audioFrame.channels = av_get_channel_layout_nb_channels(AV_CH_LAYOUT_MONO);
					audioFrame.sampleRate = audioCtx->sample_rate;
					audioFrame.dataSize = dataSize;
					//平面转交叉格式
					audioFrame.data = aframe->extended_data[0];
					audioFrame.timestamp = frame->pts;
					onAudioFrameEvent(audioFrame);
				}
				if (bPlayAudio) {
					audioPlay->playAudioData(aframe->extended_data[0], dataSize);
				}
				//记录最小时间
				if (frame->pts < minPts) {
					minPts = frame->pts;
					minNowMs = getNowTimestamp();
				}
				int64_t sc = (frame->pts - minPts) - (getNowTimestamp() - minNowMs);
				if (isfile(oformat) && sc > 0)
					std::this_thread::sleep_for(std::chrono::milliseconds(sc));
			}
			av_packet_unref(&packet);
		}
	}
	onOperateAction(OEIP_DECODER_READ, 0);
	signal.notify_all();
}

int32_t FRtmpInput::open(const char* curl, bool bVideo, bool bAudio) {
	if (bOpenPull) {
		logMessage(OEIP_INFO, "rtmp input is open.");
		return 1;
	}
	this->url = curl;
	this->bVideo = bVideo;
	this->bAudio = bAudio;
	int32_t ret = 0;
	oformat = getAvformat(this->url);
	std::string format_name = getAvformatName(oformat);
	bRtmp = oformat == OEIP_AVFORMAT_RTMP;
	{
		std::unique_lock <std::mutex> lck(mtx);
		bTempOpen = true;
		AVFormatContext* temp = avformat_alloc_context();
		//interrupt_callback需要在avformat_open_input之前设置，否则av_read_frame阻塞里应用不到
		temp->interrupt_callback.callback = decode_interrupt_cb;
		temp->interrupt_callback.opaque = this;
		if ((ret = avformat_open_input(&temp, url.c_str(), 0, nullptr)) < 0) {
			std::string msg = "rtmp input not open " + url;
			checkRet(msg, ret);
			return ret;
		}
		fmtCtx = getUniquePtr(temp);
		if ((ret = avformat_find_stream_info(fmtCtx.get(), nullptr)) < 0) {
			std::string msg = "rtmp input find stream fail" + url;
			checkRet(msg, ret);
			return ret;
		}
		//找到第一个视频流与音频流
		for (int32_t i = 0; i < fmtCtx->nb_streams; i++) {
			auto st = fmtCtx->streams[i];
			if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && videoIndex == -1) {
				videoIndex = i;
				videoEncoder.width = st->codecpar->width;
				videoEncoder.height = st->codecpar->height;
			}
			else if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && audioIndex == -1) {
				audioIndex = i;
			}
		}
		if (bVideo && videoIndex >= 0) {
			auto codec = avcodec_find_decoder(fmtCtx->streams[videoIndex]->codecpar->codec_id);
			auto temp = avcodec_alloc_context3(codec);
			if (!temp) {
				logMessage(OEIP_ERROR, "rtmp input cannot alloc video decoder");
				return -1;
			}
			videoCtx = getUniquePtr(temp);
			//如fmtCtx->streams[videoIndex]->codecpar里包含了AV_PIX_FMT_YUV422P数据
			avcodec_parameters_to_context(videoCtx.get(), fmtCtx->streams[videoIndex]->codecpar);
			videoEncoder.fps = av_q2intfloat(videoCtx->framerate);
			if (videoCtx->pix_fmt == AV_PIX_FMT_YUV420P) {
				videoEncoder.yuvType = OEIP_YUVFMT_YUV420P;
			}
			else if (videoCtx->pix_fmt == AV_PIX_FMT_YUV422P) {
				videoEncoder.yuvType = OEIP_YUVFMT_YUY2P;
			}
			if ((ret = avcodec_open2(videoCtx.get(), codec, nullptr)) < 0) {
				checkRet("rtmp input cannot open video decoder", ret);
				return ret;
			}
			videoEncoder.bitrate = videoCtx->bit_rate;
		}
		if (bAudio && audioIndex >= 0) {
			auto codec = avcodec_find_decoder(fmtCtx->streams[audioIndex]->codecpar->codec_id);
			auto temp = avcodec_alloc_context3(codec);
			if (!temp) {
				logMessage(OEIP_ERROR, "rtmp input cannot alloc audio decoder");
				return -1;
			}
			audioCtx = getUniquePtr(temp);
			//如fmtCtx->streams[videoIndex]->codecpar里包含了AV_PIX_FMT_YUV422P数据
			avcodec_parameters_to_context(audioCtx.get(), fmtCtx->streams[audioIndex]->codecpar);
			//audioEncoder
			if ((ret = avcodec_open2(audioCtx.get(), codec, nullptr)) < 0) {
				checkRet("rtmp input cannot open video decoder", ret);
				return ret;
			}
			//分配音频重采样(原始数据格式如果是平面格式，转化成交叉格式与单通)
			auto tempSwr = swr_alloc_set_opts(nullptr,
				AV_CH_LAYOUT_MONO, outSampleFormat, audioCtx->sample_rate,
				av_get_default_channel_layout(audioCtx->channels),
				audioCtx->sample_fmt, audioCtx->sample_rate, 0, nullptr);
			if (!tempSwr) {
				logMessage(OEIP_ERROR, "could not allocate resampler context!");
				return -5;
			}
			swrCtx = getUniquePtr(tempSwr);
			int32_t ret = swr_init(swrCtx.get());
			if (ret < 0) {
				checkRet("failed to initialize the resampling context!", ret);
				return ret;
			}
			audioEncoder.bitrate = audioCtx->bit_rate;
			audioEncoder.frequency = audioCtx->sample_rate;
			audioEncoder.channel = av_get_channel_layout_nb_channels(AV_CH_LAYOUT_MONO);

			//打开音频播放
			OeipAudioDesc audioInfo = {};
			audioInfo.bitSize = 16;
			audioInfo.channel = audioEncoder.channel;
			audioInfo.sampleRate = audioEncoder.frequency;
			audioPlay->openDevice(audioInfo);
		}
	}
	bOpenPull = true;
	//在这把相应的audioEncoder/videoEncoder取出
	onOperateAction(OEIP_DECODER_OPEN, ret);
	if (ret == 0) {
		logMessage(OEIP_INFO, "rtmp input open sucess.");
		std::thread ted = std::thread([&]() {
			readPack();
		});
		ted.detach();
	}
	return ret;
}

void FRtmpInput::close() {
	if (!bOpenPull)
		return;
	bTempOpen = false;
	videoIndex = -1;
	audioIndex = -1;
	//fmtCtx->max_delay = 10;
	std::unique_lock <std::mutex> lck(mtx);
	//等待信号回传	
	auto status = signal.wait_for(lck, std::chrono::seconds(2));
	if (status == std::cv_status::timeout) {
		logMessage(OEIP_WARN, "rtmp input is not closed properly.");
	}
	fmtCtx.reset();
	audioCtx.reset();
	videoCtx.reset();
	audioPlay->closeDevice();
	bOpenPull = false;
	onOperateAction(OEIP_DECODER_CLOSE, 0);
}

