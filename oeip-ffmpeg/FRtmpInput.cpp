#include "FRtmpInput.h"
#include <future>

int decode_interrupt_cb(void* ctx) {
	// abort blocking operation
	FRtmpInput* rInput = (FRtmpInput*)(ctx);
	//logMessage(OEIP_INFO, rInput->bTempOpen ? "OEPN." : "CLOSE");
	if (rInput->bTempOpen)
		return 0;
	else
		return 1;// abort
}

FRtmpInput::FRtmpInput() {
}

FRtmpInput::~FRtmpInput() {
	//videoData.clear();
}

void FRtmpInput::readPack() {
	//声明一桢数据
	auto temp = av_frame_alloc();
	frame = getUniquePtr(temp);
	while (bOpenPull) {
		int ret = 0;
		AVPacket packet;
		av_init_packet(&packet);
		//av_read_frame本身会阻塞当前线程(可能以分钟记),故以等待加状态改变在这无用
		if ((ret = av_read_frame(fmtCtx.get(), &packet)) < 0)
			break;
		if (packet.stream_index == videoIndex) {
			ret = avcodec_send_packet(videoCtx.get(), &packet);
			if (ret < 0) {
				checkRet("input rtmp avcodec_send_packet fail", ret);
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
				frame->pts = frame->best_effort_timestamp;
				if (onVideoDataEvent) {
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
					onVideoDataEvent(videoFrame);
				}
			}
			av_packet_unref(&packet);
		}
		//std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	signal.notify_all();
}

int32_t FRtmpInput::openURL(const char* curl, bool bVideo, bool bAudio) {
	if (bOpenPull) {
		logMessage(OEIP_INFO, "rtmp input is open.");
		return 1;
	}
	this->url = curl;
	this->bVideo = bVideo;
	this->bAudio = bAudio;
	OeipFAVFormat oformat = getAvformat(this->url);
	std::string format_name = getAvformatName(oformat);
	bRtmp = oformat == OEIP_AVFORMAT_RTMP;
	//std::thread ted = std::thread([&]() {
	std::future<int32_t> openUrl = std::async([&]() {
		bTempOpen = true;
		std::unique_lock <std::mutex> lck(mtx);
		int32_t ret = 0;
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
		if (bVideo && videoIndex != -1) {
			auto codec = avcodec_find_decoder(fmtCtx->streams[videoIndex]->codecpar->codec_id);
			auto temp = avcodec_alloc_context3(codec);
			if (!temp) {
				logMessage(OEIP_ERROR, "rtmp input cannot alloc video decoder");
				return -1;
			}
			videoCtx = getUniquePtr(temp);
			//如fmtCtx->streams[videoIndex]->codecpar里包含了AV_PIX_FMT_YUV422P数据
			avcodec_parameters_to_context(videoCtx.get(), fmtCtx->streams[videoIndex]->codecpar);
			if (videoCtx->pix_fmt == AV_PIX_FMT_YUV420P) {
				videoEncoder.yuvType = OEIP_YUVFMT_YUV420P;
				//videoData.resize(videoEncoder.width * videoEncoder.height * 3 / 2);
			}
			else if (videoCtx->pix_fmt == AV_PIX_FMT_YUV422P) {
				videoEncoder.yuvType = OEIP_YUVFMT_YUY2P;
				//videoData.resize(videoEncoder.width * videoEncoder.height * 2);
			}
			if ((ret = avcodec_open2(videoCtx.get(), codec, nullptr)) < 0) {
				checkRet("rtmp input cannot open video decoder", ret);
				return ret;
			}
		}
		bOpenPull = true;
		return 0;
	});	
	int32_t ret = openUrl.get();
	onOperateAction(OEIP_LIVE_OPERATE_OPEN, ret);
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
	bOpenPull = false;
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
	onOperateAction(OEIP_LIVE_OPERATE_CLOSE, 0);
}

