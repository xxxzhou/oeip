#include "FH264Encoder.h"
#include "config.h"

FH264Encoder::FH264Encoder(const OeipVideoEncoder& videoDesc) {
	encoderDesc = videoDesc;
	if (encoderDesc.fps == 0) {
		encoderDesc.fps = 25;
	}
	if (encoderDesc.bitrate == 0) {
		encoderDesc.bitrate = encoderDesc.width * encoderDesc.height * encoderDesc.fps;
	}
	openEncode();
	if (!bInit)
		return;
	frame = getUniquePtr(av_frame_alloc());
	frame->format = cdeCtx->pix_fmt;
	frame->width = cdeCtx->width;
	frame->height = cdeCtx->height;

	//linesize可以用32倍数大于width,但是在这没这需要,传入时数据已经全是处理好的
	frame->linesize[0] = cdeCtx->width;
	frame->linesize[1] = cdeCtx->width / 2;
	frame->linesize[2] = cdeCtx->width / 2;
	ysize = encoderDesc.width * encoderDesc.height;
}

FH264Encoder::~FH264Encoder() {
	//av_packet_unref(&packet);
}

void FH264Encoder::openEncode() {
	//std::vector<std::string> videoEncodes = { "h264_nvenc", "libx264", "h264_qsv" };
	std::vector<std::string> videoEncodes = { "libx264","h264_nvenc","h264_qsv" };
	for (auto encode : videoEncodes) {
		int32_t ret = findEncode(encode.c_str());
		bInit = ret >= 0;
		if (bInit) {
			break;
		}
	}
}

int32_t FH264Encoder::findEncode(const char* name) {
	//获取h264的编码相应信息
	//codec = avcodec_find_encoder(AV_CODEC_ID_H264);
	auto codec = avcodec_find_encoder_by_name(name);
	if (!codec) {
		logMessage(OEIP_WARN, "could not find h264 encoder.");
		return -1;
	}
	return openEncode(codec);
}

int32_t FH264Encoder::openEncode(AVCodec* codec) {
	AVCodecContext* temp = avcodec_alloc_context3(codec);
	if (!temp) {
		return -1;
	}
	//设置h264编码属性
	cdeCtx = getUniquePtr(temp);
	//cdeCtx->codec_id = AV_CODEC_ID_H264;
	cdeCtx->codec_type = AVMEDIA_TYPE_VIDEO;
	cdeCtx->width = encoderDesc.width;
	cdeCtx->height = encoderDesc.height;
	cdeCtx->bit_rate = encoderDesc.bitrate;
	cdeCtx->rc_buffer_size = encoderDesc.bitrate;
	cdeCtx->gop_size = encoderDesc.fps * 1;
	cdeCtx->time_base = { 1,encoderDesc.fps };
	cdeCtx->delay = 0;
	// H264
	if (OEIP_H264_ENCODER_MODE == 1) {//VBR	
		cdeCtx->flags |= AV_CODEC_FLAG_PASS1;
		cdeCtx->flags |= AV_CODEC_FLAG_QSCALE;
		cdeCtx->rc_min_rate = encoderDesc.bitrate / 2;
		cdeCtx->rc_max_rate = encoderDesc.bitrate * 3 / 2;
	}
	else if (OEIP_H264_ENCODER_MODE == 2) { //QP
		cdeCtx->qmin = 21;
		cdeCtx->qmax = 26;
	}
	cdeCtx->has_b_frames = 0;
	cdeCtx->max_b_frames = 0;
	cdeCtx->me_pre_cmp = 2;
	AVDictionary* param = nullptr;
	//av_dict_set(&param, "tune", "zerolatency", 0);
	if (encoderDesc.yuvType == OEIP_YUVFMT_YUV420P) {
		temp->pix_fmt = AV_PIX_FMT_YUV420P;
		av_dict_set(&param, "profile", "high", 0);
	}
	else if (encoderDesc.yuvType == OEIP_YUVFMT_YUY2P) {
		temp->pix_fmt = AV_PIX_FMT_YUV422P;
		av_dict_set(&param, "profile", "high422", 0);
	}
	if (cdeCtx->codec_id == AV_CODEC_ID_H264) {
		av_dict_set(&param, "tune", "zerolatency", 0);
	}
	int ret = avcodec_open2(cdeCtx.get(), codec, &param);
	if (ret < 0) {
		cdeCtx.reset();
		std::string message = "open codec faild :";
		checkRet(message, ret);
	}
	std::string message = "open codec " + std::string(codec->name);
	logMessage(OEIP_INFO, message.c_str());
	return ret;
}

AVCodecContext* FH264Encoder::getCodecCtx() {
	if (!bInit)
		return nullptr;
	return cdeCtx.get();
}

int FH264Encoder::encoder(uint8_t** indata, int length, uint64_t timestamp) {
	if (!bInit)
		return -1;
	int ret = 0;
	frame->data[0] = indata[0];
	frame->data[1] = indata[1];
	frame->data[2] = indata[2];
	frame->pts = timestamp;
	frame->pkt_dts = timestamp;
	ret = avcodec_send_frame(cdeCtx.get(), frame.get());
	if (ret < 0) {
		checkRet("h264 avcodec_send_frame error", ret);
	}
	return ret;
}

int FH264Encoder::readPacket(uint8_t* outData, int& outLength, uint64_t& timestamp) {
	if (!bInit)
		return -1;
	int ret = 0;
	av_init_packet(&packet);
	ret = avcodec_receive_packet(cdeCtx.get(), &packet);
	if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
		//av_packet_unref(&packet);
		return -100;
	}
	else if (ret < 0) {
		checkRet("h264 avcodec_receive_packet error", ret);
		av_packet_unref(&packet);
		return ret;
	}
	if (outLength < packet.size) {
		//av_packet_unref(&packet);
		return -2;
	}
	memcpy(outData, packet.data, packet.size);
	outLength = packet.size;
	timestamp = packet.pts;
	av_packet_unref(&packet);
	return 0;
}
