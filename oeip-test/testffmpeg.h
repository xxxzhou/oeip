#pragma once

#include <../oeip-ffmpeg/FH264Encoder.h>
#include <functional>
#include <future>

using namespace std;
using namespace cv;
//从解复用到解码，编码到复用

//1 音频格式对应的长度关系要对应
//avcodec_open2 相应的AVDictionary参数必填，不然时长等信息不对
namespace FFmpeg
{
	const char* src_filename = "D:\\tnmil3.flv";//2.mp4 tnmil3.flv
	const char* dest_filename = "D:\\tnmil3d.flv";//D:\\tnmil3d.flv
	const char* video_dest_filename = "D:\\tnmi13v.yuv";
	const char* audio_dest_filename = "D:\\tnmil3a.wav";
	//封装视频格式
	AVFormatContext* fmt_ctx = nullptr;
	AVCodecContext* videoCodexCtx = nullptr;
	AVCodecContext* audioCodexCtx = nullptr;
	int width = 784, height = 480;
	enum AVPixelFormat pixFmt = AV_PIX_FMT_YUV420P;
	AVStream* videoStream = nullptr, * audioStream = nullptr;
	int videoStreamIdx = -1, audioStreamIdx = -1;
	AVFrame* frame = nullptr;
	AVPacket pkt;
	int videoFrameCount = 0;
	int audioFrameCount = 0;
	int refCount = 0;
	FILE* videoDestFile = nullptr;
	FILE* audioDestFile = nullptr;
	uint8_t* videoDstData[4] = { nullptr };
	int videoDstLineSize[4] = { 784,392,392,0 };
	int videoDstBufSize = 564480;
	size_t unpadded_linesize = 4096;
	const char* videoCodecName = "h264";
	const char* audioCodecName = "";
	int videoTime = 0;
	int audioTime = 0;

	int decode_packet() {
		int ret = 0;
		int decoded = pkt.size;
		//*got_frame = 0;
		if (pkt.stream_index == videoStreamIdx) {
			//ret = avcodec_decode_video2(videoCodexCtx, frame, got_frame, &pkt);
			ret = avcodec_send_packet(videoCodexCtx, &pkt);
			if (ret < 0) {
				return ret;
			}
			while (ret >= 0) {
				ret = avcodec_receive_frame(videoCodexCtx, frame);
				//EAGAIN表明需要更多packet,AVERROR_EOF表明读到文件结尾
				if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
					break;
				}
				else if (ret < 0) {
					return ret;
				}
				av_image_copy(videoDstData, videoDstLineSize, (const uint8_t**)(frame->data),
					frame->linesize, pixFmt, width, height);
				fwrite(videoDstData[0], 1, videoDstBufSize, videoDestFile);
				videoTime = frame->pts;
			}
		}
		else if (pkt.stream_index == audioStreamIdx) {
			ret = avcodec_send_packet(audioCodexCtx, &pkt);
			if (ret < 0) {
				return ret;
			}
			while (ret >= 0) {
				ret = avcodec_receive_frame(audioCodexCtx, frame);
				//EAGAIN表明需要更多packet,AVERROR_EOF表明读到文件结尾
				if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
					break;
				}
				else if (ret < 0) {
					return ret;
				}
				unpadded_linesize = frame->nb_samples * av_get_bytes_per_sample((AVSampleFormat)frame->format);
				fwrite(frame->extended_data[0], 1, unpadded_linesize, audioDestFile);
				fwrite(frame->extended_data[1], 1, unpadded_linesize, audioDestFile);
				audioTime = frame->pts;
			}
		}
		return 0;
	}

	int open_codec_context(AVFormatContext* fmtCtx, int* streamIdx, AVCodecContext** decCtx, enum AVMediaType type) {
		int ret = 0;
		int streamIndex = 0;
		AVStream* st = nullptr;
		AVCodec* codec = nullptr;
		AVDictionary* opts = nullptr;
		auto mediaTypeStr = av_get_media_type_string(type);
		//选择文件里最优的那条视频或音频流
		ret = av_find_best_stream(fmtCtx, type, -1, -1, nullptr, 0);
		if (ret < 0) {
			stringstream sstream;
			sstream << mediaTypeStr << " not find stream.";
			logMessage(OEIP_ALORT, sstream.str().c_str());
			return ret;
		}
		streamIndex = ret;
		st = fmtCtx->streams[streamIndex];
		//找到这个流的解码器
		codec = avcodec_find_decoder(st->codecpar->codec_id);
		if (!codec) {
			stringstream sstream;
			sstream << mediaTypeStr << " not find decodec.";
			logMessage(OEIP_ALORT, sstream.str().c_str());
			return AVERROR(EINVAL);
		}
		//创建输出文件的编解码器上下文
		*decCtx = avcodec_alloc_context3(codec);
		if (!*decCtx) {
			stringstream sstream;
			sstream << mediaTypeStr << " not alloc contex.";
			logMessage(OEIP_ALORT, sstream.str().c_str());
			return AVERROR(ENOMEM);
		}
		//把当前输入流的编解码器参数复制给输出编解码器
		//音频流信息参数在st->codecpar中
		//比如视频,视频的关键参数有format, width, height, codec_type等
		if ((ret = avcodec_parameters_to_context(*decCtx, st->codecpar)) < 0) {
			return ret;
		}
		st->codecpar->codec_tag = 0;
		av_dict_set(&opts, "refcounted_frames", refCount ? "1" : "0", 0);
		//打开编解码器
		if ((ret == avcodec_open2(*decCtx, codec, &opts)) < 0) {
			return ret;
		}
		*streamIdx = streamIndex;
		return 0;
	}

	void testDemux() {
		int ret = 0;
		int got_frame;
		//根据文件填充封装格式		
		if (avformat_open_input(&fmt_ctx, src_filename, nullptr, nullptr) < 0) {
			logMessage(OEIP_ALORT, "not open input file.");
			return;
		}
		//查找这份文件封装格式是否包含流信息
		if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
			logMessage(OEIP_ALORT, "not find stream");
			return;
		}
		//查找视频流，音频流
		if (open_codec_context(fmt_ctx, &videoStreamIdx, &videoCodexCtx, AVMEDIA_TYPE_VIDEO) >= 0) {
			videoStream = fmt_ctx->streams[videoStreamIdx];
			videoDestFile = fopen(video_dest_filename, "wb");
			if (!videoDestFile) {
				logMessage(OEIP_ALORT, "not open output video file.");
				return;
			}
			width = videoCodexCtx->width;
			height = videoCodexCtx->height;
			pixFmt = videoCodexCtx->pix_fmt;
			ret = av_image_alloc(videoDstData, videoDstLineSize, width, height, pixFmt, 1);
			if (ret < 0) {

			}
			videoDstBufSize = ret;
		}
		if (open_codec_context(fmt_ctx, &audioStreamIdx, &audioCodexCtx, AVMEDIA_TYPE_AUDIO) >= 0) {
			audioStream = fmt_ctx->streams[audioStreamIdx];
			audioDestFile = fopen(audio_dest_filename, "wb");
			if (!audioDestFile) {
				return;
			}
		}
		av_dump_format(fmt_ctx, 0, src_filename, 0);
		if (!audioStream && !videoStream) {
			return;
		}
		//申请frame
		frame = av_frame_alloc();
		if (!frame) {
			ret = AVERROR(ENOMEM);
			return;
		}
		//初始化packet
		av_init_packet(&pkt);
		pkt.data = nullptr;
		pkt.size = 0;
		//解复用
		while (av_read_frame(fmt_ctx, &pkt) >= 0) {
			decode_packet();
		}
		avcodec_free_context(&videoCodexCtx);
		avcodec_free_context(&audioCodexCtx);
		avformat_close_input(&fmt_ctx);
		if (videoDestFile)
			fclose(videoDestFile);
		if (audioDestFile)
			fclose(audioDestFile);
		av_frame_free(&frame);
		av_free(videoDstData[0]);
	}

	int add_codec_context(AVFormatContext* fmtCtx, int* streamIdx, AVCodecContext** decCtx, enum AVMediaType type) {
		int ret = 0;
		int streamIndex = 0;
		AVStream* st = nullptr;
		AVCodec* codec = nullptr;
		AVCodecID cid = type == AVMEDIA_TYPE_VIDEO ? AV_CODEC_ID_H264 : AV_CODEC_ID_AAC;
		//找到这个流的编码器		
		codec = avcodec_find_encoder(cid);// avcodec_find_encoder_by_name(videoCodecName);		
		if (!codec) {
			return AVERROR(ENOMEM);
		}
		//创建输出文件的编解码器上下文
		*decCtx = avcodec_alloc_context3(codec);
		if (!*decCtx) {
			return AVERROR(ENOMEM);
		}
		if (type == AVMEDIA_TYPE_VIDEO) {
			(*decCtx)->bit_rate = 2000000;
			//(*decCtx)->rc_buffer_size = 200000;
			(*decCtx)->width = width;
			(*decCtx)->height = height;
			(*decCtx)->framerate = av_make_q(25, 1);
			(*decCtx)->time_base = av_make_q(1, 25);
			(*decCtx)->gop_size = 25;
			(*decCtx)->me_pre_cmp = 2;
			(*decCtx)->has_b_frames = 0;
			(*decCtx)->max_b_frames = 0;
			(*decCtx)->flags |= AV_CODEC_FLAG_PASS1;
			(*decCtx)->flags |= AV_CODEC_FLAG_QSCALE;
			(*decCtx)->rc_min_rate = (*decCtx)->bit_rate / 2;
			(*decCtx)->rc_max_rate = (*decCtx)->bit_rate * 3 / 2;
			(*decCtx)->pix_fmt = AV_PIX_FMT_YUV420P;
			//(*decCtx)->delay = 0;
		}
		else if (type == AVMEDIA_TYPE_AUDIO) {
			(*decCtx)->profile = FF_PROFILE_AAC_LOW;
			(*decCtx)->codec_type = AVMEDIA_TYPE_AUDIO;
			(*decCtx)->sample_rate = 44100;
			(*decCtx)->time_base = av_make_q(1, (*decCtx)->sample_rate);
			(*decCtx)->bit_rate = 128000;
			(*decCtx)->sample_fmt = AV_SAMPLE_FMT_FLTP;
			/* select other audio parameters supported by the encoder */
			(*decCtx)->channel_layout = 3;// AV_CH_LAYOUT_STEREO
			(*decCtx)->channels = av_get_channel_layout_nb_channels((*decCtx)->channel_layout);
		}
		st = avformat_new_stream(fmtCtx, codec);
		if (!st) {
			return AVERROR(ENOMEM);
		}
		avcodec_parameters_from_context(st->codecpar, *decCtx);
		AVDictionary* param = nullptr;
		if (type == AVMEDIA_TYPE_VIDEO) {
			st->time_base = av_make_q(1, 1000);
			st->r_frame_rate = av_make_q(25, 1);			
			av_dict_set(&param, "profile", "high", 0);
			av_dict_set(&param, "tune", "zerolatency", 0);
		}
		else if (type == AVMEDIA_TYPE_AUDIO) {

		}
		*streamIdx = st->index;
		if ((ret == avcodec_open2(*decCtx, codec, &param)) < 0) {
			return ret;
		}
		return 0;
	}

	int readVideo() {
		int ret = 0;
		std::vector<uint8_t> vdata;
		vdata.resize(videoDstBufSize);
		int ysize = width * height;
		//申请frame
		auto* frame = av_frame_alloc();
		if (!frame) {
			ret = AVERROR(ENOMEM);
			return ret;
		}
		frame->width = width;
		frame->height = height;
		frame->linesize[0] = width;
		frame->linesize[1] = width / 2;
		frame->linesize[2] = width / 2;
		frame->format = pixFmt;
		int tsize = 0;
		int i = 1;
		while (true) {
			fseek(videoDestFile, tsize, 0);
			tsize += videoDstBufSize;
			int read = fread(vdata.data(), 1, videoDstBufSize, videoDestFile);
			if (read < videoDstBufSize)
				break;
			frame->data[0] = vdata.data();
			frame->data[1] = vdata.data() + ysize;
			frame->data[2] = vdata.data() + ysize * 5 / 4;
			videoTime = (i++) * 40;
			frame->pts = videoTime;
			frame->pkt_dts = frame->pts;
			frame->pkt_duration = (int64_t)(1000 / av_q2d(videoCodexCtx->framerate));
			ret = avcodec_send_frame(videoCodexCtx, frame);
			if (ret < 0) {
				return ret;
			}
			while (true) {
				AVPacket packet;
				av_init_packet(&packet);
				ret = avcodec_receive_packet(videoCodexCtx, &packet);
				//packet.duration = (int64_t)(1000 / av_q2d(videoCodexCtx->framerate));
				//av_packet_rescale_ts(packet,)
				if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
					break;
				}
				else if (ret < 0) {
					return ret;
				}
				packet.stream_index = videoStreamIdx;
				ret = av_interleaved_write_frame(fmt_ctx, &packet);
				av_packet_unref(&pkt);
			}
			//if (videoTime - audioTime > 40) {
			//	std::this_thread::sleep_for(std::chrono::milliseconds(1));
			//}
		}
		av_frame_free(&frame);
		return ret;
	}

	int readAudio() {
		int ret = 0;
		std::vector<uint8_t> vdata;
		vdata.resize(unpadded_linesize * 2);
		//申请frame
		auto* frame = av_frame_alloc();
		if (!frame) {
			ret = AVERROR(ENOMEM);
			return ret;
		}
		frame->nb_samples = audioCodexCtx->frame_size;
		frame->format = audioCodexCtx->sample_fmt;
		frame->channel_layout = audioCodexCtx->channel_layout;
		frame->channels = audioCodexCtx->channels;
		frame->sample_rate = audioCodexCtx->sample_rate;
		int tsize = 0;
		int i = 0;
		while (true) {
			fseek(audioDestFile, tsize, 0);
			tsize += unpadded_linesize * 2;
			int read = fread(vdata.data(), 1, unpadded_linesize * 2, audioDestFile);
			if (read < unpadded_linesize * 2)
				break;
			//av_frame_get_buffer(frame, 0);
			//每个通道的大小
			frame->linesize[0] = audioCodexCtx->frame_size * 2;
			frame->data[0] = vdata.data();
			frame->data[1] = vdata.data() + unpadded_linesize;
			audioTime = 23 * (tsize / (unpadded_linesize * 2));
			frame->pts = audioTime;
			frame->pkt_dts = frame->pts;
			ret = avcodec_send_frame(audioCodexCtx, frame);
			if (ret < 0) {
				checkRet("error avcodec_send_frame.", ret);
				return ret;
			}
			while (true) {
				AVPacket packet;
				av_init_packet(&packet);
				ret = avcodec_receive_packet(audioCodexCtx, &packet);
				if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
					break;
				}
				else if (ret < 0) {
					return ret;
				}
				if (packet.pts < 0) {
					packet.pts = 0;
					packet.dts = 0;
				}
				//这个一定要注明清楚
				packet.stream_index = audioStreamIdx;
				ret = av_interleaved_write_frame(fmt_ctx, &packet);
				av_packet_unref(&pkt);
			}
			//if (audioTime - videoTime > 40) {
			//	std::this_thread::sleep_for(std::chrono::milliseconds(1));
			//}
		}
		vdata.clear();
		frame->data[0] = nullptr;
		frame->data[1] = nullptr;
		av_frame_free(&frame);
		return ret;
	}

	void testMux() {
		int ret = 0;
		//根据文件填充封装格式
		if (avformat_alloc_output_context2(&fmt_ctx, nullptr, "flv", dest_filename) < 0) {
			logMessage(OEIP_ALORT, "not open input file.");
			return;
		}
		if (add_codec_context(fmt_ctx, &videoStreamIdx, &videoCodexCtx, AVMEDIA_TYPE_VIDEO) >= 0) {
			videoDestFile = fopen(video_dest_filename, "rb");
		}
		if (add_codec_context(fmt_ctx, &audioStreamIdx, &audioCodexCtx, AVMEDIA_TYPE_AUDIO) >= 0) {
			audioDestFile = fopen(audio_dest_filename, "rb");
		}
		av_dump_format(fmt_ctx, 0, dest_filename, 1);
		//如果是文件，需要这步
		if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
			ret = avio_open(&fmt_ctx->pb, dest_filename, AVIO_FLAG_WRITE);
			if (ret < 0) {
				return;
			}
		}
		ret = avformat_write_header(fmt_ctx, nullptr);
		if (ret < 0) {
			return;
		}
		std::future<int> rvideo = std::async([&]() {
			//读取视频
			return readVideo();
		});
		std::future<int> raudio = std::async([&]() {
			//读取视频
			return readAudio();
		});
		//读取视频
		//readAudio();
		int vresult = rvideo.get();
		//int aresult = raudio.get();
		//写入结尾
		ret = av_write_trailer(fmt_ctx);
		//关闭资源
		avcodec_free_context(&videoCodexCtx);
		avcodec_free_context(&audioCodexCtx);
		avformat_close_input(&fmt_ctx);
		if (videoDestFile)
			fclose(videoDestFile);
		if (audioDestFile)
			fclose(audioDestFile);
		av_frame_free(&frame);
		av_free(videoDstData[0]);
	}
}
