#include "FLiveOutput.h"
#include "FAACEncoder.h"
#include "FH264Encoder.h"
#include "FRtmpOutput.h"

FLiveOutput::FLiveOutput() {
	output = std::unique_ptr<FRtmpOutput>(new FRtmpOutput());
}

FLiveOutput::~FLiveOutput() {
	close();
}

int32_t FLiveOutput::open(const char* url) {
	std::unique_lock<std::mutex> lck(mtx);
	int32_t ret = 0;
	if (bOpen)
		return 1;
	ret = output->openURL(url, bVideo, bAudio);
	if (ret == 0)
		bOpen = true;
	liveUrl = url;
	return 0;
}

void FLiveOutput::close() {
	std::unique_lock <std::mutex> lck(mtx);
	if (!bOpen)
		return;
	bOpen = false;
	if (output) {
		output->close();
	}
}

int32_t FLiveOutput::pushVideo(const OeipVideoFrame& videoFrame) {
	std::unique_lock <std::mutex> lck(mtx);
	if (!bOpen)
		return -1;
	if (videoEncoder && (videoFrame.width != videoWidth || videoFrame.height != videoHeight)) {
		//重新开始推流
		output->close();
		output->openURL(liveUrl.c_str(), bVideo, bAudio);
		logMessage(OEIP_INFO, "video resolution changes, reinitialize push flow");
	}
	if (!videoEncoder) {
		OeipVideoEncoder videoDesc = {};
		videoDesc.bitrate = videoBitrate;
		videoDesc.fps = fps;
		videoDesc.height = videoFrame.height;
		videoDesc.width = videoFrame.width;
		videoDesc.yuvType = videoFrame.fmt;
		videoEncoder = std::unique_ptr<FH264Encoder>(new FH264Encoder(videoDesc));
		videoBuffer.resize(OEIP_H264_BUFFER_MAX_SIZE);
		videoWidth = videoFrame.width;
		videoHeight = videoFrame.height;
	}
	int32_t ret = videoEncoder->encoder((uint8_t**)videoFrame.data, videoFrame.dataSize, videoFrame.timestamp);
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
		ret = output->pushVideo(videoBuffer.data(), outLen, timestmap);
		if (ret < 0) {
			output->close();
			output->openURL(liveUrl.c_str(), bVideo, bAudio);
		}
	}
	return 0;
}

int32_t FLiveOutput::pushAudio(const OeipAudioFrame& audioFrame) {
	std::unique_lock <std::mutex> lck(mtx);
	if (!bOpen)
		return -1;
	if (!audioEncoder) {
		OeipAudioEncoder audioDesc = {};
		audioDesc.bitrate = audioBitrate;
		audioDesc.channel = audioFrame.channels;
		audioDesc.frequency = audioFrame.sampleRate;

		audioEncoder = std::unique_ptr<FAACEncoder>(new FAACEncoder(audioDesc));
		audioBuffer.resize(OEIP_AAC_BUFFER_MAX_SIZE);
	}	
	int32_t ret = audioEncoder->encoder((uint8_t**)&audioFrame.data, audioFrame.dataSize, audioFrame.timestamp);
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
		ret = output->pushAudio(audioBuffer.data(), outLen, timestmap);
		if (ret < 0) {
			output->close();
			output->openURL(liveUrl.c_str(), bVideo, bAudio);
		}
	}
	return 0;
}

void FLiveOutput::setVideoBitrate(int32_t bitrate) {
	videoBitrate = bitrate;
}

void FLiveOutput::setAudioBitrate(int32_t bitrate) {
	audioBitrate = bitrate;
}

void FLiveOutput::setFps(int32_t fps) {
	this->fps = fps;
}

void FLiveOutput::enableVideo(bool bVideo) {
	this->bVideo = bVideo;
}

void FLiveOutput::enableAudio(bool bAudio) {
	this->bAudio = bAudio;
}

void FLiveOutput::setOperateEvent(onOperateHandle onHandle) {
	output->setOperateEvent(onHandle);
}

