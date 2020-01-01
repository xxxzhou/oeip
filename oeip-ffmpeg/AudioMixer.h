#pragma once
#include <RingBuffer.h>
#include "Oeipffmpeg.h"

//限定单声道,16bit,功能主要是混合声卡与麦的声音
class AudioMixer
{
public:
	AudioMixer() :AudioMixer(2) {};
	AudioMixer(int32_t size);
	~AudioMixer();
private:
	//所有输入的声音格式都要和sour一样，如果不一样，请先用AudioResample转成sour格式
	OeipAudioDesc audioDesc = {};
	std::vector<RingBuffer*> resampleBuffers;
	bool bStart = false;
	std::mutex mtx;
	std::condition_variable signal;
public:
	onAudioDataHandle onDataHandle;
public:
	void init(OeipAudioDesc desc);
	void input(int32_t index, uint8_t* data, int32_t length);
	void close();
private:
	void mixerData();
};

