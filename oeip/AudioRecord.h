#pragma once
#include "Oeip.h"
#include "PluginManager.h"
#include <vector>

//只用来写formatTag为固定的WAVE_FORMAT_PCM结构 
struct WaveFormat
{
	//写文件,在这固定为PCM格式
	uint16_t formatTag = 1;
	uint16_t channel = 1;
	uint32_t sampleRate = 44100;
	uint32_t sampleBytes = 88200;
	uint16_t blockAlign = 2;
	uint16_t bitSize = 16;//16U
	uint16_t cbSize = 0;
};

struct WAVEHEADER
{
	uint32_t dwRiff;                     // "RIFF"
	uint32_t dwSize;                     // Size
	uint32_t dwWave;                     // "WAVE"
	uint32_t dwFmt;                      // "fmt "
	uint32_t dwFmtSize;                  // Wave Format Size
};

//Static RIFF header, we'll append the format to it.
const uint8_t WaveHeader[] =
{
	'R','I','F','F',0x00,0x00,0x00,0x00,'W','A','V','E','f','m','t',' ',0x00,0x00,0x00,0x00
};

const uint8_t WaveData[] = { 'd', 'a', 't', 'a' };

class OEIPDLL_EXPORT AudioRecord
{
public:
	AudioRecord();
	virtual ~AudioRecord();
protected:
	bool bMic = false;
	bool bStart = false;
	onAudioRecordHandle onRecordHandle;
	OeipAudioDesc audioDesc;
	//音频设备事件
	onEventAction onDeviceHandle;
protected:
	virtual bool initRecord() { return false; };
public:
	virtual void close() {};
public:
	bool initRecord(bool bMic, onAudioRecordHandle handle, OeipAudioDesc& audioDesc);
};

//针对AudioDesc的bitsize为16，并且内部格式为AV_SAMPLE_FMT_S16的保证无问题，P格式肯定不行
OEIPDLL_EXPORT void getWavHeader(std::vector<uint8_t>& header, uint32_t dataSize, OeipAudioDesc audioDesc);

OEIP_DEFINE_PLUGIN_TYPE(AudioRecord);




