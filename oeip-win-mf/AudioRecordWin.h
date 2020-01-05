#pragma once
#include <AudioRecord.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <thread>
#include <mutex>
#include <PluginManager.h>
#include <atlcomcli.h>

// ns(nanosecond) : 纳秒，时间单位。一秒的十亿分之一
// 1秒=1000毫秒; 1毫秒=1000微秒; 1微秒=1000纳秒
// The REFERENCE_TIME data type defines the units for reference times in DirectShow. 
// Each unit of reference time is 100 nanoseconds.(100纳秒为一个REFERENCE_TIME时间单位)
// REFERENCE_TIME time units per second and per millisecond

#define REFTIMES_PER_SEC       (10000000)
#define REFTIMES_PER_MILLISEC  (10000)
#define EXIT_ERROR_BOOL(hr)  \
    if (FAILED(hr)) { return false; }
#define EXIT_ERROR_GOTO(hr)  \
    if (FAILED(hr)) { goto Exit; }

class AudioRecordWin : public AudioRecord
{
public:
	AudioRecordWin();
	virtual ~AudioRecordWin();
private:
	CComPtr<IAudioClient> client = nullptr;
	CComPtr<IAudioCaptureClient> capture = nullptr;
	CComPtr<IAudioRenderClient> render = nullptr;
	WAVEFORMATEX* format = nullptr;
	HANDLE hAudioSamplesReadyEvent = nullptr;
	std::mutex mtx;
	std::condition_variable signal;
protected:
	virtual bool initRecord() override;
private:
	void setWaveFormat(WAVEFORMATEX& format);
	void recordAudio();
	void getWavHeader(std::vector<uint8_t>& header, uint32_t dataSize);
	void internalClose();
public:
	virtual void close();
};

OEIP_DEFINE_PLUGIN_CLASS(AudioRecord, AudioRecordWin)


