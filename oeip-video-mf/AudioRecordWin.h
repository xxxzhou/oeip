#pragma once
#include <AudioRecord.h>
#include <AudioFile.h>
#include <mmdeviceapi.h>
#include <thread>
#include <audioclient.h>
#include <mutex>
#include <PluginManager.h>

class AudioRecordWin :
	public AudioRecord
{
public:
	AudioRecordWin();
	virtual ~AudioRecordWin(); 
private:
	AudioFile * audioFile;
	onAudioRecordHandle onAudioRecordFunc;
	bool bStart = false;
	IAudioClient* client = nullptr;
	IAudioCaptureClient* capture = nullptr;
	WAVEFORMATEX* format = nullptr;
	REFERENCE_TIME waitTime;
	std::thread recordThread;
	std::mutex mtx;
	std::condition_variable signal;
	std::wstring filePath;
	OeipAudioRecordType recordType = OEIP_Mic;
	bool bMic = false;
	bool bRecordHandle = false;
public:
	virtual bool initRecord();
private:
	void setWaveFormat(WAVEFORMATEX& format);
	void recordAudio();
public:
	virtual bool initRecord(OeipAudioRecordType audiotype, onAudioRecordHandle handle);
	virtual bool initRecord(OeipAudioRecordType audiotype, std::wstring path);
	virtual void close();
};

class MFAudioRecordFactory :public ObjectFactory<AudioRecord>
{
public:
	MFAudioRecordFactory() {};
	~MFAudioRecordFactory() {};
public:
	virtual AudioRecord* create(int type) override;
};

