#pragma once
#include "Oeip.h"
#include "PluginManager.h"
#include <memory>
#include "AudioRecord.h"

class OEIPDLL_EXPORT AudioOutput
{
public:
	AudioOutput();
	virtual ~AudioOutput();
protected:
	bool bStart = false;
	bool bMic = false;
	bool bMixer = false;
	bool bLoopBack = false;
	OeipAudioDesc micAudioDesc = {};
	OeipAudioDesc loopAudioDesc = {};
	OeipAudioDesc destAudioDesc = {};
	AudioRecord* micRecord = nullptr;
	AudioRecord* loopRecord = nullptr;
public:
	//经过处理的数据
	onAudioDataHandle onDataHandle;
	//原始声卡或麦的数据
	onAudioOutputHandle onAudioHandle;
private:
	void audioRecord(bool bmic, uint8_t* data, int dataLen, OeipAudioDataType type);
protected:
	virtual int32_t onStart() { return 0; };
	virtual int32_t onStop() { return 0; };
	virtual int32_t onDataRecord(bool bmic, uint8_t* data, int dataLen, OeipAudioDataType type) { return 0; };
public:
	int32_t start(bool bmic, bool bloopback, OeipAudioDesc destAudioDesc);
	void stop();
};

OEIP_DEFINE_PLUGIN_TYPE(AudioOutput);

