#include "AudioOutput.h"

using namespace std::placeholders;

AudioOutput::AudioOutput() {
	micRecord = PluginManager<AudioRecord>::getInstance().createModel(0);
	loopRecord = PluginManager<AudioRecord>::getInstance().createModel(0);
}

AudioOutput::~AudioOutput() {
}

void AudioOutput::audioRecord(bool bmic, uint8_t* data, int dataLen, OeipAudioDataType type) {
	if (onAudioHandle != nullptr) {
		onAudioHandle(bmic, data, dataLen, type);
	}
	if (!bStart)
		return;
	onDataRecord(bmic, data, dataLen, type);
}

int32_t AudioOutput::start(bool bmic, bool bloopback, OeipAudioDesc destAudioDesc) {
	if (micRecord == nullptr || loopRecord == nullptr)
		return OEIP_AUDIO_NORECORD;
	if (bStart) {
		logMessage(OEIP_WARN, "audio output is runing,please stop first.");
		return OEIP_SUCESS;
	}
	this->bMic = bmic;
	this->bLoopBack = bloopback;
	this->destAudioDesc = destAudioDesc;
	if (bmic) {
		micRecord->close();
		onAudioRecordHandle handle = std::bind(&AudioOutput::audioRecord, this, true, _1, _2, _3);
		micRecord->initRecord(true, handle, micAudioDesc);
	}
	if (bloopback) {
		loopRecord->close();
		onAudioRecordHandle handle = std::bind(&AudioOutput::audioRecord, this, false, _1, _2, _3);
		loopRecord->initRecord(false, handle, loopAudioDesc);
	}
	bMixer = bmic && bloopback;
	int32_t code = onStart();
	bStart = true;// code == OEIP_SUCESS;
	return code;
}

void AudioOutput::stop() {
	if (micRecord)
		micRecord->close();
	if (loopRecord)
		loopRecord->close();
	if (bStart) {
		onStop();
		bStart = false;
	}
}
