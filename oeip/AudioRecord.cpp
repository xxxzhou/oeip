#include "AudioRecord.h"
#include "OeipCommon.h"

AudioRecord::AudioRecord() {
}


AudioRecord::~AudioRecord() {

}

bool AudioRecord::initRecord(bool bMic, onAudioRecordHandle handle, OeipAudioDesc & audioDesc) {
	if (bStart) {
		logMessage(OEIP_WARN, "audio recording...");
		return false;
	}
	this->bMic = bMic;
	onRecordHandle = handle;
	if (initRecord()) {
		audioDesc = this->audioDesc;
		return true;
	}
	return false;
}

void getWavHeader(std::vector<uint8_t>& header, uint32_t dataSize, OeipAudioDesc audioDesc) {
	uint32_t headerSize = sizeof(WAVEHEADER) + sizeof(WaveFormat) + sizeof(WaveData) + sizeof(uint32_t);
	header.resize(headerSize, 0);

	WaveFormat format = {};
	format.channel = audioDesc.channel;
	format.sampleRate = audioDesc.sampleRate;
	format.bitSize = audioDesc.bitSize;
	format.blockAlign = format.channel * format.bitSize / 8;
	format.sampleBytes = format.sampleRate * format.blockAlign;

	uint8_t* writeIdx = header.data();
	//先写声音基本信息
	WAVEHEADER *waveHeader = reinterpret_cast<WAVEHEADER *>(writeIdx);
	memcpy(writeIdx, WaveHeader, sizeof(WAVEHEADER));
	writeIdx += sizeof(WAVEHEADER);
	waveHeader->dwSize = headerSize + dataSize - 2 * 4;
	waveHeader->dwFmtSize = sizeof(WaveFormat);
	//format后面可能会带cbSize个额外信息
	memcpy(writeIdx, &format, sizeof(WaveFormat));
	writeIdx += sizeof(WaveFormat);
	//写入data
	memcpy(writeIdx, WaveData, sizeof(WaveData));
	writeIdx += sizeof(WaveData);
	//写入一个结尾数据长度
	*(reinterpret_cast<uint32_t *>(writeIdx)) = static_cast<uint32_t>(dataSize);
	writeIdx += sizeof(uint32_t);
}
