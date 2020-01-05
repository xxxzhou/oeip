#include "AudioMixer.h"

#define  MIXER_USER_MAX    50

#define MIN(m, n) ((m < n)?(m):(n))
#define MAX(m, n) ((m > n)?(m):(n))
#define ABS(n) ((n < 0)?(-n):(n))
#define Q 16
#define MAXS16 (0x7FFF)
#define SGN(n) ((n < 0) ? (-1) : (1))

#define ITEMS 5

typedef struct
{
	int Rsh;
	int K;
} TABLE;


const static TABLE table[ITEMS] = {// N=8
{ 3, 0 },
{ 6, 28672 },
{ 9, 32256 },
{ 12, 32704 },
{ 15, 32760 }
};

static inline int16_t do_mix(long v) {
	long total = v;
	long n, c, d;
	n = MIN((ABS(total) >> (Q - 1)), ITEMS - 1);
	c = ABS(total) & MAXS16;
	d = (c << 2) + (c << 1) + c;
	return SGN(total) * ((d >> table[n].Rsh) + table[n].K);
}

AudioMixer::AudioMixer(int32_t size) {
	resampleBuffers.resize(size);
	for (int32_t i = 0; i < size; i++) {
		resampleBuffers[i] = new RingBuffer(100 * 1024);
	}
}

AudioMixer::~AudioMixer() {
	clearList(resampleBuffers);
}

void AudioMixer::init(OeipAudioDesc desc) {
	audioDesc = desc;
}

void AudioMixer::input(int32_t index, uint8_t* data, int32_t length) {
	std::unique_lock <std::mutex> lck(mtx);
	if (index < 0 && index >= resampleBuffers.size()) {
		logMessage(OEIP_WARN, "audiomixxer input index invalid.");
	}
	resampleBuffers[index]->push(data, length);
	if (!bStart) {
		bStart = true;
		std::thread mixthr = std::thread(std::bind(&AudioMixer::mixerData, this));
		mixthr.detach();
	}
}

void AudioMixer::mixerData() {
	//int32_t blockAlign = audioDesc.channel * audioDesc.bitSize / 8;
	int32_t sampleTime = 40;
	int sampleSize = audioDesc.sampleRate * audioDesc.channel * sampleTime / 1000;
	int frameSize = sampleSize * audioDesc.bitSize / 8;//这里功能暂限定bitsize只能是16
	int16_t* pcmbuf = new int16_t[sampleSize];
	long total = 0;
	while (bStart) {
		std::this_thread::sleep_for(std::chrono::milliseconds(sampleTime));
		std::vector<std::vector<int16_t>> vecBuffers;
		for (auto& ring : resampleBuffers) {
			std::vector<int16_t> temp(sampleSize, 0);
			if (ring->pull((uint8_t*)temp.data(), frameSize) >= 0) {
				vecBuffers.push_back(temp);
			}
		}
		if (vecBuffers.empty()) {
			continue;
		}
		int32_t size = vecBuffers.size();
		for (int32_t i = 0; i < sampleSize; i++) {
			total = 0;
			for (int32_t j = 0; j < size; j++) {
				total += vecBuffers[j][i];
			}
			pcmbuf[i] = do_mix(total);
		}
		if (onDataHandle) {
			onDataHandle((uint8_t*)pcmbuf, frameSize);
		}
	}
	signal.notify_all();
}

void AudioMixer::close() {
	std::unique_lock <std::mutex> lck(mtx);
	if (bStart) {
		bStart = false;
		auto status = signal.wait_for(lck, std::chrono::seconds(2));
		if (status == std::cv_status::timeout) {
			logMessage(OEIP_WARN, "audio mixer is not closed properly.");
		}
		for (int32_t i = 0; i < resampleBuffers.size(); i++) {
			resampleBuffers[i]->clear();
		}
		onDataHandle = nullptr;
	}
}
