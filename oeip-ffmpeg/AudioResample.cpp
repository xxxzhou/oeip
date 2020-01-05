#include "AudioResample.h"

AudioResample::AudioResample() {
}


AudioResample::~AudioResample() {
}

int32_t AudioResample::init(OeipAudioDesc sour, OeipAudioDesc dest) {
	this->sour = sour;
	this->dest = dest;
	sourBlockAlign = sour.channel * (sour.bitSize / 8);
	destBlockAlign = dest.channel * (dest.bitSize / 8);
	auto temp = swr_alloc_set_opts(nullptr, av_get_default_channel_layout(dest.channel),
		dest.bitSize == 16 ? AV_SAMPLE_FMT_S16 : AV_SAMPLE_FMT_FLT, dest.sampleRate,
		av_get_default_channel_layout(sour.channel),
		sour.bitSize == 16 ? AV_SAMPLE_FMT_S16 : AV_SAMPLE_FMT_FLT, sour.sampleRate,
		0, nullptr);
	if (!temp) {
		logMessage(OEIP_ERROR, "could not allocate resampler context!");
		return -1;
	}
	swrCtx = getUniquePtr(temp);
	int ret = swr_init(swrCtx.get());
	if (ret < 0) {
		logMessage(OEIP_ERROR, "failed to initialize the resampling context!");
		return -2;
	}
	bInit = true;
	return 0;
}

//av_get_bytes_per_sample(AV_SAMPLE_FMT_S16)
int32_t AudioResample::resampleData(const uint8_t* indata, int32_t inSize) {
	if (!bInit)
		return -1;
	//时间 毫秒
	uint64_t sm = inSize * 1000 / sour.sampleRate;
	//目标framesize,对应的概念应该是时间*采样率,所以/每个块的长度
	int32_t frameSize = inSize / sourBlockAlign;
	//目标framesize,可能不能整除，多申请一个
	//int32_t oframeSize = frameSize * sour.sampleRate / dest.sampleRate + 1;
	int32_t oframeSize = av_rescale_rnd(frameSize, dest.sampleRate, sour.sampleRate, AV_ROUND_UP);
	//目标大小
	int32_t outsize = oframeSize * destBlockAlign;

	std::vector<uint8_t> outdata;
	outdata.resize(outsize);

	uint8_t* out[1] = { outdata.data() };
	const uint8_t* in[1] = { indata };

	int osize = swr_get_out_samples(swrCtx.get(), inSize);
	int32_t ret = 0;
	ret = swr_convert(swrCtx.get(), out, osize, in, frameSize);
	//string message = " outSize:" + to_string(outsize) + "-" + to_string(osize) + " ret:" + to_string(ret);
	//logMessage(zmf_info, message.c_str());
	if (fabs(ret - oframeSize) > 1) {
		//int osize = swr_get_out_samples(swrCtx, inSize);
		std::string message = "inSize:" + std::to_string(inSize) + "-" + std::to_string(frameSize) +
			" outSize:" + std::to_string(outsize) + "-" + std::to_string(oframeSize) + " ret:" + std::to_string(ret);
		logMessage(OEIP_INFO, message.c_str());
	}
	if (ret < 0) {
		logMessage(OEIP_ERROR, "resampleData error!");
		return -1;
	}
	if (onDataHandle) {
		onDataHandle(outdata.data(), ret * destBlockAlign);
	}
	return ret;
}
