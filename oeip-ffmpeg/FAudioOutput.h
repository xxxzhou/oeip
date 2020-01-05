#pragma once
#include "AudioMixer.h"
#include "AudioResample.h"
#include <AudioOutput.h>
#include <memory>

//用来记录麦与声卡以特定格式输出，有必要的话会混合麦与声卡输出
class FAudioOutput : public AudioOutput
{
public:
	FAudioOutput();
	~FAudioOutput();
private:
	bool bReMic = false;
	bool bReLoopBack = false;
	std::unique_ptr<AudioResample> micResample = nullptr;
	std::unique_ptr<AudioResample> loopResample = nullptr;
	std::unique_ptr<AudioMixer> mixer = nullptr;
protected:
	virtual int32_t onStart() override;
	virtual int32_t onStop() override;
	virtual int32_t onDataRecord(bool bmic, uint8_t* data, int dataLen, OeipAudioDataType type) override;
private:
	void onResampleData(bool bmic, uint8_t* data, int32_t lenght);
	void onMixData(uint8_t* data, int32_t lenght);;
};

OEIP_DEFINE_PLUGIN_CLASS(AudioOutput, FAudioOutput)

extern "C" __declspec(dllexport) bool bCanLoad();
extern "C" __declspec(dllexport) void registerFactory();