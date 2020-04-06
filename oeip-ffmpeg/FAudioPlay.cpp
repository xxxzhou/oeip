#include "FAudioPlay.h"

#ifdef __cplusplus
extern "C" {
#endif
#include <SDL.h>
#include <SDL_audio.h>
#ifdef __cplusplus
}
#endif

void audioCallback(void* userdata, uint8_t* stream, int len) {

}

FAudioPlay::FAudioPlay() {	
}

FAudioPlay::~FAudioPlay() {
	
}

bool FAudioPlay::openDevice(const OeipAudioDesc& audioDesc) {

	SDL_AudioSpec spec = {};
	SDL_memset(&spec, 0, sizeof(spec));
	spec.freq = audioDesc.sampleRate;
	spec.format = audioDesc.bitSize == 16 ? AUDIO_S16SYS : AUDIO_F32SYS;
	spec.channels = audioDesc.channel;
	spec.silence = 0;
	spec.samples = 1024;
	spec.callback = nullptr;
	spec.userdata = nullptr;
	SDL_AudioSpec obtained = {};
	//open audio devcie	
	for (int i = 0; i < SDL_GetNumAudioDrivers(); ++i) {
		SDL_Log("%i: %s", i, SDL_GetAudioDriver(i));
	}
	for (int i = 0; i < SDL_GetNumAudioDevices(0); ++i) {
		SDL_Log("%i: %s", i, SDL_GetAudioDeviceName(i, 0));
	}
	deviceId = SDL_OpenAudioDevice(nullptr, 0, &spec, nullptr, 0);//SDL_AUDIO_ALLOW_ANY_CHANGE
	if (deviceId <= 0) {
		return false;
	}
	SDL_PauseAudioDevice(deviceId, 0);
	return true;
}

void FAudioPlay::playAudioData(uint8_t* data, int32_t lenght) {
	int ret = SDL_QueueAudio(deviceId, data, lenght);
	if (SDL_GetAudioDeviceStatus(deviceId) == SDL_AUDIO_PLAYING) {
		int size = SDL_GetQueuedAudioSize(deviceId);
		SDL_PauseAudioDevice(deviceId, 0);
	}
}

void FAudioPlay::closeDevice() {
	SDL_ClearQueuedAudio(deviceId);
	SDL_CloseAudioDevice(deviceId);
}
