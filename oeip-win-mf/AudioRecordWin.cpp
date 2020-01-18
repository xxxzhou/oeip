#include "AudioRecordWin.h"
#include <OeipCommon.h>
#include "../oeip-win/DxHelper.h"

#define BUFFER_TIME_100NS (5 * 10000000)
#define RECONNECT_INTERVAL 400

AudioRecordWin::AudioRecordWin() {
}

AudioRecordWin::~AudioRecordWin() {
	CoTaskMemFree(format);
	close();
}

bool AudioRecordWin::initRecord() {
	CComPtr<IMMDeviceEnumerator> enumerator;
	HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), reinterpret_cast<void**>(&enumerator));
	CComPtr<IMMDevice> device;
	//hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
	hr = enumerator->GetDefaultAudioEndpoint(bMic ? eCapture : eRender,
		bMic ? eCommunications : eConsole, &device);//bMic ? eCommunications : eConsole
	EXIT_ERROR_BOOL(hr);
	hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, reinterpret_cast<void**>(&client));
	EXIT_ERROR_BOOL(hr);
	hr = client->GetMixFormat(&format);
	EXIT_ERROR_BOOL(hr);
	REFERENCE_TIME waitTime;
	hr = client->GetDevicePeriod(&waitTime, NULL);
	//setWaveFormat(*format);
	DWORD flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK;
	if (!bMic) {
		flags |= AUDCLNT_STREAMFLAGS_LOOPBACK;
	}
	else {
		flags |= AUDCLNT_STREAMFLAGS_NOPERSIST;
	}
	hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, BUFFER_TIME_100NS, 0, format, nullptr);
	EXIT_ERROR_BOOL(hr);
	if (!bMic) {
		UINT32 frames;
		LPBYTE buffer;
		WAVEFORMATEX* wformat;
		CComPtr<IAudioClient> client;
		hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&client);
		EXIT_ERROR_BOOL(hr);
		hr = client->GetMixFormat(&wformat);
		EXIT_ERROR_BOOL(hr);
		hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, BUFFER_TIME_100NS, 0, wformat, nullptr);
		EXIT_ERROR_BOOL(hr);
		hr = client->GetBufferSize(&frames);
		EXIT_ERROR_BOOL(hr);
		hr = client->GetService(__uuidof(IAudioRenderClient), (void**)&render);
		EXIT_ERROR_BOOL(hr);
		hr = render->GetBuffer(frames, &buffer);
		EXIT_ERROR_BOOL(hr);
		memset(buffer, 0, frames * wformat->nBlockAlign);
		render->ReleaseBuffer(frames, 0);
		CoTaskMemFree(wformat);
	}
	hr = client->GetService(__uuidof(IAudioCaptureClient), (void**)(&capture));
	EXIT_ERROR_BOOL(hr);
	hAudioSamplesReadyEvent = CreateEventEx(NULL, NULL, 0, EVENT_MODIFY_STATE | SYNCHRONIZE);
	if (hAudioSamplesReadyEvent == nullptr) {
		std::string msg = "audio redcord CreateEventEx fail:" + GetLastError();
		logMessage(OEIP_ERROR, msg.c_str());
	}
	hr = client->SetEventHandle(hAudioSamplesReadyEvent);
	EXIT_ERROR_BOOL(hr);
	audioDesc.channel = format->nChannels;
	audioDesc.bitSize = format->wBitsPerSample;
	audioDesc.sampleRate = format->nSamplesPerSec;
	hr = client->Start();
	bStart = true;
	std::thread recordThread = std::thread(std::bind(&AudioRecordWin::recordAudio, this));
	recordThread.detach();
	return SUCCEEDED(hr);
}

void AudioRecordWin::setWaveFormat(WAVEFORMATEX& format) {
	//转成16通道输出，32位完全没必要 
	if (format.wFormatTag == WAVE_FORMAT_IEEE_FLOAT) {
		//format.wFormatTag = WAVE_FORMAT_PCM;
	}
	else if (format.wFormatTag == WAVE_FORMAT_EXTENSIBLE) {
		PWAVEFORMATEXTENSIBLE pEx = reinterpret_cast<PWAVEFORMATEXTENSIBLE>(&format);
		if (IsEqualGUID(KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, pEx->SubFormat)) {
			pEx->SubFormat = KSDATAFORMAT_SUBTYPE_PCM;
			pEx->Samples.wValidBitsPerSample = 16;
		}
	}
	format.wBitsPerSample = 16;
	format.nBlockAlign = format.nChannels * format.wBitsPerSample / 8;
	format.nAvgBytesPerSec = format.nBlockAlign * format.nSamplesPerSec;
}

void AudioRecordWin::recordAudio() {
	uint32_t byteSize = 0;
	/* Output devices don't signal, so just make it check every 10 ms */
	DWORD dur = bMic ? RECONNECT_INTERVAL : 10;
	int32_t muteSize = dur * format->nSamplesPerSec * format->nBlockAlign / 1000;
	std::vector<uint8_t> muteData;
	std::vector<uint8_t> audioData;
	muteData.resize(muteSize, 0);
	audioData.resize(muteSize, 0);
	//如果开始录的时候就静音,flags & AUDCLNT_BUFFERFLAGS_SILENT判断不了
	bool bReadData = false;
	HRESULT hr;
	while (true) {
		if (!bStart)
			break;
		DWORD waitResult = WaitForSingleObjectEx(hAudioSamplesReadyEvent, dur, FALSE);
		if (waitResult == WAIT_OBJECT_0 || waitResult == WAIT_TIMEOUT) {
			HRESULT res;
			LPBYTE buffer;
			UINT32 frames;
			DWORD flags;
			UINT64 pos, ts;
			UINT captureSize = 0;
			while (true) {
				if (!bStart)
					break;
				hr = capture->GetNextPacketSize(&captureSize);
				if (FAILED(hr)) {
					std::string audioName = bMic ? "mic" : "loopback";
					std::string msg = audioName + " audio record netx packet faild.";
					checkHR(hr, msg.c_str());
					internalClose();
					return;
				}
				hr = capture->GetBuffer(&buffer, &frames, &flags, &pos, &ts);
				if (FAILED(hr)) {
					std::string audioName = bMic ? "mic" : "loopback";
					std::string msg = audioName + " audio record get buffer faild.";
					checkHR(hr, msg.c_str());
					internalClose();
					return;
				}
				//先把数据复制出来，然后释放掉
				int32_t frameByte = frames * format->nBlockAlign;
				memcpy(audioData.data(), buffer, min(frameByte, muteSize));
				capture->ReleaseBuffer(frames);
				if (onRecordHandle) {
					//静音处理 captureSize == 0 判断开始记录就静音的情况
					if ((captureSize == 0 && !bReadData) || (flags & AUDCLNT_BUFFERFLAGS_SILENT)) {
						onRecordHandle(muteData.data(), muteSize, OEIP_AudioData_None);
						byteSize += muteSize;
					}
					//开始静音,然后有数据,后续静音就用AUDCLNT_BUFFERFLAGS_SILENT判断
					if (captureSize == 0) {
						break;
					}
					bReadData = true;
					if (frameByte > 0) {
						onRecordHandle(audioData.data(), frameByte, OEIP_Audio_Data);
						byteSize += frameByte;
					}
				}
			}
		}
	}
	if (onRecordHandle) {
		std::vector<uint8_t> wavHeader;
		getWavHeader(wavHeader, byteSize);
		onRecordHandle(wavHeader.data(), wavHeader.size(), OEIP_Audio_WavHeader);
	}
	if (client)
		client->Stop();
	//通知结束
	signal.notify_all();
}

void AudioRecordWin::getWavHeader(std::vector<uint8_t>& header, uint32_t dataSize) {
	DWORD headerSize = sizeof(WAVEHEADER) + sizeof(WAVEFORMATEX) + format->cbSize + sizeof(WaveData) + sizeof(DWORD);
	header.resize(headerSize, 0);

	uint8_t* writeIdx = header.data();
	//先写声音基本信息
	WAVEHEADER* waveHeader = reinterpret_cast<WAVEHEADER*>(writeIdx);
	memcpy(writeIdx, WaveHeader, sizeof(WAVEHEADER));
	writeIdx += sizeof(WAVEHEADER);
	waveHeader->dwSize = headerSize + dataSize - 2 * 4;
	waveHeader->dwFmtSize = sizeof(WAVEFORMATEX) + format->cbSize;
	//format后面可能会带cbSize个额外信息
	memcpy(writeIdx, format, sizeof(WAVEFORMATEX) + format->cbSize);
	writeIdx += sizeof(WAVEFORMATEX) + format->cbSize;
	//写入data
	memcpy(writeIdx, WaveData, sizeof(WaveData));
	writeIdx += sizeof(WaveData);
	//写入一个结尾数据长度
	*(reinterpret_cast<DWORD*>(writeIdx)) = static_cast<DWORD>(dataSize);
	writeIdx += sizeof(DWORD);
}

void AudioRecordWin::internalClose() {
	bStart = false;
	signal.notify_all();
	client.Release();
	render.Release();
	capture.Release();
}

void AudioRecordWin::close() {
	if (bStart) {
		bStart = false;
		std::unique_lock <std::mutex> lck(mtx);
		auto status = signal.wait_for(lck, std::chrono::seconds(2));
		if (status == std::cv_status::timeout) {
			logMessage(OEIP_WARN, "audio loop back is not closed properly.");
		}
		onRecordHandle = nullptr;
		client.Release();
		render.Release();
		capture.Release();
	}
}

