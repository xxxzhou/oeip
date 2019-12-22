#include "AudioRecordWin.h"
#include <OeipCommon.h>
#include <atlcomcli.h>

AudioRecordWin::AudioRecordWin()
{
}


AudioRecordWin::~AudioRecordWin()
{
	close();
}

bool AudioRecordWin::initRecord()
{
	CComPtr<IMMDeviceEnumerator> enumerator = nullptr;
	CComPtr<IMMDevice> device;
	HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_INPROC_SERVER, __uuidof(IMMDeviceEnumerator), reinterpret_cast<void**>(&enumerator));
	//hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
	EDataFlow dataFlow = eCapture;
	if (recordType == OEIP_Loopback) {
		dataFlow = eRender;
	}
	else if (recordType == OEIP_Mic_Loopback) {
		dataFlow = eAll;
	}

	hr = enumerator->GetDefaultAudioEndpoint(dataFlow, eConsole, &device);
	hr = device->Activate(__uuidof(IAudioClient), CLSCTX_SERVER, NULL, reinterpret_cast<void**>(&client));

	hr = client->GetMixFormat(&format);
	setWaveFormat(*format);
	//以100纳秒为单位
	hr = client->GetDevicePeriod(&waitTime, NULL);
	if (recordType == OEIP_Loopback)
	{
		hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK, 0, 0, format, NULL);
		hr = client->GetService(__uuidof(IAudioCaptureClient), reinterpret_cast<void**>(&capture));
	}
	else
	{
		hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_NOPERSIST,// AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_NOPERSIST,
			0, 0, format, NULL);
		hr = client->GetService(__uuidof(IAudioCaptureClient), reinterpret_cast<void**>(&capture));
	}
	hr = client->Start();

	bStart = true;
	recordThread = std::thread(std::bind(&AudioRecordWin::recordAudio, this));
	recordThread.detach();
	return SUCCEEDED(hr);
}

void AudioRecordWin::setWaveFormat(WAVEFORMATEX& format)
{
	//转成16通道输出，32位完全没必要 
	if (format.wFormatTag == WAVE_FORMAT_IEEE_FLOAT)
	{
		format.wFormatTag = WAVE_FORMAT_PCM;
	}
	else if (format.wFormatTag == WAVE_FORMAT_EXTENSIBLE)
	{
		PWAVEFORMATEXTENSIBLE pEx = reinterpret_cast<PWAVEFORMATEXTENSIBLE>(&format);
		if (IsEqualGUID(KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, pEx->SubFormat))
		{
			pEx->SubFormat = KSDATAFORMAT_SUBTYPE_PCM;
			pEx->Samples.wValidBitsPerSample = 16;
		}
	}
	format.wBitsPerSample = 16;
	format.nBlockAlign = format.nChannels * format.wBitsPerSample / 8;
	format.nAvgBytesPerSec = format.nBlockAlign * format.nSamplesPerSec;
	if (!bRecordHandle)
		audioFile->setAudioInfo(filePath, format.nChannels, format.wBitsPerSample, format.nSamplesPerSec);

}

void AudioRecordWin::recordAudio()
{
	int sampleTime = waitTime / 2 / (10 * 1000);
	while (bStart)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(sampleTime));

		BYTE* data = nullptr;
		UINT32 size;
		DWORD flags;
		UINT64 device, performance;
		HRESULT hr = capture->GetNextPacketSize(&size);

		hr = capture->GetBuffer(&data, &size, &flags, &device, &performance);
		int byteWrite = size * format->nBlockAlign;
		uint8_t* formatData = (uint8_t*)data;
		if (bRecordHandle)
			onAudioRecordFunc(data, byteWrite, format->nSamplesPerSec, format->nChannels);
		else
			audioFile->writeData(formatData, byteWrite);
		hr = capture->ReleaseBuffer(size);
	}
	client->Stop();
	safeRelease(client);
	safeRelease(capture);
	//通知结束
	signal.notify_all();
}

bool AudioRecordWin::initRecord(OeipAudioRecordType audiotype, onAudioRecordHandle handle)
{
	if (bStart) {
		logMessage(OEIP_WARN, "audio recording...");
		return false;
	}
	onAudioRecordFunc = handle;
	bRecordHandle = true;
	return initRecord();
}

bool AudioRecordWin::initRecord(OeipAudioRecordType audiotype, std::wstring path) {
	if (bStart) {
		logMessage(OEIP_WARN, "audio recording...");
		return false;
	}
	recordType = audiotype;
	audioFile = new AudioFile();
	filePath = path;
	bRecordHandle = false;
	return initRecord();
}

void AudioRecordWin::close() {
	if (bStart)
	{
		bStart = false;
		std::unique_lock <std::mutex> lck(mtx);
		auto status = signal.wait_for(lck, std::chrono::seconds(2));
		if (status == std::cv_status::timeout) {
			logMessage(OEIP_WARN, "audio loop back is not closed properly.");
		}
		if (bRecordHandle) {
			onAudioRecordFunc = nullptr;
		}
		else if (audioFile) {
			audioFile->close();
			delete audioFile;
			audioFile = nullptr;
		}
	}
}

AudioRecord* MFAudioRecordFactory::create(int type) {
	AudioRecordWin* av = new AudioRecordWin();
	return av;
}
