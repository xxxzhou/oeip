#include "MFVideoManager.h"
#include "MFCaptureDevice.h"
#include "VideoCaptureDevice.h"
#include "AudioRecordWin.h"

MFVideoManager::MFVideoManager() {
	CComPtr<IMFAttributes> pAttributes = nullptr;
	auto hr = MFCreateAttributes(&pAttributes, 1);
	if (SUCCEEDED(hr)) {
		hr = pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	}
	if (SUCCEEDED(hr)) {
		IMFActivate** ppDevices = nullptr;
		UINT32 count = -1;
		hr = MFEnumDeviceSources(pAttributes, &ppDevices, &count);
		if (SUCCEEDED(hr)) {
			if (count > 0) {
				for (UINT32 i = 0; i < count; i++) {
					//过滤掉RealSense2					
					if (MFCaptureDevice::Init(ppDevices[i])) {
						VideoCaptureDevice* vc = new VideoCaptureDevice();
						if (vc->init(ppDevices[i], i)) {
							videoList.push_back(vc);
						}
						else {
							delete vc;
						}
					}
					ppDevices[i]->Release();
				}
			}
			CoTaskMemFree(ppDevices);
		}
	}
}


MFVideoManager::~MFVideoManager() {
	HRESULT hr = MFShutdown();
	CoUninitialize();
}

std::vector<VideoDevice*> MFVideoManager::getDeviceList() {
	return videoList;
}

bool bCanLoad() {
	auto hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
	hr &= MFStartup(MF_VERSION);
	hr &= CoInitialize(NULL);
	if (!SUCCEEDED(hr)) {
		logMessage(OEIP_ERROR, "Media foundation not initialized correctly.");
		return false;
	}
	else {
		logMessage(OEIP_INFO, "Media foundation correctly initialized.");
		return true;
	}
}

void registerFactory() {
	registerFactory(new MFVideoManagerFactory(), VideoDeviceType::OEIP_MF, "video mf");
	registerFactory(new AudioRecordWinFactory(), 0, "audio record mf");
}


////IMFSourceReader可以把压缩转末压缩的，但是不能把末压缩的转压缩的
//bool MediaManager::CombinVideoAudio(const std::wstring& videoPath, const std::wstring& audioPath, const std::wstring& resultPath)
//{
//	HRESULT hr;
//	MediaType mt;
//	DWORD videoIndex;
//	DWORD audioIndex;
//	UINT32 preSecond, perSample, numChannels;
//	LONGLONG videoTime, audioTime;// 100-nanosecond units.100纳秒 1秒= 1000000000纳秒
//	IMFSourceResolver* pSourceResolver = nullptr;
//	IMFMediaSource* videoSource = nullptr;
//	CComPtr<IMFSourceReader> videoReader = nullptr;
//	IMFMediaSource* audioSource = nullptr;
//	CComPtr<IMFSourceReader> audioReader = nullptr;
//	IMFMediaType* videoMediaType = nullptr;
//	IMFMediaType* audioMediaType = nullptr;
//	IUnknown* uSource = nullptr;
//	IMFAttributes* pVideoReaderAttributes = nullptr;
//	MF_OBJECT_TYPE ObjectType = MF_OBJECT_INVALID;
//	//得到原视频信息
//	hr = MFCreateSourceResolver(&pSourceResolver);
//	hr = pSourceResolver->CreateObjectFromURL(videoPath.data(), MF_RESOLUTION_MEDIASOURCE, nullptr, &ObjectType, &uSource);
//	hr = uSource->QueryInterface(IID_PPV_ARGS(&videoSource));
//	hr = MFCreateAttributes(&pVideoReaderAttributes, 4);
//	hr = pVideoReaderAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
//	hr = pVideoReaderAttributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1);
//	hr = pVideoReaderAttributes->SetUINT32(MF_LOW_LATENCY, FALSE); // Allows better multithreading	
//	hr = MFCreateSourceReaderFromMediaSource(videoSource, pVideoReaderAttributes, &videoReader);
//	hr = videoReader->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, &videoMediaType);
//	mt = FormatReader::Read(videoMediaType);
//	SafeRelease(pSourceResolver);
//	SafeRelease(videoSource);
//	SafeRelease(videoMediaType);
//	SafeRelease(uSource);
//	SafeRelease(pVideoReaderAttributes);
//	//得到音频信息
//	hr = MFCreateSourceResolver(&pSourceResolver);
//	hr = pSourceResolver->CreateObjectFromURL(audioPath.data(), MF_RESOLUTION_MEDIASOURCE, nullptr, &ObjectType, &uSource);
//	hr = uSource->QueryInterface(IID_PPV_ARGS(&audioSource));
//	hr = MFCreateAttributes(&pVideoReaderAttributes, 1);
//	hr = pVideoReaderAttributes->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, TRUE);
//	hr = MFCreateSourceReaderFromMediaSource(audioSource, pVideoReaderAttributes, &audioReader);
//	hr = audioReader->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, &audioMediaType);
//	GUID subtype;
//	hr = audioMediaType->GetGUID(MF_MT_SUBTYPE, &subtype);
//	if (subtype != MFAudioFormat_AAC && subtype != MFAudioFormat_MP3)
//	{
//		LogMessage(error, "Please confirm that the incoming audio file is in aac or mp3 format.");
//		return false;
//	}
//	hr = audioReader->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, nullptr, audioMediaType);
//	hr = audioMediaType->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &preSecond);
//	hr = audioMediaType->GetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, &perSample);
//	hr = audioMediaType->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &numChannels);
//	SafeRelease(pSourceResolver);
//	SafeRelease(audioSource);
//	SafeRelease(uSource);
//	SafeRelease(pVideoReaderAttributes);
//	SafeRelease(audioMediaType);
//	//合成视频设置
//	IMFAttributes* attr = nullptr;
//	IMFSinkWriter* pSinkWriter = nullptr;
//	MFCreateAttributes(&attr, 2);
//	attr->SetGUID(MF_TRANSCODE_CONTAINERTYPE, MFTranscodeContainerType_MPEG4);
//	attr->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, 1); // Or 1, doesn't make any difference
//	hr = MFCreateSinkWriterFromURL(resultPath.data(), nullptr, attr, &pSinkWriter);
//	SafeRelease(attr);
//	//合成视频对应的设置
//	IMFMediaType* pVideoOutType = nullptr;
//	hr = MFCreateMediaType(&pVideoOutType);
//	hr = pVideoOutType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
//	hr = pVideoOutType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264);
//	UINT32 bitRate = mt.bitRate / 100 * 100;
//	hr = pVideoOutType->SetUINT32(MF_MT_AVG_BITRATE, bitRate);
//	hr = MFSetAttributeSize(pVideoOutType, MF_MT_FRAME_SIZE, mt.width, mt.height);
//	hr = MFSetAttributeRatio(pVideoOutType, MF_MT_FRAME_RATE, mt.frameRate, 1);
//	hr = MFSetAttributeRatio(pVideoOutType, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
//	hr = pVideoOutType->SetUINT32(MF_MT_INTERLACE_MODE, 2);
//	hr = pSinkWriter->AddStream(pVideoOutType, &videoIndex);
//	SafeRelease(pVideoOutType);
//	//合成音频对应设置
//	IMFMediaType* pAudioOutType = nullptr;
//	hr = MFCreateMediaType(&pAudioOutType);
//	hr = pAudioOutType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
//	hr = pAudioOutType->SetGUID(MF_MT_SUBTYPE, subtype);
//	hr = pAudioOutType->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, preSecond);
//	hr = pAudioOutType->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, perSample);
//	hr = pAudioOutType->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, numChannels);
//	hr = pAudioOutType->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, 12000);
//	UINT32 block = numChannels * (perSample / 8);
//	hr = pAudioOutType->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, block);
//	hr = pSinkWriter->AddStream(pAudioOutType, &audioIndex);
//	SafeRelease(pAudioOutType);
//	IMFMediaType* pAudioOutTypeCopy = nullptr;
//	hr = MFCreateMediaType(&pAudioOutTypeCopy);
//	hr = pAudioOutTypeCopy->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
//	hr = pAudioOutTypeCopy->SetGUID(MF_MT_SUBTYPE, subtype);
//	hr = pAudioOutTypeCopy->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, preSecond);
//	hr = pAudioOutTypeCopy->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, perSample);
//	hr = pAudioOutTypeCopy->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, numChannels);
//	hr = pAudioOutTypeCopy->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, 12000);
//	SafeRelease(pAudioOutTypeCopy);
//	hr = GetDuration(videoReader, &videoTime);
//	checkHR(hr, "can't get the right video time");
//	hr = GetDuration(audioReader, &audioTime);
//	checkHR(hr, "can't get the right audio time");
//	//合成视频
//	hr = pSinkWriter->BeginWriting();
//	//https://social.msdn.microsoft.com/Forums/windowsdesktop/en-US/20727947-98fe-4245-ad3a-8056d168a1b5/imfsinkwriter-very-slow-and-use-of-mfsinkwriterdisablethrottling?forum=mediafoundationdevelopment
//	//这里是个大坑，在一个线程同时写音频与视频会导致pSinkWriter->WriteSample非常慢，因为同时写的时候，会自动去同步音频与视频的时间戳.
//	//在同一线程就会造成要同步时就卡一段时间，故用二个线程同时写，让pSinkWriter->WriteSample能自动同步不需要等待
//	std::future<bool> writeVideo = std::async([&videoReader, &videoIndex, &pSinkWriter, &audioTime]() {
//		bool result = true;
//		LONGLONG videoTimeStamp = 0;// 100-nanosecond units.100纳秒 1秒= 1000000000纳秒
//		while (true)
//		{
//			DWORD streamIndex, flags;
//			CComPtr<IMFSample> videoSample = nullptr;
//			HRESULT hr = videoReader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, &streamIndex, &flags, &videoTimeStamp, &videoSample);
//			if (SUCCEEDED(hr) && videoSample)
//			{
//				videoSample->SetSampleTime(videoTimeStamp);
//				hr = pSinkWriter->WriteSample(videoIndex, videoSample);
//			}
//			else
//			{
//				if (FAILED(hr))
//					result = false;
//				break;
//			}
//			if (videoTimeStamp > audioTime)
//				break;
//		}
//		return result;
//	});
//	std::future<bool> writeAudio = std::async([&audioReader, &audioIndex, &pSinkWriter, &videoTime]() {
//		bool result = true;
//		LONGLONG audioTimeStamp = 0;
//		while (true)
//		{
//			DWORD streamIndex, flags;
//			CComPtr<IMFSample> audioSample = nullptr;
//			HRESULT hr = audioReader->ReadSample(MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, &streamIndex, &flags, &audioTimeStamp, &audioSample);
//			if (SUCCEEDED(hr) && audioSample)
//			{
//				audioSample->SetSampleTime(audioTimeStamp);
//				hr = pSinkWriter->WriteSample(audioIndex, audioSample);
//			}
//			else
//			{
//				if (FAILED(hr))
//					result = false;
//				break;
//			}
//			if (audioTimeStamp > videoTime)
//				break;
//		}
//		return result;
//	});
//	bool result = writeVideo.get() && writeAudio.get();
//	pSinkWriter->Finalize();
//	SafeRelease(pSinkWriter);
//	return result;
//}
//
//bool MediaManager::PCM2AAC(const std::wstring& pcmPath, const std::wstring& accPath)
//{
//	bool result = false;
//	HRESULT hr;
//	DWORD audioIndex;
//	UINT32 preSecond, perSample, numChannels;
//	DWORD streamIndex, flags;
//	LONGLONG llVideoTimeStamp;
//	CComPtr<IMFSourceResolver> pSourceResolver = nullptr;
//	CComPtr<IMFMediaSource> audioSource = nullptr;
//	CComPtr<IMFSourceReader> audioReader = nullptr;
//	IMFMediaType* audioMediaType = nullptr;
//	CComPtr<IUnknown> uSource = nullptr;
//	CComPtr<IMFAttributes> pVideoReaderAttributes = nullptr;
//	MF_OBJECT_TYPE ObjectType = MF_OBJECT_INVALID;
//	//PCM to acc
//	CComPtr<IMFTransform> pTransform = nullptr;
//	CComPtr<IUnknown> spTransformUnk = nullptr;
//	hr = CoCreateInstance(CLSID_AACMFTEncoder, NULL, CLSCTX_INPROC_SERVER,
//		IID_IUnknown, (void**)&spTransformUnk);//CLSID_AACMFTEncoder,CLSID_MP3ACMCodecWrapper
//	hr = spTransformUnk->QueryInterface(IID_PPV_ARGS(&pTransform));
//	//acc音频设置
//	CComPtr<IMFAttributes> attr = nullptr;
//	CComPtr<IMFSinkWriter> pSinkWriter = nullptr;
//	MFCreateAttributes(&attr, 2);
//	attr->SetGUID(MF_TRANSCODE_CONTAINERTYPE, MFTranscodeContainerType_MPEG4);
//	attr->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, 1); // Or 1, doesn't make any difference
//	hr = MFCreateSinkWriterFromURL(accPath.data(), nullptr, attr, &pSinkWriter);
//	if (!checkHR(hr, "the current aac file cannot be written."))
//		return false;
//	//pcm音频信息
//	hr = MFCreateSourceResolver(&pSourceResolver);
//	hr = pSourceResolver->CreateObjectFromURL(pcmPath.data(), MF_RESOLUTION_MEDIASOURCE, nullptr, &ObjectType, &uSource);
//	hr = uSource->QueryInterface(IID_PPV_ARGS(&audioSource));
//	hr = MFCreateAttributes(&pVideoReaderAttributes, 1);
//	hr = pVideoReaderAttributes->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, TRUE);
//	hr = MFCreateSourceReaderFromMediaSource(audioSource, pVideoReaderAttributes, &audioReader);
//	hr = audioReader->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, &audioMediaType);
//	GUID subtype;
//	hr = audioMediaType->GetGUID(MF_MT_SUBTYPE, &subtype);
//	if (subtype != MFAudioFormat_PCM)
//		return false;
//	hr = audioReader->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, nullptr, audioMediaType);
//	hr = audioMediaType->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &preSecond);
//	hr = audioMediaType->GetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, &perSample);
//	hr = audioMediaType->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &numChannels);
//	hr = pTransform->SetInputType(0, audioMediaType, 0);
//	if (!checkHR(hr, "converter input setting error"))
//		return false;
//	//合成音频对应设置	
//	IMFMediaType* pAudioOutType = nullptr;
//	hr = MFCreateMediaType(&pAudioOutType);
//	hr = pAudioOutType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
//	hr = pAudioOutType->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_AAC);
//	hr = pAudioOutType->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, preSecond);
//	hr = pAudioOutType->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, perSample);
//	hr = pAudioOutType->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, numChannels);
//	hr = pAudioOutType->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, 12000);
//	UINT32 block = numChannels * (perSample / 8);
//	hr = pAudioOutType->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, block);
//	hr = pSinkWriter->AddStream(pAudioOutType, &audioIndex);
//	IMFMediaType* pAudioOutTypeCopy = nullptr;
//	hr = MFCreateMediaType(&pAudioOutTypeCopy);
//	hr = pAudioOutTypeCopy->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
//	hr = pAudioOutTypeCopy->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_AAC);
//	hr = pAudioOutTypeCopy->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, preSecond);
//	hr = pAudioOutTypeCopy->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, perSample);
//	hr = pAudioOutTypeCopy->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, numChannels);
//	hr = pAudioOutTypeCopy->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, 12000);
//	hr = pAudioOutTypeCopy->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, block);
//	if (!checkHR(hr, "converter output setting error"))
//		return false;
//	hr = pTransform->SetOutputType(0, pAudioOutTypeCopy, 0);
//	SafeRelease(pAudioOutType);
//	SafeRelease(pAudioOutTypeCopy);
//	//开始转码
//	result = SUCCEEDED(pSinkWriter->BeginWriting());
//	while (result)
//	{
//		CComPtr<IMFSample> videoSample = nullptr;
//		hr = audioReader->ReadSample(MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, &streamIndex, &flags, &llVideoTimeStamp, &videoSample);
//		if (SUCCEEDED(hr) && videoSample)
//		{
//			DWORD mftOutFlags = 0;
//			hr = pTransform->ProcessInput(0, videoSample, 0);
//			hr = pTransform->GetOutputStatus(&mftOutFlags);
//			if (mftOutFlags == MFT_OUTPUT_STATUS_SAMPLE_READY)
//			{
//				MFT_OUTPUT_STREAM_INFO StreamInfo;
//				//H264的数据
//				hr = pTransform->GetOutputStreamInfo(0, &StreamInfo);
//				while (true)
//				{
//					CComPtr<IMFSample> mftOutSample = nullptr;
//					CComPtr<IMFMediaBuffer> pBuffer1 = nullptr;
//					MFT_OUTPUT_DATA_BUFFER outputDataBuffer = {};
//					MFCreateSample(&mftOutSample);
//					MFCreateMemoryBuffer(StreamInfo.cbSize, &pBuffer1);
//					mftOutSample->AddBuffer(pBuffer1);
//					outputDataBuffer.dwStreamID = 0;
//					outputDataBuffer.dwStatus = 0;
//					outputDataBuffer.pEvents = nullptr;
//					outputDataBuffer.pSample = mftOutSample;
//
//					DWORD processOutputStatus = 0;
//					HRESULT mftProcessOutput = pTransform->ProcessOutput(0, 1, &outputDataBuffer, &processOutputStatus);
//					if (mftProcessOutput != MF_E_TRANSFORM_NEED_MORE_INPUT)
//					{
//						long long sampleDuration;
//						videoSample->GetSampleDuration(&sampleDuration);
//						outputDataBuffer.pSample->SetSampleDuration(sampleDuration);
//						hr = pSinkWriter->WriteSample(audioIndex, outputDataBuffer.pSample);
//					}
//					else
//						break;
//				}
//			}
//		}
//		else
//			break;
//	}
//	pSinkWriter->Finalize();
//	SafeRelease(audioMediaType);
//	return result;
//}