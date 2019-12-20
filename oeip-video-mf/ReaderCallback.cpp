#include "ReaderCallback.h"
#include <shlwapi.h>
#include "MediaStruct.h"

ReaderCallback::ReaderCallback()
{
}

ReaderCallback::~ReaderCallback()
{
}

void ReaderCallback::setSourceReader(IMFSourceReader* pReader, unsigned long dwStreamIndex)
{
	std::lock_guard<std::mutex> mtx_locker(mtx);
	reader = pReader;
	streamIndex = dwStreamIndex;
}

void ReaderCallback::setBufferRevice(function<void(unsigned long, byte*)> reviceFunc)
{
	onReviceBuffer = reviceFunc;
}

void ReaderCallback::setDeviceEvent(onEventHandle eventHandle)
{
	onDeviceEvent = eventHandle;
}

bool ReaderCallback::setPlay(bool pPlayVideo)
{
	std::lock_guard<std::mutex> mtx_locker(mtx);
	HRESULT hr = 0;
	//设置播放
	if (pPlayVideo) {
		if (bPlay)
			return true;
		int i = 0;
		hr = reader->ReadSample(streamIndex, 0, nullptr, nullptr, nullptr, nullptr);
		//最多试验三次
		while (FAILED(hr) && i++ < 3) {
			hr = reader->ReadSample(streamIndex, 0, nullptr, nullptr, nullptr, nullptr);
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
		if (SUCCEEDED(hr)) {
			bPlay = true;
			logMessage(OEIP_INFO, "start reading data.");
			onDeviceHandle(OEIP_DeviceOpen, 0);
			return true;
		}
		else {
			bPlay = false;
			logMessage(OEIP_ERROR, "cannot start reading data.");
			onDeviceHandle(OEIP_DeviceNoOpen, 0);
			return false;
		}
	}
	else {
		if (bPlay) {
			bPlay = false;
			//加了这个，后面才能改变不同的fomat,如nv12变成mjpg
			try {
				if (reader != nullptr)
					hr = reader->Flush(streamIndex);
			}
			catch (exception e) {
				logMessage(OEIP_WARN, e.what());
			}
			//等待OnFlush完成
			std::unique_lock <std::mutex> lck(mtx2);
			auto status = signal.wait_for(lck, std::chrono::seconds(3));
			if (status == std::cv_status::timeout) {
				logMessage(OEIP_WARN, "MF device is not closed properly.");
			}
		}
		return SUCCEEDED(hr);
	}
}

void ReaderCallback::onDeviceHandle(OeipDeviceEventType eventType, int32_t data)
{
	if (onDeviceEvent) {
		onDeviceEvent(eventType, data);
	}
}

HRESULT ReaderCallback::QueryInterface(REFIID riid, void** ppvObject)
{
	static const QITAB qit[] =
	{
		QITABENT(ReaderCallback, IMFSourceReaderCallback),
		{ 0 },
	};
	return QISearch(this, qit, riid, ppvObject);
}

ULONG ReaderCallback::AddRef(void)
{
	return InterlockedIncrement(&refCount);
}

ULONG ReaderCallback::Release(void)
{
	ULONG uCount = InterlockedDecrement(&refCount);
	if (uCount == 0) {
		onReviceBuffer = nullptr;
		delete this;
	}
	// For thread safety, return a temporary variable.
	return uCount;
}

HRESULT ReaderCallback::OnReadSample(HRESULT hrStatus, DWORD dwStreamIndex, DWORD dwStreamFlags, LONGLONG llTimestamp, IMFSample* pSample)
{
	//std::lock_guard<std::mutex> mtx_locker(mtx);
	HRESULT hr = S_OK;
	//人为中断
	if (!bPlay)
		return hr;
	if (pSample) {
		CComPtr<IMFMediaBuffer> pBuffer = nullptr;
		DWORD lenght;
		pSample->GetTotalLength(&lenght);
		hr = pSample->GetBufferByIndex(0, &pBuffer);
		if (pBuffer) {
			unsigned long length = 0;
			unsigned long maxLength = 0;
			pBuffer->GetCurrentLength(&length);
			//pBuffer->GetMaxLength(&maxLength);
			if (onReviceBuffer) {
				byte* data = nullptr;
				auto hr = pBuffer->Lock(&data, &length, &length);
				onReviceBuffer(length, (uint8_t*)data);
				pBuffer->Unlock();
			}
		}		
	}
	// Request the next frame.
	if (SUCCEEDED(hr)) {
		hr = reader->ReadSample(streamIndex, 0, nullptr, nullptr, nullptr, nullptr);
	}
	if (FAILED(hr)) {
		bPlay = false;
		onDeviceHandle(OEIP_DeviceDropped, 0);
		logMessage(OEIP_WARN, "Data interruption.");
	}
	return hr;
}

HRESULT ReaderCallback::OnFlush(DWORD dwStreamIndex)
{
	//通知closeDevice能关闭设备了
	signal.notify_all();
	return S_OK;
}

HRESULT ReaderCallback::OnEvent(DWORD dwStreamIndex, IMFMediaEvent* pEvent)
{
	return S_OK;
}
