#pragma once
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <functional>
#include <mutex>
#include <OeipCommon.h>

using namespace std;
class ReaderCallback :public IMFSourceReaderCallback
{
private:
	long refCount = 0;
	bool bPlay = false;
	unsigned long streamIndex = -1;
	IMFSourceReader* reader = nullptr;
	function<void(unsigned long, uint8_t*)> onReviceBuffer = nullptr;
	onEventHandle onDeviceEvent = nullptr;
public:
	ReaderCallback();
	~ReaderCallback();
	//std::mutex mtx;
	std::mutex mtx;

	std::mutex mtx2;
	//信号量.
	std::condition_variable signal;
public:
	bool IsOpen() { return bPlay; }
	void setSourceReader(IMFSourceReader* pReader, unsigned long dwStreamIndex);
	void setBufferRevice(function<void(unsigned long, uint8_t*)> reviceFunc);
	void setDeviceEvent(onEventHandle eventHandle);
	bool setPlay(bool pPlayVideo);
	void onDeviceHandle(OeipDeviceEventType eventType, int32_t code);
	// 通过 IMFSourceReaderCallback 继承
	virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;
	virtual ULONG STDMETHODCALLTYPE AddRef(void) override;
	virtual ULONG STDMETHODCALLTYPE Release(void) override;
	virtual HRESULT STDMETHODCALLTYPE OnReadSample(HRESULT hrStatus, DWORD dwStreamIndex, DWORD dwStreamFlags, LONGLONG llTimestamp, IMFSample* pSample) override;
	virtual HRESULT STDMETHODCALLTYPE OnFlush(DWORD dwStreamIndex) override;
	virtual HRESULT STDMETHODCALLTYPE OnEvent(DWORD dwStreamIndex, IMFMediaEvent* pEvent) override;
};

