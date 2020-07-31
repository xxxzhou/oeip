#pragma once
#import "OeipLiveCom.tlb" named_guids raw_interface_only
#include "../oeip-live/OeipLiveBack.h"
#include <string>
using namespace OeipLiveCom;

typedef std::function<void(std::string server, int32_t port, int32_t userId)> onServeBack;

class OeipLiveBackCom : public IOeipLiveCallBack
{
public:
	OeipLiveBackCom() {};
	~OeipLiveBackCom();
private:
	OeipLiveBack* liveBack = nullptr;
	onServeBack onSeverFunc = nullptr;
	long refCount = 0;
public:
	void setLiveBack(OeipLiveBack* callBack, onServeBack serverBack) {
		liveBack = callBack;
		onSeverFunc = serverBack;
	}
public:
	// 通过 IUnknown 继承
	virtual HRESULT __stdcall QueryInterface(REFIID riid, void** ppvObject) override;
	virtual ULONG __stdcall AddRef(void) override;
	virtual ULONG __stdcall Release(void) override;
public:
	// 通过 IOeipLiveCallBack 继承
	virtual HRESULT __stdcall raw_OnInitRoom(long code) override;
	virtual HRESULT __stdcall raw_OnUserChange(long userId, VARIANT_BOOL bAdd) override;
	virtual HRESULT __stdcall raw_OnLoginRoom(long code, BSTR server, long port) override;
	virtual HRESULT __stdcall raw_OnStreamUpdate(long userId, long index, VARIANT_BOOL bAdd) override;
	virtual HRESULT __stdcall raw_OnLogoutRoom() override;
	virtual HRESULT __stdcall raw_OnOperateResult(long operate, long code, BSTR message) override;
	virtual HRESULT __stdcall raw_Dispose() override;
};

