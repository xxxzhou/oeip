#pragma once
#import "OeipLiveCom.tlb" named_guids raw_interface_only
#include "../oeip-live/OeipLiveBack.h"
using namespace OeipLiveCom;

class OeipLiveBackCom : public IOeipLiveCallBack
{
public:
	std::string mediaServer = "";
	int32_t userId = 0;
private:
	OeipLiveBack* liveBack = nullptr;
	long refCount = 0;
public:
	void setLiveBack(OeipLiveBack* callBack) {
		liveBack = callBack;
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
};

