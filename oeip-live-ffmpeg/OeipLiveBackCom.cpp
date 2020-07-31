#include "OeipLiveBackCom.h"

#include "../oeip/OeipCommon.h"

OeipLiveBackCom::~OeipLiveBackCom() {
}

HRESULT __stdcall OeipLiveBackCom::QueryInterface(REFIID riid, void** ppvObject) {
	if (riid == __uuidof(IOeipLiveCallBack) || riid == __uuidof(IUnknown)) {
		//if (riid == __uuidof(IOeipLiveCallBack)) {
		*ppvObject = this;
		AddRef();
		return S_OK;
	}
	return E_NOINTERFACE;
}

ULONG __stdcall OeipLiveBackCom::AddRef(void) {
	return InterlockedIncrement(&refCount);
}

ULONG __stdcall OeipLiveBackCom::Release(void) {
	ULONG uCount = InterlockedDecrement(&refCount);
	if (uCount == 0) {
		onSeverFunc = nullptr;
		delete this;
	}
	return uCount;
}

HRESULT __stdcall OeipLiveBackCom::raw_OnInitRoom(long code) {
	if (!liveBack)
		return -1;
	liveBack->onInitRoom(code);
	return 0;
}

HRESULT __stdcall OeipLiveBackCom::raw_OnUserChange(long userId, VARIANT_BOOL bAdd) {
	if (!liveBack)
		return -1;
	liveBack->onUserChange(userId, bAdd);
	return 0;
}

HRESULT __stdcall OeipLiveBackCom::raw_OnLoginRoom(long code, BSTR server, long port) {
	if (!liveBack)
		return -1;
	std::string ip = wstring2string(server);
	if (onSeverFunc) {
		onSeverFunc(ip, port, code);
	}
	//mediaServer = "rtmp://" + ip + ":" + std::to_string(port) + "/live/";
	liveBack->onLoginRoom(code, code);
	return 0;
}

HRESULT __stdcall OeipLiveBackCom::raw_OnStreamUpdate(long userId, long index, VARIANT_BOOL bAdd) {
	if (!liveBack)
		return -1;
	liveBack->onStreamUpdate(userId, index, bAdd);
	return 0;
}

HRESULT __stdcall OeipLiveBackCom::raw_OnLogoutRoom() {
	if (!liveBack)
		return -1;	
	liveBack->onLogoutRoom(0);
	return 0;
}

HRESULT __stdcall OeipLiveBackCom::raw_OnOperateResult(long operate, long code, BSTR message) {
	if (!liveBack)
		return -1;
	std::string msg = wstring2string(message);
	liveBack->onOperateResult(operate, code, msg.c_str());
	return 0;
}

HRESULT __stdcall OeipLiveBackCom::raw_Dispose() {
	//在这只置空，不去delete,由持有liveBack对象的对象管理
	liveBack = nullptr;
	return 0;
}
