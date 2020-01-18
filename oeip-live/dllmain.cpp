// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include <Windows.h>
#include <string>
#include <vector>
#include "../oeip/OeipCommon.h"
#include "../oeip/OeipManager.h"
#include "OeipLiveRoom.h"

#pragma comment(lib,"shlwapi.lib")

#ifdef _WINDLL
#include <sysinfoapi.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

LONG WINAPI MyUnhandledFilter(struct _EXCEPTION_POINTERS* lpExceptionInfo) {
	LONG ret = EXCEPTION_EXECUTE_HANDLER;

	TCHAR szFileName[64];
	SYSTEMTIME st;
	::GetLocalTime(&st);
	wsprintf(szFileName, TEXT("OEIP_LIVE_%04d%02d%02d-%02d%02d%02d-%ld-%ld.dmp"), st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, GetCurrentProcessId(), GetCurrentThreadId());

	HANDLE hFile = ::CreateFile(szFileName, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile != INVALID_HANDLE_VALUE) {
		MINIDUMP_EXCEPTION_INFORMATION ExInfo;
		ExInfo.ThreadId = ::GetCurrentThreadId();
		ExInfo.ExceptionPointers = lpExceptionInfo;
		ExInfo.ClientPointers = false;

		// write the dump
		BOOL bOK = MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, MiniDumpNormal, &ExInfo, NULL, NULL);
		::CloseHandle(hFile);
	}
	return ret;
}
#endif // _WINDLL

static bool bLoad = false;

void loadAllDll() {
	std::vector<std::wstring> dllList = { L"oeip-live-ffmpeg" };
	loadDllArray(dllList);
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD  dwReason, LPVOID lpReserved) {
	if (dwReason == DLL_PROCESS_ATTACH) {
		if (!bLoad) {
			loadAllDll();
			bLoad = true;
#ifdef _WINDLL
			SetUnhandledExceptionFilter((LPTOP_LEVEL_EXCEPTION_FILTER)MyUnhandledFilter);
#endif
		}
	}
	else if (dwReason == DLL_PROCESS_DETACH) {
		if (bLoad) {
			PluginManager<OeipLiveRoom>::clean(true);
			bLoad = false;
		}
	}
	return TRUE;
}

