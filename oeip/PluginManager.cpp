#include "PluginManager.h"
#include <string>
#include <stdio.h>
#include <ctime>
#include <iomanip>
#include <Shlwapi.h>
#include <iostream>
#include "OeipCommon.h"
#include "OeipManager.h"

#pragma comment(lib,"shlwapi.lib")

#ifdef _WINDLL
#include <Windows.h>
#include <sysinfoapi.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

LONG WINAPI MyUnhandledFilter(struct _EXCEPTION_POINTERS* lpExceptionInfo) {
	LONG ret = EXCEPTION_EXECUTE_HANDLER;

	TCHAR szFileName[64];
	SYSTEMTIME st;
	::GetLocalTime(&st);
	wsprintf(szFileName, TEXT("OEIP_%04d%02d%02d-%02d%02d%02d-%ld-%ld.dmp"), st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, GetCurrentProcessId(), GetCurrentThreadId());

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

#define OEIP_MODEL_NAME L"oeip.dll"

//static HINSTANCE hdll = nullptr;
static bool bLoad = false;
static std::vector<HINSTANCE> hdlls;

typedef bool(*bCanLoad)();
typedef void(*registerfactory)();

void loadDll(std::wstring dllName, std::wstring subDirt) {
	HINSTANCE hdll = LoadLibraryEx(dllName.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
	std::string sdname = wstring2string(dllName);
	if (hdll == nullptr) {
		if (!subDirt.empty()) {
			wchar_t temp[512] = { 0 };
			GetDllDirectory(512, temp);
			wchar_t sz[512] = { 0 };
			HMODULE ihdll = GetModuleHandle(OEIP_MODEL_NAME);
			::GetModuleFileName(ihdll, sz, 512);
			::PathRemoveFileSpec(sz);
			::PathAppend(sz, subDirt.c_str());
			SetDllDirectory(sz);//SetCurrentDirectory(sz);
			hdll = LoadLibraryEx(dllName.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
			SetDllDirectory(temp);
		}
		if (hdll == nullptr) {
			DWORD error_id = GetLastError();
#if OEIP_LOADDLL_OUTPUT
			std::string message = "load dll:" + sdname + " error-" + std::to_string(error_id);
			loadMessage(OEIP_ERROR, message.c_str());
#endif
		}
	}
	if (hdll) {
		bCanLoad bcd = (bCanLoad)GetProcAddress(hdll, "bCanLoad");
		if (bcd && bcd()) {
			registerfactory rf = (registerfactory)GetProcAddress(hdll, "registerFactory");
			if (rf)
				rf();
#if OEIP_LOADDLL_OUTPUT
			std::string message = "load dll:" + sdname + " sucess.";
			loadMessage(OEIP_INFO, message.c_str());
#endif
			hdlls.push_back(hdll);
		}
		else {
			FreeLibrary(hdll);
			hdll = nullptr;
#if OEIP_LOADDLL_OUTPUT
			std::string message = "dll:" + sdname + " loading conditions do not match.";
			loadMessage(OEIP_ERROR, message.c_str());
#endif
		}
	}
}

void loadDllArray(std::vector<std::wstring> dllNames) {
	wchar_t temp[512] = { 0 };
	GetDllDirectory(512, temp);
	wchar_t sz[512] = { 0 };
	HMODULE ihdll = GetModuleHandle(OEIP_MODEL_NAME);
	::GetModuleFileName(ihdll, sz, 512);
	::PathRemoveFileSpec(sz);
	SetDllDirectory(sz);
	for (auto dll : dllNames) {
		std::wstring subDirt = L"";
		if (dll == L"oeip-win-cuda")
			subDirt = L"cuda";
		loadDll(dll, subDirt);
	}
	SetDllDirectory(temp);
}

std::string getProgramPath() {
	char sz[512] = { 0 };
	HMODULE ihdll = GetModuleHandle(OEIP_MODEL_NAME);
	::GetModuleFileNameA(ihdll, sz, 512);
	::PathRemoveFileSpecA(sz);
	std::string path = sz;
	return path;
}

void loadAllDll() {
	std::vector<std::wstring> dllList = { L"oeip-win-dx11",L"oeip-win-cuda",L"oeip-win-mf",L"oeip-ffmpeg" };
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
			cleanPlugin(true);
			for (auto& dll : hdlls) {
				FreeLibrary(dll);
				dll = nullptr;
			}
			hdlls.clear();
			bLoad = false;
		}
	}
	return TRUE;
}

void loadMessage(int level, const char* message) {
	auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	struct tm t;   //tm结构指针
	localtime_s(&t, &now);   //获取当地日期和时间
	//用std::cout会导致UE4烘陪失败,记录下
	std::wstring wmessage = string2wstring(message);
	std::wcout << std::put_time(&t, L"%Y-%m-%d %X") << L" Level: " << level << L" " << wmessage << std::endl;
}
