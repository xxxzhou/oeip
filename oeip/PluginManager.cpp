#include "PluginManager.h"
#include <string>
#include <stdio.h>
#include <ctime>
#include <iomanip>
#include <Shlwapi.h>
#include "OeipCommon.h"
#include "OeipManager.h"
#pragma comment(lib,"shlwapi.lib")

//static HINSTANCE hdll = nullptr;
static bool bLoad = false;
static std::vector<HINSTANCE> hdlls;

typedef bool(*bCanLoad)();
typedef void(*registerfactory)();

void loadDll(std::wstring dllName, std::wstring subDirt)
{
	HINSTANCE hdll = LoadLibraryEx(dllName.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
	std::string sdname = wstring2string(dllName);
	if (hdll == nullptr) {
		if (!subDirt.empty()) {
			wchar_t temp[512] = { 0 };
			GetDllDirectory(512, temp);
			wchar_t sz[512] = { 0 };
			HMODULE ihdll = GetModuleHandle(L"zmf.dll");
			::GetModuleFileName(ihdll, sz, 512);
			::PathRemoveFileSpec(sz);
			::PathAppend(sz, subDirt.c_str());
			SetDllDirectory(sz);//SetCurrentDirectory(sz);
			hdll = LoadLibraryEx(dllName.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
			SetDllDirectory(temp);
		}
		if (hdll == nullptr) {
			DWORD error_id = GetLastError();
			std::string message = "load dll:" + sdname + " error-" + std::to_string(error_id);
			logMessage(OEIP_ERROR, message.c_str());
		}
	}
	if (hdll) {
		bCanLoad bcd = (bCanLoad)GetProcAddress(hdll, "bCanLoad");
		if (bcd && bcd()) {
			registerfactory rf = (registerfactory)GetProcAddress(hdll, "registerFactory");
			if (rf)
				rf();
			std::string message = "load dll:" + sdname + " sucess.";
			logMessage(OEIP_INFO, message.c_str());
			hdlls.push_back(hdll);
		}
		else {
			FreeLibrary(hdll);
			hdll = nullptr;
			std::string message = "dll:" + sdname + " loading conditions do not match.";
			logMessage(OEIP_ERROR, message.c_str());
		}
	}
}

void loadDllArray(std::vector<std::wstring> dllNames)
{
	wchar_t temp[512] = { 0 };
	GetDllDirectory(512, temp);
	wchar_t sz[512] = { 0 };
	HMODULE ihdll = GetModuleHandle(L"zmf.dll");
	::GetModuleFileName(ihdll, sz, 512);
	::PathRemoveFileSpec(sz);
	SetDllDirectory(sz);
	for (auto dll : dllNames)
	{
		std::wstring subDirt = L"";
		if (dll == L"oeip-win-cuda")
			subDirt = L"cuda";
		loadDll(dll, subDirt);
	}
	SetDllDirectory(temp);
}

std::string getProgramPath()
{
	char sz[512] = { 0 };
	HMODULE ihdll = GetModuleHandle(L"oeip.dll");
	::GetModuleFileNameA(ihdll, sz, 512);
	::PathRemoveFileSpecA(sz);
	std::string path = sz;
	return path;
}
//vector<wstring> videoTypeList = { L"MFVideoFormat_NV12" ,L"MFVideoFormat_YUY2"

void loadAllDll()
{
	std::vector<std::wstring> dllList = { L"oeip-win-dx11",L"oeip-win-cuda",L"oeip-video-mf",L"oeip-video-realsense",L"oeip-video-decklink" };
	loadDllArray(dllList);
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD  dwReason, LPVOID lpReserved)
{
	if (dwReason == DLL_PROCESS_ATTACH) {
		if (!bLoad) {
			loadAllDll();
			bLoad = true;
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
	else if (dwReason == DLL_THREAD_ATTACH) {
	}
	return TRUE;
}