// oeip-test.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <iostream>
#include "testcamer.h"
#include "templatetest.h"
#include "testaudio.h"
#include <mutex>
#include "../oeip-win/DxHelper.h"
#include "testencodervideo.h"
#include "testencoderaudio.h"
#include "testliveoutput.h"

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define new  new(_CLIENT_BLOCK, __FILE__, __LINE__)
//
int main() {
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
	initOeip();
	std::cout << "Hello World!\n";

	//OeipCamera::testCamera();
	//OeipAudio::test();
	//OeipEncoderVideo::test();
	//OeipEncoderAudio::test();
	OeipLiveOutput::test();

	std::cout << "close!\n";
	shutdownOeip();
	_CrtDumpMemoryLeaks();
	return 0;
}

//int main() {
//	//OeipCamera::testCamera();
//	//OeipAudio::test();
//	//OeipEncoderVideo::test();
//	//OeipEncoderAudio::test();
//	OeipLiveOutput::test();
//	std::cout << "Hello World!\n";
//}


