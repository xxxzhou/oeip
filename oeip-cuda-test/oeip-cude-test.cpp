#include "trainDarknet.h"
#include "testdarknet.h"
#include "testgrabcut.h"
#include <Windows.h>
#include "../oeip/OeipCommon.h"

int main(int argc, char** argv)
{
	std::string path = getProgramPath();
	//让VS调试目录与双击运行目录一至
	SetCurrentDirectoryA(path.c_str());
	Grabcut::testGrabcut();
	//Grabcut::testSeedGrabCude();
	//DarknetPerson::testDarknetPerson();
	//trainDarknet();
	//validateDarknet();
	std::cin.get();
}