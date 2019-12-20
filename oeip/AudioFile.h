#pragma once

/*
@author Adam Stark
*/
//https://github.com/adamstark/AudioFile

#include <iostream>
#include <vector>
#include <assert.h>
#include <string>
#include "Oeip.h"

//PCM音频写入文件，为了能长期保存数据，加上数据缓存，大于某值后写入一次
class OEIPDLL_EXPORT AudioFile
{
public:
	AudioFile();
	~AudioFile();
	bool writeData(uint8_t* data, int lenght);
	void setAudioInfo(std::wstring path, int numChannels, int numBits, int sampleRate);
	void close();
	bool IsInit() { return bInit; } 
private:
	enum class Endianness
	{
		LittleEndian,
		BigEndian
	};
	void clearAudioBuffer();

	void addStringToFileData(std::vector<uint8_t>& fileData, std::string s);
	void addInt32ToFileData(std::vector<uint8_t>& fileData, int32_t i, Endianness endianness = Endianness::LittleEndian);
	void addInt16ToFileData(std::vector<uint8_t>& fileData, int16_t i, Endianness endianness = Endianness::LittleEndian);

	bool writeHeader();
	bool writeDataToFile(std::vector<uint8_t>& fileData, bool bApp, bool bEnd = false);

	uint32_t sampleRate;
	int32_t bitDepth;
	int channelCount;
	std::wstring filePath;
	std::vector<uint8_t> samples;
	int32_t dataPositon = 0;
	int32_t dataLenght = 0;
	bool bInit = false;
};

