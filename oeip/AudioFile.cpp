/** @file AudioFile.cpp
*  @author Adam Stark
*  @copyright Copyright (C) 2017  Adam Stark
*/

#include "AudioFile.h"
#include <fstream>
#include <unordered_map>
#include <iterator>


AudioFile::AudioFile()
{
	bitDepth = 16;
	sampleRate = 44100;
	samples.resize(0);
}

AudioFile::~AudioFile()
{
	samples.clear();
}

bool AudioFile::writeHeader()
{
	std::vector<uint8_t> fileData;
	// -----------------------------------------------------------
	// HEADER CHUNK
	addStringToFileData(fileData, "RIFF");
	addInt32ToFileData(fileData, 0);
	addStringToFileData(fileData, "WAVE");

	// -----------------------------------------------------------
	// FORMAT CHUNK
	addStringToFileData(fileData, "fmt ");
	addInt32ToFileData(fileData, 16); // format chunk size (16 for PCM)
	addInt16ToFileData(fileData, 1); // audio format = 1
	addInt16ToFileData(fileData, (int16_t)channelCount); // num channels
	addInt32ToFileData(fileData, (int32_t)sampleRate); // sample rate

	int32_t numBytesPerSecond = (int32_t)((channelCount * sampleRate * bitDepth) / 8);
	addInt32ToFileData(fileData, numBytesPerSecond);

	int16_t numBytesPerBlock = channelCount * (bitDepth / 8);
	addInt16ToFileData(fileData, numBytesPerBlock);

	addInt16ToFileData(fileData, (int16_t)bitDepth);

	// -----------------------------------------------------------
	// DATA CHUNK
	addStringToFileData(fileData, "data");
	dataPositon = fileData.size();
	addInt32ToFileData(fileData, 0);
	// try to write the file
	return writeDataToFile(fileData, false);
}

bool AudioFile::writeData(uint8_t* data, int lenght)
{
	if (lenght == 0)
		return true;
	char* cdata = (char*)data;
	//vector<char> newSample(cdata, cdata + lenght);	
	//samples.insert(samples.end(), newSample.begin(), newSample.end());
	samples.insert(samples.end(), cdata, cdata + lenght);
	//这有个问题，音频采样每次大约只有5毫秒，IO写入的时间可能大于这个，就会造成采样时间变长，声音不对
	//当缓存的数据超过1M，写入文件
	if (samples.size() > 1024 * 1024 * 10) {
		std::vector<uint8_t> newSamples(samples.begin(), samples.end());
		samples.clear();
		writeDataToFile(newSamples, true, false);
	}
	return true;
}

void AudioFile::setAudioInfo(std::wstring path, int numChannels, int numBits, int rate)
{
	bInit = true;
	filePath = path;
	channelCount = numChannels;
	bitDepth = numBits;
	sampleRate = rate;
	writeHeader();
}

void AudioFile::close()
{
	writeDataToFile(samples, true, true);
	samples.clear();
	bInit = false;
}

bool AudioFile::writeDataToFile(std::vector<uint8_t>& fileData, bool bApp, bool bEnd)
{
	dataLenght += fileData.size();
	//app是追加，只能在末尾写，想后期修改，必需用std::ios::in | std::ios::ate，单独的ate会删除文件
	auto fileMask = (std::ios::binary | std::ios::in | std::ios::ate);
	//从头开始写
	if (!bApp)
		fileMask = std::ios::binary;
	std::ofstream outputFile(filePath, fileMask);
	if (outputFile.is_open()) {
		//outputFile.seekp(0, std::ios::end);
		int length = outputFile.tellp();
		if (fileData.size() > 0) {
			outputFile.write((char*)fileData.data(), fileData.size());
		}
		if (bEnd) {
			length = outputFile.tellp();

			outputFile.seekp(4, std::ios::beg);
			std::vector<uint8_t> streamlength;
			addInt32ToFileData(streamlength, length - 8);
			outputFile.write((char*)streamlength.data(), 4);

			outputFile.seekp(dataPositon, std::ios::beg);
			std::vector<uint8_t> chunlength;
			addInt32ToFileData(chunlength, length - (4 + 24 + 8 + 8));
			outputFile.write((char*)chunlength.data(), 4);

			outputFile.seekp(0, std::ios::end);
			length = outputFile.tellp();
		}
		outputFile.close();
		return true;
	}
	return false;
}

//=============================================================

void AudioFile::addStringToFileData(std::vector<uint8_t>& fileData, std::string s)
{
	for (int i = 0; i < s.length(); i++)
		fileData.push_back((uint8_t)s[i]);
}

//=============================================================

void AudioFile::addInt32ToFileData(std::vector<uint8_t>& fileData, int32_t i, Endianness endianness)
{
	uint8_t bytes[4];
	if (endianness == Endianness::LittleEndian) {
		bytes[3] = (i >> 24) & 0xFF;
		bytes[2] = (i >> 16) & 0xFF;
		bytes[1] = (i >> 8) & 0xFF;
		bytes[0] = i & 0xFF;
	}
	else {
		bytes[0] = (i >> 24) & 0xFF;
		bytes[1] = (i >> 16) & 0xFF;
		bytes[2] = (i >> 8) & 0xFF;
		bytes[3] = i & 0xFF;
	}
	for (int i = 0; i < 4; i++)
		fileData.push_back(bytes[i]);
}

//=============================================================

void AudioFile::addInt16ToFileData(std::vector<uint8_t>& fileData, int16_t i, Endianness endianness)
{
	uint8_t bytes[2];

	if (endianness == Endianness::LittleEndian) {
		bytes[1] = (i >> 8) & 0xFF;
		bytes[0] = i & 0xFF;
	}
	else {
		bytes[0] = (i >> 8) & 0xFF;
		bytes[1] = i & 0xFF;
	}
	fileData.push_back(bytes[0]);
	fileData.push_back(bytes[1]);
}

//=============================================================

void AudioFile::clearAudioBuffer()
{
	samples.clear();
}

