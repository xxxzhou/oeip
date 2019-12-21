#pragma once
#include "Oeip.h"
#include "PluginManager.h"

class OEIPDLL_EXPORT AudioRecord
{
public:
	AudioRecord();
	virtual ~AudioRecord();
public:
	virtual bool initRecord() { return false; };
public:
	virtual bool initRecord(OeipAudioRecordType audiotype, onAudioRecordHandle handle) { return false; };
	virtual bool initRecord(OeipAudioRecordType audiotype, std::wstring path) { return false; };
	virtual void close() {};
};

template OEIPDLL_EXPORT void registerFactory<AudioRecord>(ObjectFactory<AudioRecord>* factory, int32_t type, std::string name);



