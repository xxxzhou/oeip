#include "AudioRecord.h"



AudioRecord::AudioRecord()
{
}


AudioRecord::~AudioRecord() 
{

}

void registerFactory(ObjectFactory<AudioRecord>* factory, int32_t type, std::string name)
{
	PluginManager<AudioRecord>::getInstance().registerFactory(factory, type, name);
}