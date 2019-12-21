#pragma once

#include "OeipCommon.h"
#include <vector>

template<typename T>
class ObjectFactory {
public:
	virtual T* create(int type) = 0;
};

template<typename T>
class PluginManager
{
	struct FactoryIndex
	{
		ObjectFactory<T>* factory = nullptr;
		int factoryType = -1;
		std::vector<T*> models;
		std::string name;
	};
public:
	static PluginManager<T>& getInstance()
	{
		static PluginManager m_instance;
		return m_instance;
	};
	~PluginManager()
	{
		//Release();
	};
private:
	//static PluginManager<T>* instance;
	PluginManager() {};
private:
	std::vector<FactoryIndex> factorys;
public:
	//注册生产类,type不能设置为-1,-1表示特殊意义	
	void registerFactory(ObjectFactory<T>* vpf, int type, std::string name)
	{
		bool bhave = false;
		for (const auto& fi : factorys) {
			if (fi.factoryType == type) {
				std::string message = std::to_string(type) + " " + name + " have register.";
				logMessage(OEIP_ERROR, message.c_str());
				bhave = true;
				break;
			}
		}
		if (!bhave) {
			FactoryIndex fi = {};
			fi.factory = vpf;
			fi.factoryType = type;
			fi.name = name;
			factorys.push_back(fi);
		}
	};
	//产生一个实体	
	T* createModel(int type)
	{
		for (auto& fi : factorys) {
			if (fi.factoryType == type) {
				T* model = fi.factory->create(type);
				fi.models.push_back(model);
				return model;
			}
		}
		return nullptr;
	};

	bool bHaveType(int type)
	{
		bool bHave = false;
		for (auto& fi : factorys) {
			if (fi.factoryType == type) {
				bHave = true;
				break;
			}
		}
		return bHave;
	}

	//vtype取负一，表示所有model
	void getModelList(std::vector<T*>& models, int vtype)
	{
		for (auto& fi : factorys) {
			if (vtype == -1 || fi.factoryType == vtype) {
				models.insert(models.end(), fi.models.begin(), fi.models.end());
			}
		}
	};

	//一般注册的如果是管理类(单例),调用此方法默认创建一个实例
	void getFactoryDefaultModel(std::vector<T*>& models, int vtype)
	{
		for (auto& fi : factorys) {
			if (vtype == -1 || fi.factoryType == vtype) {
				if (fi.models.size() == 0)
					createModel(fi.factoryType);
				models.insert(models.end(), fi.models.begin(), fi.models.end());
			}
		}
	};
	//UE4/Unity3D编辑器时每次Play/EndPlay调用
	void release()
	{
		for (auto& fi : factorys) {
			for (auto& model : fi.models) {
				safeDelete(model);
			}
			fi.models.clear();
			//safeDelete(fi.factory);
		}
	};
	//在引用oeip进程关闭时调用
	static void clean(bool bFactory = false)
	{
		getInstance().release();
		if (bFactory) {
			for (auto& fi : getInstance().factorys)
			{
				safeDelete(fi.factory);
			}
			getInstance().factorys.clear();
		}
	}

	std::string getTypeDesc(int type)
	{
		for (auto& fi : factorys) {
			if (fi.factoryType == type) {
				return fi.name;
			}
		}
		return "";
	}
};


template<typename T>
void registerFactory(ObjectFactory<T>* factory, int32_t type, std::string name)
{
	PluginManager<T>::getInstance().registerFactory(factory, type, name);
}
