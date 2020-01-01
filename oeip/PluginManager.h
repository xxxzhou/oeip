#pragma once

#include "OeipCommon.h"
#include <vector>
#include <memory>

//在加载dll时打印输出信息，Unity3D/UE4可能会引起问题
#define OEIP_LOADDLL_OUTPUT 1
#if OEIP_LOADDLL_OUTPUT
void loadMessage(int level, const char* message);
#endif

template<typename T>
class ObjectFactory {
public:
	virtual T* create(int type) = 0;
};

template<typename T>
class PluginManager
{
	struct FactoryIndex {
		ObjectFactory<T>* factory = nullptr;
		//std::unique_ptr< ObjectFactory<T>> factory = nullptr;
		int32_t factoryType = -1;
		std::vector<T*> models;
		//std::vector<std::shared_ptr<T>> models;
		std::string name;
	};
public:
	static PluginManager<T>& getInstance() {
		static PluginManager m_instance;
		return m_instance;
	};
	~PluginManager() {
		//Release();
	};
private:
	//static PluginManager<T>* instance;
	PluginManager() {};
private:
	std::vector<FactoryIndex> factorys;
public:
	//注册生产类,type不能设置为-1,-1表示特殊意义	
	void registerFactory(ObjectFactory<T>* vpf, int32_t type, std::string name) {
		bool bhave = false;
		for (const auto& fi : factorys) {
			if (fi.factoryType == type) {
				std::string message = std::to_string(type) + " " + name + " have register.";
				loadMessage(OEIP_ERROR, message.c_str());
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
			std::string message = std::to_string(type) + " " + name + " register.";
			loadMessage(OEIP_INFO, message.c_str());
		}
	};
	//产生一个实体	
	T* createModel(int32_t type) {
		for (auto& fi : factorys) {
			if (fi.factoryType == type) {
				T* model = fi.factory->create(type);
				//std::shared_ptr<T> model(fi.factory->create(type));
				fi.models.push_back(model);
				return model;
			}
		}
		return nullptr;
	};

	bool bHaveType(int32_t type) {
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
	void getModelList(std::vector<T*>& models, int32_t vtype) {
		for (auto& fi : factorys) {
			if (vtype == -1 || fi.factoryType == vtype) {
				models.insert(models.end(), fi.models.begin(), fi.models.end());
			}
		}
	};

	//一般注册的如果是管理类(单例),调用此方法默认创建一个实例
	void getFactoryDefaultModel(std::vector<T*>& models, int32_t vtype) {
		for (auto& fi : factorys) {
			if (vtype == -1 || fi.factoryType == vtype) {
				if (fi.models.size() == 0)
					createModel(fi.factoryType);
				models.insert(models.end(), fi.models.begin(), fi.models.end());
			}
		}
	};
	//UE4/Unity3D编辑器时每次Play/EndPlay调用
	void release() {
		for (auto& fi : factorys) {
			for (auto& model : fi.models) {
				//safeDelete(model);
			}
			fi.models.clear();
			//safeDelete(fi.factory);
		}
	};
	//在引用oeip进程关闭时调用
	static void clean(bool bFactory = false) {
		getInstance().release();
		if (bFactory) {
			for (auto& fi : getInstance().factorys) {
				safeDelete(fi.factory);
			}
			getInstance().factorys.clear();
		}
	}

	std::string getTypeDesc(int type) {
		for (auto& fi : factorys) {
			if (fi.factoryType == type) {
				return fi.name;
			}
		}
		return "";
	}
};


template<typename T>
void registerFactory(ObjectFactory<T>* factory, int32_t type, std::string name) {
	PluginManager<T>::getInstance().registerFactory(factory, type, name);
}

//导出公共类给插件层实现
#define OEIP_DEFINE_PLUGIN_TYPE(model) \
	template OEIPDLL_EXPORT void registerFactory<model>(ObjectFactory<model>* factory, int32_t type, std::string name);

//抽象一个公共的子类插件声明
#define OEIP_DEFINE_PLUGIN_CLASS(model,childModel) \
	class childModel##Factory :public ObjectFactory<model> \
	{ \
		public: \
			childModel##Factory() {};\
			~childModel##Factory() {};\
		public:\
			virtual model* create(int type) override{;\
				childModel *md = new childModel();\
				return md; \
			}; \
	}; 