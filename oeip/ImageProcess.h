#pragma once

#include <string>
#include <vector>
#include "Oeip.h"
#include "BaseLayer.h"
#include "PluginManager.h"
#include <memory>
#include <mutex>

//各个input先设置大小setInput->initLayers->initbuffer
class OEIPDLL_EXPORT ImageProcess
{
public:
	ImageProcess() {};
	virtual ~ImageProcess() {};
protected:
	std::recursive_mutex mtx;
	int32_t runInit = 0;
	std::vector<std::shared_ptr<BaseLayer>> layers;
	onProcessHandle onProcessEvent;
	bool bInitLayer = false;
	bool bInitBuffer = false;
	//bool bRunFirst = false;
protected:
	//链接各层
	virtual bool onInitLayers() { return true; };
	virtual void onRunLayers() {};
	virtual BaseLayer* onAddLayer(OeipLayerType layerType) = 0;
public:
	bool initLayers();
	void runLayers();
	//updateLayer里根据参数决定是否需要重新初始化
	void resetLayers() {
		std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
		bInitLayer = false;
	};

	int32_t addLayer(const std::string& name, OeipLayerType layerType);
	template<typename T>
	int32_t addLayer(const std::string& name, OeipLayerType layerType, const T& t);
	template<typename T>
	bool updateLayer(int32_t index, const T& t);
public:
	int32_t findLayer(const std::string& name);
	void getLayerOutConnect(int32_t layerIndex, LayerConnect& outConnect, int32_t outIndex);
	void getLayerInConnect(int32_t layerIndex, LayerConnect& inConnect, int32_t inIndex);
public:
	//输入层大小改变后调用这个
	void setInput(int32_t layerIndex, int32_t width, int32_t height, int32_t dataType, int32_t inputIndex = 0);
	void updateInput(int32_t layerIndex, uint8_t* data, int32_t inputIndex = 0);
	void outputData(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t dataType);
	void setOutputGpuTex(int32_t layerIndex, void* device, void* tex, int32_t outputIndex = 0);
	void setDataProcess(onProcessHandle processHandle) {
		std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
		onProcessEvent = processHandle;
	}
};

template<typename T>
inline int32_t ImageProcess::addLayer(const std::string& name, OeipLayerType layerType, const T& t)
{
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	int32_t index = addLayer(name, layerType);
	updateLayer(index, t);
	return index;
};

template<typename T>
inline bool ImageProcess::updateLayer(int32_t index, const T& t)
{
	std::lock_guard<std::recursive_mutex> mtx_locker(mtx);
	if (index<0 || index > layers.size())
		return false;
	BaseLayerTemplate<T>* layer = dynamic_cast<BaseLayerTemplate<T>*>(layers[index].get());
	if (layer == nullptr) {
		std::string message = "update layer in:" + std::to_string(index) + " paramet no match " + typeid(T).name();
		logMessage(OEIP_WARN, message.c_str());
		return false;
	}
	layer->updateParamet(t);
	return true;
};

OEIPDLL_EXPORT void registerFactory(ObjectFactory<ImageProcess>* factory, int32_t type, std::string name);
