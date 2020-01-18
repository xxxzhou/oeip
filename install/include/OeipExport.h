#pragma once
#include "Oeip.h"

//__cdecl 默认C/C++语言调用规则 调用者来清栈 UE4
//__stdcall 默认C#调用规则 被调用的函数自己清栈 Unity

//template<typename T>
//bool updatePipeParamet(int32_t pipeId, int32_t layerIndex, const T& t);
//实例化如下几个函数，用于导出时生成相应接口(具体化可以针对特定类实现不同模板的逻辑)
//template OEIPDLL_EXPORT bool updatePipeParamet<InputParamet>(int32_t pipeId, int32_t layerIndex, const InputParamet& paramet);
//template OEIPDLL_EXPORT bool updatePipeParamet<YUV2RGBAParamet>(int32_t pipeId, int32_t layerIndex, const YUV2RGBAParamet& paramet);

extern "C"
{
	//回调action后缀的给c#使用，打印oeip产生的各种日志信息
	OEIPDLL_EXPORT void setLogAction(logEventAction logHandle);
	//回调handle后缀的给c++使用
	OEIPDLL_EXPORT void setLogHandle(logEventHandle logHandle);
	//初始化OEIP环境
	OEIPDLL_EXPORT void initOeip();
	//销毁OEIP产生的各种资源
	OEIPDLL_EXPORT void shutdownOeip();
	//根据设备的视频类型返回对应YUV类型，如果非YUV类型，返回OEIP_YUVFMT_OTHER
	OEIPDLL_EXPORT OeipYUVFMT getVideoYUV(OeipVideoType videoType);
	//相应颜色参数一般用uint32_t来表示，用来给用户根据各个通道分量生成uint32_t颜色，通道分量范围0.f-1.f
	OEIPDLL_EXPORT uint32_t getColor(float r, float g, float b, float a);

#pragma region camera device 
	//得到支持的捕获视频设备数量(主要包含webCamera,decklink)
	OEIPDLL_EXPORT int32_t getDeviceCount();
	//得到捕获视频设备列表，其中deviceList为传入的列表指针,lenght为上面getDeviceCount返回的数量
	OEIPDLL_EXPORT void getDeviceList(OeipDeviceInfo* deviceList, int32_t lenght, int32_t index = 0);
	//捕获视频设备的图像格式数量
	OEIPDLL_EXPORT int32_t getFormatCount(int32_t deviceIndex);
	//捕获视频设备的图像格式列表
	OEIPDLL_EXPORT int32_t getFormatList(int32_t deviceIndex, VideoFormat* formatList, int32_t lenght, int32_t index = 0);
	//得到捕获视频设备当前所用的图像格式索引
	OEIPDLL_EXPORT int32_t getFormat(int32_t deviceIndex);
	//捕获视频设备设置对应格式
	OEIPDLL_EXPORT void setFormat(int32_t deviceIndex, int32_t formatIndex);
	//运行设备
	OEIPDLL_EXPORT bool openDevice(int32_t deviceIndex);
	//关闭设备
	OEIPDLL_EXPORT void closeDevice(int32_t deviceIndex);
	//设备是否打开状态中
	OEIPDLL_EXPORT bool bOpen(int32_t deviceIndex);
	//查找一下最适合传入的长度格式的索引,需要注意并不完全对应传入的长宽,想要相应输出图像，需要自己加入Resize层
	OEIPDLL_EXPORT int32_t findFormatIndex(int32_t deviceIndex, int32_t width, int32_t height, int32_t fps = 30);
	//得到WebCamera的如曝光，焦距等等参数
	OEIPDLL_EXPORT void getDeviceParametrs(int32_t deviceIndex, CamParametrs* parametrs);
	//设置WebCamera的如曝光，焦距等等参数
	OEIPDLL_EXPORT void setDeviceParametrs(int32_t deviceIndex, const CamParametrs* parametrs);
	//设置捕获视频设备每桢处理完后的数据回调，回调包含长，宽，数据指针，对应数据输出类型,用于C/C#使用
	OEIPDLL_EXPORT void setDeviceDataAction(int32_t deviceIndex, onReviceAction onProcessData);
	//设置捕获视频设备每桢处理完后的数据回调，回调包含长，宽，数据指针，对应数据输出类型。用于C++使用。
	OEIPDLL_EXPORT void setDeviceDataHandle(int32_t deviceIndex, onReviceHandle onProcessData);
	//设置捕获视频设备事件回调，如没有正常打开,意外断掉等。用于C/C#使用
	OEIPDLL_EXPORT void setDeviceEventAction(int32_t deviceIndex, onEventAction onDeviceEvent);
	//设置捕获视频设备事件回调，如没有正常打开,意外断掉等。用于C++使用
	OEIPDLL_EXPORT void setDeviceEventHandle(int32_t deviceIndex, onEventHandle onDeviceEvent);
#pragma endregion

#pragma region audio
	//返回麦或声卡的原始数据
	OEIPDLL_EXPORT void setAudioOutputHandle(onAudioOutputHandle outDatahandle);
	//开始采集麦与声卡，并返回麦与声卡的经给定格式的重采样输出,如果麦与声卡都为true,则混合输出
	OEIPDLL_EXPORT void startAudioOutput(bool bMic, bool bLoopback, OeipAudioDesc desc, onAudioDataHandle dataHandle);
	//关闭采集麦与声卡
	OEIPDLL_EXPORT void closeAudioOutput();
#pragma endregion

#pragma region gpgpu pipe
	//初始化一个GPU计算管线
	OEIPDLL_EXPORT int32_t initPipe(OeipGpgpuType gpgpuType);
	//释放一个GPU计算管线,相应pipeId在没有再次initPipe得到,不能调用下面运用管线的API
	OEIPDLL_EXPORT bool closePipe(int32_t pipeId);
	//管线添加一层,paramet表示管线对应的参数结构,请传递对应结构
	OEIPDLL_EXPORT int32_t addPiepLayer(int32_t pipeId, const char* layerName, OeipLayerType layerType, const void* paramet = nullptr);
	//设定连接层级,一般跨级连接调用，默认下层连接上层
	OEIPDLL_EXPORT void connectLayerName(int32_t pipeId, int32_t layerIndex, const char* forwardName, int32_t inputIndex = 0, int32_t selfIndex = 0);
	//设定连接层级,一般跨级连接调用，默认下层连接上层
	OEIPDLL_EXPORT void connectLayerIndex(int32_t pipeId, int32_t layerIndex, int32_t forwardIndex, int32_t inputIndex = 0, int32_t selfIndex = 0);
	//设定当前层是否可用
	OEIPDLL_EXPORT void setEnableLayer(int32_t pipeId, int32_t layerIndex, bool bEnable);
	//设定当前层及关联这层的分支全部不可用
	OEIPDLL_EXPORT void setEnableLayerList(int32_t pipeId, int32_t layerIndex, bool bEnable);
	//设置计算管线处理完后的数据回调，回调包含长，宽，数据指针，对应数据输出类型,用于C/C#使用
	OEIPDLL_EXPORT void setPipeDataAction(int32_t pipeId, onProcessAction onProcessData);
	//设置计算管线处理完后的数据回调，回调包含长，宽，数据指针，对应数据输出类型。用于C++使用。
	OEIPDLL_EXPORT void setPipeDataHandle(int32_t pipeId, onProcessHandle onProcessData);
	//设置计算管线的输入
	OEIPDLL_EXPORT void setPipeInput(int32_t pipeId, int32_t layerIndex, int32_t width, int32_t height, int32_t dataType = OEIP_CV_8UC1, int32_t inputIndex = 0);
	//更新计算管线的数据输入
	OEIPDLL_EXPORT void updatePipeInput(int32_t pipeId, int32_t layerIndex, uint8_t* data, int32_t inputIndex = 0);
	//运行管线,如果可能,尽量与updatePipeParamet在同一线程,最好是UE4/Unity3D的主线程(游戏线程),或是WinForm里的UI线程
	OEIPDLL_EXPORT void runPipe(int32_t pipeId);
	//把另一个DX11上下文中的纹理当做当前管线的输入源
	OEIPDLL_EXPORT void setPipeInputGpuTex(int32_t pipeId, int32_t layerIndex, void* device, void* tex, int32_t inputIndex = 0);
	//把当前管线的输出结果直接放入另一个DX11上下文的纹理中,因为相应的纹理一般在单独的渲染线程，回调需要切换线程，固让用户根据需要在对应渲染线程获取
	OEIPDLL_EXPORT void setPipeOutputGpuTex(int32_t pipeId, int32_t layerIndex, void* device, void* tex, int32_t outputIndex = 0);
	//更新当前层的参数，需要注意paramet是当前层的参数结构，不同会引发想不到的问题
	OEIPDLL_EXPORT bool updatePipeParamet(int32_t pipeId, int32_t layerIndex, const void* paramet);
#pragma endregion
}


