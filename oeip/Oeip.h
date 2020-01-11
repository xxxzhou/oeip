#pragma once

#ifdef OEIP_EXPORT 
#define OEIPDLL_EXPORT __declspec(dllexport) 
#else
#define OEIPDLL_EXPORT __declspec(dllimport)
#endif

#include "OeipDefine.h"
#include <functional>

enum OeipLogLevel : int32_t
{
	OEIP_INFO,
	OEIP_WARN,
	OEIP_ERROR,
	OEIP_ALORT,
};

enum OeipVideoType : int32_t
{
	OEIP_VIDEO_OTHER,
	OEIP_VIDEO_NV12,//OEIP_YUVFMT_YUV420SP
	OEIP_VIDEO_YUY2,//OEIP_YUVFMT_YUY2I
	OEIP_VIDEO_YVYU,//OEIP_YUVFMT_YVYUI
	OEIP_VIDEO_UYVY,//OEIP_YUVFMT_UYVYI
	//设定自动编码成OEIP_VIDEO_YUY2
	OEIP_VIDEO_MJPG,
	OEIP_VIDEO_RGB24,
	OEIP_VIDEO_ARGB32,
	OEIP_VIDEO_RGBA32,
	OEIP_VIDEO_DEPTH,//U16
};

//Planar(YUV各自分开)Semi-Planar(Y单独分开,UV交并)Interleaved(YUV交并)
enum OeipYUVFMT : int32_t
{
	OEIP_YUVFMT_OTHER,
	OEIP_YUVFMT_YUV420SP,//Semi-Planar 一般用于图像设备 NV12
	OEIP_YUVFMT_YUY2I,//Interleaved 一般用于图像设备
	OEIP_YUVFMT_YVYUI,//Interleaved 一般用于图像设备
	OEIP_YUVFMT_UYVYI,//Interleaved 一般用于图像设备
	OEIP_YUVFMT_YUY2P,//Planar 一般用于传输
	OEIP_YUVFMT_YUV420P,//Planar 一般用于传输
};

//添加相关层需要在BaseLayer::updateParamet里注册下
enum OeipLayerType : int32_t
{
	OEIP_NONE_LAYER,
	OEIP_INPUT_LAYER,
	OEIP_OUTPUT_LAYER,
	OEIP_YUV2RGBA_LAYER,
	OEIP_MAPCHANNEL_LAYER,
	OEIP_RGBA2YUV_LAYER,
	OEIP_RESIZE_LAYER,
	OEIP_OPERATE_LAYER,
	OEIP_BLEND_LAYER,
	OEIP_GUIDEDFILTER_LAYER,
	//此层计划只有CUDA的实现，DX11实现太麻烦了
	OEIP_GRABCUT_LAYER,
	OEIP_DARKNET_LAYER,
	OEIP_MAX_LAYER,
};

#define AllLayerParamet int32_t, InputParamet, OutputParamet,\
	YUV2RGBAParamet, MapChannelParamet, RGBA2YUVParamet,\
	ResizeParamet,OperateParamet,BlendParamet,\
	GuidedFilterParamet,GrabcutParamet, DarknetParamet,void

enum OeipGpgpuType : int32_t
{
	OEIP_GPGPU_OTHER,
	OEIP_DX11,
	OEIP_CUDA,
	OEIP_Vulkun,
};

enum OeipDeviceEventType :int32_t
{
	OEIP_Event_Other,
	//关闭成功
	OEIP_DeviceStop,
	//打开成功
	OEIP_DeviceOpen,
	//打开失败
	OEIP_DeviceNoOpen,
	//掉线
	OEIP_DeviceDropped,
};

enum VideoDeviceType : int32_t
{
	OEIP_VideoDevice_Other,
	OEIP_MF,
	OEIP_Decklink,
	OEIP_Realsense,
	OEIP_Virtual,
};

enum OeipAudioDataType : int32_t
{
	//静音
	OEIP_AudioData_None = 0,
	OEIP_Audio_Data,
	OEIP_Audio_WavHeader,
};

//Paramet结构里不用一字节类型数据，一是memcmp后面会用，
//二是C#/C++里bool都只有一字节，但是可能因为不同对齐方式导致差异，故与C#交互的结构bool全使用int32
//默认从CPU输入,如果要支持GPU输入,bGpu=true
struct InputParamet
{
	int32_t bCpu = true;
	int32_t bGpu = false;
};

struct OutputParamet
{
	int32_t bCpu = true;
	int32_t bGpu = true;
};

struct YUV2RGBAParamet
{
	OeipYUVFMT yuvType = OEIP_YUVFMT_YUV420SP;
};

struct RGBA2YUVParamet
{
	OeipYUVFMT yuvType = OEIP_YUVFMT_YUV420SP;
};

//ARGB<->BGRA<->RGBA<->RRRR
struct MapChannelParamet
{
	uint32_t red = 0;
	uint32_t green = 1;
	uint32_t blue = 2;
	uint32_t alpha = 3;
};

struct ResizeParamet
{
	int32_t bLinear = true;
	int32_t width = 1920;
	int32_t height = 1080;
};

//方便C#交互不做额外设置，以及GPU参数结构对应,bool全用int表示
struct OperateParamet
{
	int32_t bFlipX = false;
	int32_t bFlipY = false;
	float gamma = 1.f;
};

//二图混合，第二图显示在第一图中下面前四个参数组成的RECT中
struct BlendParamet
{
	//所有值范围在0.1
	float left = 0.f;
	float top = 0.f;
	float width = 0.f;
	float height = 0.f;
	//不透明度
	float opacity = 0.f;
};

struct GuidedFilterParamet
{
	//导向滤波的特性，以缩放后的图像处理能快速得到一样结果
	int32_t zoom = 8;
	int32_t	softness = 5;
	float eps = 0.00001f;
	float intensity = 0.2f;
};

struct GrabcutParamet
{
	int32_t iterCount = 1;
	float gamma = 90.f;
	float lambda = 450.f;
	int32_t count = 250;
};

struct DarknetParamet
{
	//控制网络是否加载
	int32_t bLoad = false;
	char confile[512];
	char weightfile[512];
	//框的显示阀值(越大则要求准确性越高)
	float thresh = 0.3f;
	//交并多少合并(越大则有重合的框容易合成一个)
	float nms = 0.4f;
	int32_t bDraw = false;
	uint32_t drawColor = 255;
};

struct PersonBox
{
	//是人的概率
	float prob = 0.f;
	float centerX = 0.f;
	float centerY = 0.f;
	float width = 0.f;
	float height = 0.f;
};

struct VideoFormat
{
	int32_t index = -1;
	int32_t width = 0;
	int32_t height = 0;
	OeipVideoType videoType = OEIP_VIDEO_OTHER;
	int32_t fps = 0;
};

struct Parametr
{
	long CurrentValue;
	long Min;
	long Max;
	long Step;
	long Default;
	long Flag;
};

struct CamParametrs
{
	Parametr Brightness;
	Parametr Contrast;
	Parametr Hue;
	Parametr Saturation;
	Parametr Sharpness;
	Parametr Gamma;
	Parametr ColorEnable;
	Parametr WhiteBalance;
	Parametr BacklightCompensation;
	Parametr Gain;

	Parametr Pan;
	Parametr Tilt;
	Parametr Roll;
	Parametr Zoom;
	Parametr Exposure;
	Parametr Iris;
	Parametr Focus;
};

struct OeipDeviceInfo
{
	int32_t id = -1;
	wchar_t deviceName[512];
	wchar_t deviceId[512];
};

//声音与图像类似,对应硬件捕获设备一般用交差格式,而传输一般用平面格式
struct OeipAudioDesc
{
	int32_t channel;//1,2 
	int32_t sampleRate;//8000,11025,22050,44100
	int32_t bitSize;//16，24，32 

	bool operator == (OeipAudioDesc& rhs) const {
		return (channel == rhs.channel &&
			sampleRate == rhs.sampleRate &&
			bitSize == rhs.bitSize);
	};
};

struct OeipVideoDesc
{
	int32_t width;
	int32_t height;
	int32_t fps;
	OeipVideoType videoType;
};

struct OeipDateDesc
{
	int32_t elementSize;
	int32_t elementChannel;
};

//日志回调
typedef void(*logEventAction)(int32_t level, const char* message);
typedef std::function<void(int32_t, const char*)> logEventHandle;

//原始声音回调
typedef void(*onAudioRecordAction)(uint8_t* data, int32_t dataLen, OeipAudioDataType dataType);
typedef std::function<void(uint8_t*, int32_t, OeipAudioDataType)> onAudioRecordHandle;
//原始声音回调，指明是否是麦
typedef std::function<void(bool bMic, uint8_t * data, int32_t size, OeipAudioDataType type)> onAudioOutputHandle;
//处理后声音回调(如重采样,混音)
typedef std::function<void(uint8_t * data, int32_t size)> onAudioDataHandle;

//设备事件，如中断等
typedef void(*onEventAction)(int32_t type, int32_t code);
typedef std::function<void(int32_t, int32_t)> onEventHandle;

//摄像机的数据处理回调，dataType指明data数据类型
typedef void(*onReviceAction)(uint8_t* data, int32_t width, int32_t height);
typedef std::function<void(uint8_t*, int32_t, int32_t)> onReviceHandle;

//GPU运算管线返回
typedef void(*onProcessAction)(int32_t layerIndex, uint8_t* data, int32_t width, int32_t height, int32_t outputIndex);
typedef std::function<void(int32_t, uint8_t*, int32_t, int32_t, int32_t)> onProcessHandle;