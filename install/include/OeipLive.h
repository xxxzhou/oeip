#pragma once
#include <Oeip.h>

#ifdef OEIPLIVE_EXPORTS 
#define OEIPLIVEDLL_EXPORT __declspec(dllexport) 
#else
#define OEIPLIVEDLL_EXPORT __declspec(dllimport)
#endif

enum OeipLiveMode : int32_t
{
	OIEP_FFMPEG,
};

enum OeipLiveStatus : int32_t
{
	OEIP_LIVE_UNINIT,
	OEIP_LIVE_INIT,
	OEIP_LIVE_LOGING,
	OEIP_LIVE_LOGIN,
	OEIP_LIVE_LOGOUTING,
	OEIP_LIVE_LOGOUT,
};

enum OeipLiveOperate : int32_t
{
	OEIP_LIVE_OPERATE_NONE,
	//初始化
	OEIP_LIVE_OPERATE_INIT,
	//是否已经得到媒体服务器地址
	OEIP_LIVE_OPERATE_MEDIASERVE,
	//推流与拉流打开
	OEIP_LIVE_OPERATE_OPEN,
	OEIP_LIVE_OPERATE_CLOSE,
};

struct OeipLiveContext
{
	OeipLiveMode liveMode = OIEP_FFMPEG;
	//是否采集声卡
	int32_t bLoopback = false;
	char liveServer[512];
};

struct OeipVideoEncoder
{
	int32_t width = 1920;
	int32_t height = 1080;
	int32_t fps = 30;
	int32_t bitrate = 4000000;
	OeipYUVFMT yuvType = OEIP_YUVFMT_YUY2P;
};

struct OeipAudioEncoder
{
	int32_t frequency = 32000;
	int32_t channel = 1;
	int32_t bitrate = 48000;
};

//用于传输，用data[4]而不是data,主要是为了整合别的直播SDK考虑
//同理，data[4]直接拿到就是内存地址，在回调要么copy内存保存，要么直接处理完，不要保存地址
struct OeipVideoFrame
{
	//传输一般用平面YUV格式
	uint8_t* data[4];
	uint32_t dataSize;
	uint64_t timestamp;
	uint32_t width;
	uint32_t height;
	OeipYUVFMT fmt = OEIP_YUVFMT_YUY2P;
};

struct OeipAudioFrame
{
	uint8_t* data;
	uint32_t dataSize;
	uint64_t timestamp;
	uint32_t sampleRate = 8000;
	uint32_t channels = 1;
	uint32_t bitDepth = 16;
};

struct OeipPushSetting
{
	//推音频
	int32_t bAudio = true;
	//推视频
	int32_t bVideo = true;
	OeipVideoEncoder videoEncoder = {};
	OeipAudioEncoder audioEncoder = {};
};

typedef void(*onInitRoomAction)(int32_t code);
typedef void(*onLoginRoomAction)(int32_t code, int32_t userId);
typedef void(*onUserChangeAction)(int32_t userId, bool bAdd);
typedef void (*onStreamUpdateAction)(int32_t userId, int32_t index, bool bAdd);
typedef void (*onVideoFrameAction)(int32_t userId, int32_t index, OeipVideoFrame videoFrame);
typedef void (*onAudioFrameAction)(int32_t userId, int32_t index, OeipAudioFrame audioFrame);
typedef void(*onLogoutRoomAction)(int32_t code);
typedef void (*onOperateResultAction)(int32_t operate, int32_t code, std::string message);
typedef void (*onPushStreamAction)(int32_t index, int32_t code);
typedef void (*onPullStreamAction)(int32_t userId, int32_t index, int32_t code);

//C# 包装实现
struct LiveBackWrapper
{
	onInitRoomAction initRoomAction = nullptr;
	onLoginRoomAction loginRoomAction = nullptr;
	onUserChangeAction userChangeAction = nullptr;
	onStreamUpdateAction streamUpdateAction = nullptr;
	onVideoFrameAction videoFrameAction = nullptr;
	onAudioFrameAction audioFrameAction = nullptr;
	onLogoutRoomAction logoutRoomAction = nullptr;
	onOperateResultAction operateResultAction = nullptr;
	onPushStreamAction pushStreamAction = nullptr;
	onPullStreamAction pullStreamAction = nullptr;
};