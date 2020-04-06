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

struct OeipPushSetting
{
	//推音频,如果为true,自己管理，如果为false,系统自动推麦的声音
	int32_t bAudio = false;
	//推视频
	int32_t bVideo = true;
	OeipVideoEncoder videoEncoder = {};
	OeipAudioEncoder audioEncoder = {};
};

typedef void(*onInitRoomAction)(int32_t code);
typedef void(*onLoginRoomAction)(int32_t code, int32_t userId);
typedef void(*onUserChangeAction)(int32_t userId, bool bAdd);
typedef void (*onStreamUpdateAction)(int32_t userId, int32_t index, bool bAdd);
typedef void (*onLiveVideoFrameAction)(int32_t userId, int32_t index, OeipVideoFrame videoFrame);
typedef void (*onLiveAudioFrameAction)(int32_t userId, int32_t index, OeipAudioFrame audioFrame);
typedef void(*onLogoutRoomAction)(int32_t code);
typedef void (*onOperateResultAction)(int32_t operate, int32_t code, const char* message);
typedef void (*onPushStreamAction)(int32_t index, int32_t code);
typedef void (*onPullStreamAction)(int32_t userId, int32_t index, int32_t code);

//C# 包装实现
struct LiveBackWrapper
{
	onInitRoomAction initRoomAction = nullptr;
	onLoginRoomAction loginRoomAction = nullptr;
	onUserChangeAction userChangeAction = nullptr;
	onStreamUpdateAction streamUpdateAction = nullptr;
	onLiveVideoFrameAction videoFrameAction = nullptr;
	onLiveAudioFrameAction audioFrameAction = nullptr;
	onLogoutRoomAction logoutRoomAction = nullptr;
	onOperateResultAction operateResultAction = nullptr;
	onPushStreamAction pushStreamAction = nullptr;
	onPullStreamAction pullStreamAction = nullptr;
};