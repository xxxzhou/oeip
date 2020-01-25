#include "FFmpegLiveRoom.h"
using namespace std::placeholders;

FFmpegLiveRoom::FFmpegLiveRoom() {
	liveCom = new OeipLiveBackCom();
	liveOuts.resize(LIVE_OUTPUT_MAX);
	for (int32_t i = 0; i < LIVE_OUTPUT_MAX; i++) {
		liveOuts[i] = std::make_unique< FLiveOutput>();
		onOperateHandle operateAction = std::bind(&FFmpegLiveRoom::onOperateAction, this, true, i, _1, _2);
		liveOuts[i]->setOperateEvent(operateAction);
	}
	//audioOutput = std::make_unique<FAudioOutput>();
	////设定音频输出信息
	//audioDesc.bitSize = 16;
	//audioDesc.channel = 1;
	//audioDesc.sampleRate = 8000;
}

FFmpegLiveRoom::~FFmpegLiveRoom() {
	shutdownRoom();
}

void FFmpegLiveRoom::onServerBack(std::string server, int32_t port, int32_t userId) {
	mediaServer = "rtmp://" + server + ":" + std::to_string(port) + "/live/";
	if (userId > 0)
		this->userId = userId;
}

void FFmpegLiveRoom::onOperateAction(bool bPush, int32_t index, int32_t operate, int32_t code) {
	if (bPush) {
		if (operate == OEIP_LIVE_OPERATE_OPEN) {
			liveBack->onPushStream(index, code);
		}
		else {
			liveBack->onOperateResult(operate, code, "");
		}
	}
	else {
		if (operate == OEIP_LIVE_OPERATE_OPEN) {
			if (index >= liveIns.size())
				return;
			LiveInput& input = liveIns[index];
			liveBack->onPullStream(input.userId, input.index, code);
		}
		else {
			liveBack->onOperateResult(operate, code, "");
		}
	}
}

int32_t FFmpegLiveRoom::findLiveInput(int32_t userId, int32_t index, LiveInput& input) {
	for (int32_t i = 0; i < liveIns.size(); i++) {
		if (liveIns[i].userId == userId && liveIns[i].index == index) {
			input = liveIns[i];
			return i;
		}
	}
	return -1;
}

void FFmpegLiveRoom::onAudioData(int32_t index, uint8_t* data, int32_t size) {
}

void FFmpegLiveRoom::onAudioFrame(int32_t userId, int32_t index, OeipAudioFrame audioFrame) {
	liveBack->onAudioFrame(userId, index, audioFrame);
}

void FFmpegLiveRoom::onVideoFrame(int32_t userId, int32_t index, OeipVideoFrame videoFrame) {
	liveBack->onVideoFrame(userId, index, videoFrame);
}

bool FFmpegLiveRoom::initRoom() {
	HRESULT hr = CoInitialize(NULL);
	hr = CoCreateInstance(CLSID_OeipLiveClient, nullptr, CLSCTX_INPROC_SERVER, IID_IOeipLiveClient, (void**)&engine);
	if (FAILED(hr))
		return false;
	onServeBack serverbackEvent = std::bind(&FFmpegLiveRoom::onServerBack, this, _1, _2, _3);
	liveCom->setLiveBack(liveBack, serverbackEvent);
	IOeipLiveCallBack* tempLiveCom = liveCom;
	engine->SetLiveCallBack(&tempLiveCom);
	auto finit = engine->InitRoom(liveCtx.liveServer);
	return finit != 0;
	//return true;
}

bool FFmpegLiveRoom::loginRoom() {
	int32_t code = engine->LoginRoom(roomName.c_str(), userId);
	if (code > 0) {
		userId = code;
	}
	//liveCom->userId = userId;
	bLogin = true;
	return code >= 0;
}

bool FFmpegLiveRoom::pushStream(int32_t index, const OeipPushSetting& setting) {
	if (mediaServer.empty()) {
		liveBack->onOperateResult(11, -1, "login no return media sever.");
		return false;
	}
	if (index >= LIVE_OUTPUT_MAX) {
		return false;
	}
	std::string uri = mediaServer + roomName + "_" + std::to_string(userId) + "_" + std::to_string(index);
	std::string message = "push url:" + uri;
	logMessage(OEIP_INFO, message.c_str());
	liveOuts[index]->enableAudio(setting.bAudio);
	liveOuts[index]->enableVideo(setting.bVideo);
	if (setting.videoEncoder.bitrate > 0) {
		liveOuts[index]->setVideoBitrate(setting.videoEncoder.bitrate);
	}
	if (setting.audioEncoder.bitrate > 0) {
		liveOuts[index]->setAudioBitrate(setting.audioEncoder.bitrate);
	}
	auto result = liveOuts[index]->open(uri.c_str());
	//限定只有主流推音频 音频也让外面自己推，在这不集成，更大的自由度,必要的话，在外面在包装一层 摄像头/麦与声卡
	//if (result >= 0 && index == 0) {
	//	onAudioDataHandle dataHandle = std::bind(&FFmpegLiveRoom::onAudioData, this, index, _1, _2);
	//	audioOutput->onDataHandle = dataHandle;
	//	audioOutput->start(true, liveCtx.bLoopback, audioDesc);
	//}
	int32_t code = engine->PushStream(index, setting.bVideo, setting.bAudio);
	return code >= 0;
}

bool FFmpegLiveRoom::stopPushStream(int32_t index) {
	if (index >= LIVE_OUTPUT_MAX) {
		return false;
	}
	int32_t code = engine->StopPushStream(index);
	liveOuts[index]->close();
	return code >= 0;
}

bool FFmpegLiveRoom::pullStream(int32_t userId, int32_t index) {
	LiveInput input = {};
	//如果没有，就新增
	int32_t inIndex = findLiveInput(userId, index, input);
	if (inIndex < 0) {
		input.userId = userId;
		input.index = index;
		input.In = std::make_shared<FLiveInput>();
		onVideoDataHandle videoHandle = std::bind(&FFmpegLiveRoom::onVideoFrame, this, userId, index, _1);
		input.In->setVideoDataEvent(videoHandle);
		onOperateHandle operateAction = std::bind(&FFmpegLiveRoom::onOperateAction, this, false, liveIns.size(), _1, _2);
		input.In->setOperateEvent(operateAction);
		liveIns.push_back(input);
	}
	engine->PullStream(userId, index);
	std::string uri = mediaServer + roomName + "_" + std::to_string(userId) + "_" + std::to_string(index);
	int32_t code = input.In->open(uri.c_str());
	return code >= 0;
}

bool FFmpegLiveRoom::stopPullStream(int32_t userId, int32_t index) {
	LiveInput input = {};
	//如果没有，就新增
	int32_t code = 0;
	if (findLiveInput(userId, index, input) >= 0) {
		input.In->close();
		code = engine->StopPullStream(userId, index);
	}
	return code >= 0;
}

bool FFmpegLiveRoom::logoutRoom() {
	if (!bLogin)
		return true;
	for (auto& liveIn : liveIns) {
		liveIn.In->close();
	}
	int32_t code = 0;
	try {
		code = engine->LogoutRoom();
	}
	catch (const _com_error&) {
		code = -1;
	}
	liveIns.clear();
	mediaServer = "";
	userId = 0;
	bLogin = false;
	return code >= 0;
}

bool FFmpegLiveRoom::shutdownRoom() {
	if (!bShutDown) {
		for (auto& liveOut : liveOuts) {
			liveOut->setOperateEvent(nullptr);
		}
		logoutRoom();
		engine->Shutdown();
		engine->Release();
		liveOuts.clear();
		liveIns.clear();
		bShutDown = true;
	}
	return true;
}

bool FFmpegLiveRoom::pushVideoFrame(int32_t index, const OeipVideoFrame& videoFrame) {
	if (index >= LIVE_OUTPUT_MAX) {
		return false;
	}
	int32_t code = liveOuts[index]->pushVideo(videoFrame);
	return code >= 0;
}

bool FFmpegLiveRoom::pushAudioFrame(int32_t index, const OeipAudioFrame& audioFrame) {
	if (index >= LIVE_OUTPUT_MAX) {
		return false;
	}
	int32_t code = liveOuts[index]->pushAudio(audioFrame);
	return code >= 0;
}

void registerFactory() {
	registerLiveFactory(new FFmpegLiveRoomFactory(), OIEP_FFMPEG, "ffmpeg live dll");
}

bool bCanLoad() {
	return true;
}
