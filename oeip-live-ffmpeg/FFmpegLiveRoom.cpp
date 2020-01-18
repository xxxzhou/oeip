#include "FFmpegLiveRoom.h"
using namespace std::placeholders;

FFmpegLiveRoom::FFmpegLiveRoom() {
	liveCom = std::make_unique<OeipLiveBackCom>();
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
	logoutRoom();
	shutdownRoom();
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
	liveCom->setLiveBack(liveBack);
	engine->liveBack = liveCom.get();
	liveCom->AddRef();
	auto finit = engine->InitRoom(liveCtx.liveServer);
	return finit != 0;
}

bool FFmpegLiveRoom::loginRoom() {
	int32_t code = engine->LoginRoom(roomName.c_str(), userId);
	if (code > 0) {
		userId = code;
	}
	liveCom->userId = userId;
	return code >= 0;
}

bool FFmpegLiveRoom::pushStream(int32_t index, const OeipPushSetting& setting) {
	if (liveCom->mediaServer.empty()) {
		liveBack->onOperateResult(11, -1, "login no return media sever.");
		return false;
	}
	if (index >= LIVE_OUTPUT_MAX) {
		return false;
	}
	std::string uri = liveCom->mediaServer + roomName + "_" + std::to_string(userId) + "_" + std::to_string(index);
	std::string message = "推流地址:" + uri;
	logMessage(OEIP_INFO, message.c_str());
	liveOuts[index]->enableAudio(setting.bAudio);
	liveOuts[index]->enableVideo(setting.bVideo);
	auto result = liveOuts[index]->open(uri.c_str());
	//限定只有主流推音频
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
	std::string uri = liveCom->mediaServer + roomName + "_" + std::to_string(userId) + "_" + std::to_string(index);
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
	for (auto& liveIn : liveIns) {
		liveIn.In->close();
	}
	int32_t code = engine->LogoutRoom();
	liveIns.clear();
	return code >= 0;
}

bool FFmpegLiveRoom::shutdownRoom() {
	engine->Shutdown();
	liveOuts.clear();
	liveIns.clear();
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
