#include "DarknetLayer.h"
#include <filesystem>
#include <fstream>

void resize_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, bool bLinear, cudaStream_t stream);
void image2netData_gpu(PtrStepSz<uchar4> source, float* outData, cudaStream_t stream = nullptr);
void drawRect_gpu(PtrStepSz<uchar4> source, cv::Rect rect, int radius, uchar4 drawColor, cudaStream_t stream = nullptr);

DarknetLayerCuda::DarknetLayerCuda() {
}

DarknetLayerCuda::~DarknetLayerCuda() {
	SAFE_GPUDATA(netInput);
}

void DarknetLayerCuda::onParametChange(DarknetParamet oldT) {
	if (layerParamet.bLoad && !oldT.bLoad) {
		bool bInCfg = std::tr2::sys::exists(layerParamet.confile);
		bool bInWgt = std::tr2::sys::exists(layerParamet.weightfile);
		if (!bInCfg || !bInWgt) {
			logMessage(OEIP_WARN, "cfg or weights not find.");
			return;
		}
		net = load_network(layerParamet.confile, layerParamet.weightfile, 0);
		netWidth = network_width(net);
		netHeight = network_height(net);
		set_batch_network(net, 1);
		netFrame.create(netHeight, netWidth, CV_8UC4);
		reCudaAllocGpu((void**)&netInput, netHeight*netWidth * sizeof(float) * 3);
	}
}

bool DarknetLayerCuda::onInitBuffer() {
	return false;
}

void DarknetLayerCuda::onRunLayer() {
	if (!layerParamet.bLoad || !net)
		return;
	resize_gpu(inMats[0], netFrame, true, nullptr);
	image2netData_gpu(netFrame, netInput);
	network_predict_gpudata(net, netInput);
	int nboxes = 0;
	detection *dets = get_network_boxes(net, netWidth, netHeight, layerParamet.thresh, 0, 0, 1, &nboxes);
	//排序,nms 合并框(满足条件的放前面,被合并的放后面并置0)
	if (layerParamet.nms)
		do_nms_sort(dets, nboxes, classs, layerParamet.nms);
	int objectIndex = 0;//objectIndex 0 person 39 bottle，自己训练的模型只有人物
	std::vector<PersonBox> personDets;
	for (int i = 0; i < nboxes; i++) {
		if (dets[i].objectness > layerParamet.thresh) {
			auto det = dets[i];
			PersonBox box = {};
			box.prob = det.prob[objectIndex];
			box.centerX = det.bbox.x;
			box.centerY = det.bbox.y;
			box.width = det.bbox.w + 0.05;
			box.height = det.bbox.h + 0.05;
			personDets.push_back(box);
			if (layerParamet.bDraw) {
				cv::Rect rectangle2;
				rectangle2.width = box.width*inMats[0].cols;
				rectangle2.height = box.height*inMats[0].rows;
				rectangle2.x = box.centerX*inMats[0].cols - rectangle2.width / 2;
				rectangle2.y = box.centerY*inMats[0].rows - rectangle2.height / 2;
				//drawRect_gpu(sourceOpFrame, rectangle2, 3, make_uchar4(255, 0, 0, 255), cudaStream);
			}
		}
	}
	ipCuda->outputData(layerIndex, (uint8_t*)personDets.data(), sizeof(PersonBox), personDets.size(), 0);
}
