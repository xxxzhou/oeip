#include "CudaComputeLayer.h"
#include "cuda_help.h"
#include <opencv2/core/cuda.hpp>  

void textureMap_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, MapChannelParamet paramt, cudaStream_t stream);
void yuv2rgb_gpu(PtrStepSz<uchar> source, PtrStepSz<uchar4> dest, int32_t yuvtype, cudaStream_t stream);
void yuv2rgb_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, bool ufront, bool yfront, cudaStream_t stream);
void rgb2yuv_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> dest, int32_t yuvtype, cudaStream_t stream);
void rgb2yuv_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, bool ufront, bool yfront, cudaStream_t stream);
void resize_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, bool bLinear, cudaStream_t stream);
void blend_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> blendTex, PtrStepSz<uchar4> dest,
	int32_t left, int32_t top, float opacity, cudaStream_t stream);
void operate_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, OperateParamet paramt, cudaStream_t stream);

void MapChannelLayerCuda::onRunLayer() {
	textureMap_gpu(inMats[0], outMats[0], layerParamet, ipCuda->cudaStream);
}

YUV2RGBALayerCuda::YUV2RGBALayerCuda() {
	initConnect();
}

void YUV2RGBALayerCuda::initConnect() {
	if (layerParamet.yuvType == OEIP_YUVFMT_YUV420SP || layerParamet.yuvType == OEIP_YUVFMT_YUY2P || layerParamet.yuvType == OEIP_YUVFMT_YUV420P) {
		selfConnects[0].dataType = OEIP_CV_8UC1;
		outConnects[0].dataType = OEIP_CV_8UC4;
	}
	else {
		selfConnects[0].dataType = OEIP_CV_8UC4;
		outConnects[0].dataType = OEIP_CV_8UC4;
	}
}

void YUV2RGBALayerCuda::onParametChange(YUV2RGBAParamet oldT) {
	initConnect();
	//yuvType的改变会影响后续层
	ipCuda->resetLayers();
}

void YUV2RGBALayerCuda::onInitLayer() {
	if (layerParamet.yuvType == OEIP_YUVFMT_YUV420SP || layerParamet.yuvType == OEIP_YUVFMT_YUV420P || layerParamet.yuvType == OEIP_YUVFMT_YUY2P) {
		outConnects[0].width = selfConnects[0].width;
		outConnects[0].height = selfConnects[0].height * 2 / 3;
		if (layerParamet.yuvType == OEIP_YUVFMT_YUY2P) {
			outConnects[0].height = selfConnects[0].height / 2;
		}
	}
	else if (layerParamet.yuvType == OEIP_YUVFMT_YUY2I || layerParamet.yuvType == OEIP_YUVFMT_YVYUI || layerParamet.yuvType == OEIP_YUVFMT_UYVYI) {
		outConnects[0].width = selfConnects[0].width * 2;
		outConnects[0].height = selfConnects[0].height;
	}
}

void YUV2RGBALayerCuda::onRunLayer() {
	if (layerParamet.yuvType == OEIP_YUVFMT_YUV420SP || layerParamet.yuvType == OEIP_YUVFMT_YUV420P || layerParamet.yuvType == OEIP_YUVFMT_YUY2P) {
		yuv2rgb_gpu(inMats[0], outMats[0], layerParamet.yuvType, ipCuda->cudaStream);
	}
	else if (layerParamet.yuvType == OEIP_YUVFMT_YUY2I || layerParamet.yuvType == OEIP_YUVFMT_YVYUI || layerParamet.yuvType == OEIP_YUVFMT_UYVYI) {
		bool ufront = true;
		bool yfront = true;
		if (layerParamet.yuvType == OEIP_YUVFMT_YVYUI)
			ufront = false;
		if (layerParamet.yuvType == OEIP_YUVFMT_UYVYI)
			yfront = false;
		yuv2rgb_gpu(inMats[0], outMats[0], ufront, yfront, ipCuda->cudaStream);
	}
}

void ResizeLayerCuda::onParametChange(ResizeParamet oldT) {
	ipCuda->resetLayers();
}

void ResizeLayerCuda::onInitLayer() {
	outConnects[0].width = layerParamet.width;
	outConnects[0].height = layerParamet.height;
}

void ResizeLayerCuda::onRunLayer() {
	resize_gpu(inMats[0], outMats[0], layerParamet.bLinear, ipCuda->cudaStream);
}

RGBA2YUVLayerCuda::RGBA2YUVLayerCuda() {
	initConnect();
}

void RGBA2YUVLayerCuda::initConnect() {
	if (layerParamet.yuvType == OEIP_YUVFMT_YUV420SP || layerParamet.yuvType == OEIP_YUVFMT_YUY2P || layerParamet.yuvType == OEIP_YUVFMT_YUV420P) {
		selfConnects[0].dataType = OEIP_CV_8UC4;
		outConnects[0].dataType = OEIP_CV_8UC1;
	}
	else {
		selfConnects[0].dataType = OEIP_CV_8UC4;
		outConnects[0].dataType = OEIP_CV_8UC4;
	}
}

void RGBA2YUVLayerCuda::onParametChange(RGBA2YUVParamet oldT) {
	initConnect();
	//yuvType的改变会影响后续层
	ipCuda->resetLayers();
}

void RGBA2YUVLayerCuda::onInitLayer() {
	if (layerParamet.yuvType == OEIP_YUVFMT_YUV420SP || layerParamet.yuvType == OEIP_YUVFMT_YUV420P || layerParamet.yuvType == OEIP_YUVFMT_YUY2P) {
		outConnects[0].width = selfConnects[0].width;
		outConnects[0].height = selfConnects[0].height * 3 / 2;
		if (layerParamet.yuvType == OEIP_YUVFMT_YUY2P) {
			outConnects[0].height = selfConnects[0].height * 2;
		}
	}
	else if (layerParamet.yuvType == OEIP_YUVFMT_YUY2I || layerParamet.yuvType == OEIP_YUVFMT_YVYUI || layerParamet.yuvType == OEIP_YUVFMT_UYVYI) {
		outConnects[0].width = selfConnects[0].width / 2;
		outConnects[0].height = selfConnects[0].height;
	}
}

void RGBA2YUVLayerCuda::onRunLayer() {
	if (layerParamet.yuvType == OEIP_YUVFMT_YUV420SP || layerParamet.yuvType == OEIP_YUVFMT_YUV420P || layerParamet.yuvType == OEIP_YUVFMT_YUY2P) {
		rgb2yuv_gpu(inMats[0], outMats[0], layerParamet.yuvType, ipCuda->cudaStream);
	}
	else if (layerParamet.yuvType == OEIP_YUVFMT_YUY2I || layerParamet.yuvType == OEIP_YUVFMT_YVYUI || layerParamet.yuvType == OEIP_YUVFMT_UYVYI) {
		bool ufront = true;
		bool yfront = true;
		if (layerParamet.yuvType == OEIP_YUVFMT_YVYUI)
			ufront = false;
		if (layerParamet.yuvType == OEIP_YUVFMT_UYVYI)
			yfront = false;
		rgb2yuv_gpu(inMats[0], outMats[0], ufront, yfront, ipCuda->cudaStream);
	}
}

void BlendLayerCuda::onParametChange(BlendParamet oldT) {
	if (layerParamet.width != oldT.width || layerParamet.height != oldT.height) {
		ipCuda->resetLayers();
	}
}

bool BlendLayerCuda::onInitBuffer() {
	LayerCuda::onInitBuffer();
	int32_t tempWidth = layerParamet.width * inMats[0].cols;
	int32_t tempHeight = layerParamet.height * inMats[0].rows;
	tempMat.create(tempHeight, tempWidth, OEIP_CV_8UC4);
	return true;
}

void BlendLayerCuda::onRunLayer() {
	top = layerParamet.top * inMats[0].rows;
	left = layerParamet.left * inMats[0].cols;
	resize_gpu(inMats[1], tempMat, true, ipCuda->cudaStream);
	blend_gpu(inMats[0], tempMat, outMats[0], left, top, layerParamet.opacity, ipCuda->cudaStream);
}

void OperateLayerCuda::onRunLayer() {
	OperateParamet temp = layerParamet;
	temp.gamma = 1.f / layerParamet.gamma;
	operate_gpu(inMats[0], outMats[0], temp, ipCuda->cudaStream);
}
