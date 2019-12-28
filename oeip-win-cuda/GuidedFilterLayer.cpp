#include "GuidedFilterLayer.h"

//void resize_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, bool bLinear, cudaStream_t stream);
void resize_gpuf(PtrStepSz<float4> source, PtrStepSz<float4> dest, bool bLinear, cudaStream_t stream);

void findMatrix_gpu(PtrStepSz<float4> source, PtrStepSz<float3> dest,
	PtrStepSz<float3> dest1, PtrStepSz<float3> dest2, cudaStream_t stream);
void guidedFilter_gpu(PtrStepSz<float4> source, PtrStepSz<float3> col1,
	PtrStepSz<float3> col2, PtrStepSz<float3> col3, PtrStepSz<float4> dest, float eps, cudaStream_t stream);
void guidedFilterResult_gpu(PtrStepSz<float4> source, PtrStepSz<float4> guid, PtrStepSz<uchar4> dest,
	float intensity, cudaStream_t stream);

GuidedFilterLayerCuda::GuidedFilterLayerCuda() {
	selfConnects[0].dataType = OEIP_CV_32FC4;
	outConnects[0].dataType = OEIP_CV_8UC4;
}

GuidedFilterLayerCuda::~GuidedFilterLayerCuda() {
}

void GuidedFilterLayerCuda::onParametChange(GuidedFilterParamet oldT) {
	if (oldT.zoom != layerParamet.zoom) {
		ipCuda->resetLayers();
	}
}

bool GuidedFilterLayerCuda::onInitBuffer() {
	LayerCuda::onInitBuffer();
	int32_t width = inMats[0].cols;
	int32_t height = inMats[0].rows;
	scaleWidth = width / layerParamet.zoom;
	scaleHeight = height / layerParamet.zoom;

	gpuResize.create(scaleHeight, scaleWidth, CV_32FC4);//I_sub+p_sub
	mean_I.create(scaleHeight, scaleWidth, CV_32FC4);
	mean_Ipv.create(scaleHeight, scaleWidth, CV_32FC3);
	var_I_rxv.create(scaleHeight, scaleWidth, CV_32FC3);
	var_I_gbxfv.create(scaleHeight, scaleWidth, CV_32FC3);
	mean_Ip.create(scaleHeight, scaleWidth, CV_32FC3);
	var_I_rx.create(scaleHeight, scaleWidth, CV_32FC3);
	var_I_gbxf.create(scaleHeight, scaleWidth, CV_32FC3);
	meanv.create(scaleHeight, scaleWidth, CV_32FC4);
	means.create(scaleHeight, scaleWidth, CV_32FC4);
	mean.create(height, width, CV_32FC4);
	//npp的流切换到本线程所用(希望npp每个线程有独立的stream,不然这句在这没意义)
	setNppStream(ipCuda->cudaStream);
	return true;
}

void GuidedFilterLayerCuda::onRunLayer() {
	NppiPoint oSrcOffset = { 0, 0 };
	NppiSize oSizeROI = { scaleWidth ,scaleHeight }; //NPPI blue	
	int softness = max(layerParamet.softness, 1);// max(keySetting.softness, 3);	
	NppiSize oMaskSize = { softness,softness };
	NppiPoint oAnchor = { softness / 2,softness / 2 };
	//缩放大小
	resize_gpuf(inMats[0], gpuResize, false, ipCuda->cudaStream);
	findMatrix_gpu(gpuResize, mean_Ipv, var_I_rxv, var_I_gbxfv, ipCuda->cudaStream);
	//模糊原始值
	nppiFilterBoxBorder_32f_C4R((Npp32f*)gpuResize.ptr<float4>(), gpuResize.step, oSizeROI, oSrcOffset, (Npp32f*)mean_I.ptr<float4>(), mean_I.step, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
	//模糊矩阵
	nppiFilterBoxBorder_32f_C3R((Npp32f*)mean_Ipv.ptr<float3>(), mean_Ipv.step, oSizeROI, oSrcOffset, (Npp32f*)mean_Ip.ptr<float3>(), mean_Ip.step, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
	nppiFilterBoxBorder_32f_C3R((Npp32f*)var_I_rxv.ptr<float3>(), var_I_rxv.step, oSizeROI, oSrcOffset, (Npp32f*)var_I_rx.ptr<float3>(), var_I_rx.step, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
	nppiFilterBoxBorder_32f_C3R((Npp32f*)var_I_gbxfv.ptr<float3>(), var_I_gbxfv.step, oSizeROI, oSrcOffset, (Npp32f*)var_I_gbxf.ptr<float3>(), var_I_gbxf.step, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
	//求导
	guidedFilter_gpu(mean_I, mean_Ip, var_I_rx, var_I_gbxf, meanv, layerParamet.eps, ipCuda->cudaStream);
	nppiFilterBoxBorder_32f_C4R((Npp32f*)meanv.ptr<float4>(), meanv.step, oSizeROI, oSrcOffset, (Npp32f*)means.ptr<float4>(), means.step, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
	resize_gpuf(means, mean, true, ipCuda->cudaStream);
	//结果
	guidedFilterResult_gpu(inMats[0], mean, outMats[0], layerParamet.intensity, ipCuda->cudaStream);
}
