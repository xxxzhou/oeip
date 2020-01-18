#include "GrabcutLayerCuda.h"

template <typename T>
void resize_gpu(PtrStepSz<T> source, PtrStepSz<T> dest, bool bLinear, cudaStream_t stream);
void showKmeans_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> clusterIndex, cudaStream_t stream = nullptr);
void rgb2rgba_gpu(PtrStepSz<uchar3> source, PtrStepSz<uchar4> dest, cudaStream_t stream = nullptr);
void rgba2bgr_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar3> dest, cudaStream_t stream);
void setMask_gpu(PtrStepSz<uchar> mask, cv::Rect rect, cudaStream_t stream = nullptr);
void setMask_gpu(PtrStepSz<uchar> mask, int x, int y, int radius, int vmask, cudaStream_t stream = nullptr);
void setMask_gpu(PtrStepSz<uchar> source, PtrStepSz<uchar> mask, int radius, cv::Rect rect, cudaStream_t stream = nullptr);
void showSeedMask_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, PtrStepSz<uchar> mask, cudaStream_t stream = nullptr);
void showMask_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> mask, cudaStream_t stream = nullptr);
void combinGrabcutMask_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, PtrStepSz<uchar> mask, cudaStream_t stream = nullptr);

//(0 GC_BGD) (1 GC_FGD) (2 GC_PR_BGD) (3 GC_PR_FGD)
GrabcutLayerCuda::GrabcutLayerCuda() {
	kmeans = std::make_unique<KmeansCuda>();
	gmm = std::make_unique<GMMCuda>();
	graph = std::make_unique<GraphCuda>();
}

GrabcutLayerCuda::~GrabcutLayerCuda() {

}

void GrabcutLayerCuda::onParametChange(GrabcutParamet oldT) {
	//画背景点或是grabcut扣像
	if (layerParamet.bDrawSeed != oldT.bDrawSeed) {
		//计算一桢(可以换成cv::grabCut来算,opencv本身grabCut用的八向，切割的边缘更清晰)
		if (layerParamet.bDrawSeed) {
			//假定全可能是前景,好展示给用户
			cudaMemset2DAsync(mask.ptr(), mask.step, 3, mask.cols, mask.rows, ipCuda->cudaStream);
			//临时保证一桢数据
			resize_gpu<uchar4>(inMats[0], grabMat, true, ipCuda->cudaStream);
			//这种是根据传入的矩形框算前后景的GMM模型(也可手动画点)
			if (layerParamet.rect.width >= 0 && layerParamet.rect.height >= 0) {
				cv::Rect rect = {};
				rect.width = layerParamet.rect.width * width;
				rect.height = layerParamet.rect.height * height;
				rect.x = (layerParamet.rect.centerX - layerParamet.rect.width / 2) * width;
				rect.y = (layerParamet.rect.centerY - layerParamet.rect.height / 2) * height;
				setMask_gpu(mask, rect);
				//用opencv的方式来计算mask,因为opencv本身用八向，切割的边缘更清晰,CPU只算一次
				if (!layerParamet.bGpuSeed) {
					GpuMat temp = GpuMat(height, width, CV_8UC3);
					cv::Mat cpuTemp;
					cv::Mat cpuResult;
					GpuMat tempMask;
					rgba2bgr_gpu(grabMat, temp, ipCuda->cudaStream);
					temp.download(cpuTemp, ipCuda->stream);
					cudaStreamSynchronize(ipCuda->cudaStream);
					cv::Mat bgModel, fgModel;
					cv::grabCut(cpuTemp, cpuResult, rect, bgModel, fgModel, layerParamet.iterCount, cv::GC_INIT_WITH_RECT);
					//显示为3的就是GC_PR_FGD，
					tempMask.upload(cpuResult, ipCuda->stream);
					//根据resultMask与rectangle构建mask
					setMask_gpu(tempMask, mask, 1, rect, ipCuda->cudaStream);
				}
			}
			bComputeSeed = false;
		}
		else {
			//用上面RECT算的mask重新算K-means值
			kmeans->compute(grabMat, clusterIndex, mask, 0, true);
			//根据K-means 算前景与背景高斯混合模型
			gmm->learning(grabMat, clusterIndex, mask, true);
			//根据高斯混合模型重新计算clusterIndex
			gmm->assign(grabMat, clusterIndex, mask);
			//用高斯计算的clusterIndex更新高斯混合模型
			gmm->learning(grabMat, clusterIndex, mask, true);
			cudaMemset2DAsync(mask.ptr(), mask.step, 2, mask.cols, mask.rows, ipCuda->cudaStream);
			bComputeSeed = true;
		}
	}
}

bool GrabcutLayerCuda::onInitBuffer() {
	width = selfConnects[0].width / 2;
	height = selfConnects[0].height / 2;

	showMask.create(selfConnects[0].height, selfConnects[0].width, CV_8UC1);

	mask.create(height, width, CV_8UC1);
	clusterIndex.create(height, width, CV_8UC1);
	//source = GpuMat(height, width, CV_8UC4);
	grabMat.create(height, width, CV_8UC4);

	kmeans->init(width, height, ipCuda->cudaStream);
	graph->init(width, height, ipCuda->cudaStream);
	gmm->init(width, height, ipCuda->cudaStream);

	//(0 GC_BGD) (1 GC_FGD) (2 GC_PR_BGD) (3 GC_PR_FGD)
	cudaMemset2D(mask.ptr(), mask.step, 2, mask.cols, mask.rows);

	return LayerCuda::onInitBuffer();
}

void GrabcutLayerCuda::onRunLayer() {
	if (layerParamet.bDrawSeed) {
		//手工指出前景与背景,后面可以添加支持
		if (groundMode >= 0 && uvX > 0 && uvY > 0) {
			setMask_gpu(mask, uvX * width / selfConnects[0].width, uvY * height / selfConnects[0].height, 5, groundMode, ipCuda->cudaStream);
		}
		if (layerParamet.bGpuSeed) {
			kmeans->compute(grabMat, clusterIndex, mask, 0);//testShow(showSource, clusterIndex, "image");
			gmm->learning(grabMat, clusterIndex, mask);	//testShow(showSource, clusterIndex, "seg image");
			//计算图论的边
			graph->addEdges(grabMat, layerParamet.gamma);
			int index = 0;
			while (index++ < layerParamet.iterCount) {
				//根据mask重新分配分类的索引值
				gmm->assign(grabMat, clusterIndex, mask);
				//根据分类重新计算GMM模型
				gmm->learning(grabMat, clusterIndex, mask);
				//计算图论的formSource,toSinke
				graph->addTermWeights(grabMat, mask, *(gmm->bg), *(gmm->fg), layerParamet.lambda);
				//最大流
				graph->maxFlow(mask, layerParamet.seedCount);
			}
		}
		//grabCut->mask变成特定大小
		resize_gpu<uchar>(mask, showMask, true, ipCuda->cudaStream);
		resize_gpu<uchar4>(grabMat, inMats[0], true, ipCuda->cudaStream);
		//showMat(showMask);
		//显示mask,结果显示到result中
		showSeedMask_gpu(inMats[0], outMats[0], showMask, ipCuda->cudaStream);
	}
	else {
		if (!bComputeSeed) {
			cudaMemcpy2DAsync(outMats[0].ptr(), outMats[0].step, inMats[0].ptr(), inMats[0].step, inMats[0].cols * 4, inMats[0].rows, cudaMemcpyDeviceToDevice, ipCuda->cudaStream);
			return;
		}
		resize_gpu<uchar4>(inMats[0], grabMat, true, ipCuda->cudaStream);
		//计算图论的边
		graph->addEdges(grabMat, layerParamet.gamma);
		//计算图论的formSource,toSinke
		graph->addTermWeights(grabMat, mask, *(gmm->bg), *(gmm->fg), layerParamet.lambda);
		//最大流
		graph->maxFlow(mask, layerParamet.count);
		//grabCut->mask变成特定大小
		resize_gpu<uchar>(mask, showMask, true, ipCuda->cudaStream);
		//实际透明通道处理
		//combinGrabcutMask_gpu(inMats[0], outMats[0], showMask, ipCuda->cudaStream);
		//用于非透明显示
		showSeedMask_gpu(inMats[0], outMats[0], showMask, ipCuda->cudaStream);
		//showMat(inMats[0], outMats[0], showMask);
	}
}
