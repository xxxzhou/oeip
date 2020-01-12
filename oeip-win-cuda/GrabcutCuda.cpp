#include "GrabcutCuda.h"
#include "../oeip/OeipCommon.h"

void showKmeans_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> clusterIndex, cudaStream_t stream = nullptr);
void rgb2rgba_gpu(PtrStepSz<uchar3> source, PtrStepSz<uchar4> dest, cudaStream_t stream = nullptr);
void setMask_gpu(PtrStepSz<uchar> mask, cv::Rect rect, cudaStream_t stream = nullptr);
void setMask_gpu(PtrStepSz<uchar> mask, int x, int y, int radius, int vmask, cudaStream_t stream = nullptr);
void showSeedMask_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, PtrStepSz<uchar> mask, cudaStream_t stream = nullptr);
void showMask_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> mask, cudaStream_t stream = nullptr);

//用于image Watch查看Kmeans/GMM的分组效果图
void testShow(GpuMat source, GpuMat clusterIndex) {
	showKmeans_gpu(source, clusterIndex);
	cv::Mat result;
	source.download(result);
}

GrabcutCuda::GrabcutCuda() {
	kmeans = new KmeansCuda();
	graph = new GraphCuda();
	gmm = new GMMCuda();
}

GrabcutCuda::~GrabcutCuda() {
	mask.release();
	clusterIndex.release();
	//source.release();
	showSource.release();
	SAFE_DELETE(kmeans);
	SAFE_DELETE(graph);
	SAFE_DELETE(gmm);
}

void GrabcutCuda::init(int dwidth, int dheight, cudaStream_t stream) {
	width = dwidth;
	height = dheight;
	cudaStream = stream;

	mask.create(height, width, CV_8UC1);
	clusterIndex.create(height, width, CV_8UC1);
	//source = GpuMat(height, width, CV_8UC4);
	showSource.create(height, width, CV_8UC4);

	kmeans->init(dwidth, dheight, cudaStream);
	graph->init(dwidth, dheight, cudaStream);
	gmm->init(dwidth, dheight, cudaStream);

	//(0 GC_BGD) (1 GC_FGD) (2 GC_PR_BGD) (3 GC_PR_FGD)
	cudaMemset2D(mask.ptr(), mask.step, 2, mask.cols, mask.rows);
}

void GrabcutCuda::setSeedMode(bool bDraw) {
	if (bDraw) {
		cudaMemset2DAsync(mask.ptr(), mask.step, 3, mask.cols, mask.rows, cudaStream);
		bComputeSeed = false;
	}
}

void GrabcutCuda::computeSeed(GpuMat fsource) {
	//算K-means值
	kmeans->compute(fsource, clusterIndex, mask, 0, true);
	//根据K-means 算前景与背景高斯混合模型
	gmm->learning(fsource, clusterIndex, mask, true);
	gmm->assign(fsource, clusterIndex, mask);
	gmm->learning(fsource, clusterIndex, mask, true);
	bComputeSeed = true;
	cudaMemset2DAsync(mask.ptr(), mask.step, 2, mask.cols, mask.rows, cudaStream);
}

void GrabcutCuda::renderFrame(GpuMat source, cv::Rect rect) {
	setMask_gpu(mask, rect);
	//算K-means值
	kmeans->compute(source, clusterIndex, mask, 0);
	//查看分组显示效果图
	//testShow(showSource, clusterIndex);
	//根据K-means 算用GMM再分组
	gmm->learning(source, clusterIndex, mask);
	//testShow(showSource, clusterIndex);
	//计算图论的边
	graph->addEdges(source, gamma);
	int index = 0;
	while (index++ < iterCount) {
		//根据mask重新分配分类的索引值
		gmm->assign(source, clusterIndex, mask);
		//根据分类重新计算GMM模型
		gmm->learning(source, clusterIndex, mask);
		//计算图论的formSource,toSinke
		graph->addTermWeights(source, mask, *(gmm->bg), *(gmm->fg), lambda);
		//最大流
		graph->maxFlow(mask, maxCount);
	}
}

void GrabcutCuda::renderFrame(GpuMat x4source) {
	if (!bComputeSeed)
		return;
	graph->addEdges(x4source, gamma);
	int index = 0;
	//while (index++ < iterCount)
	{
		graph->addTermWeights(x4source, *(gmm->bg), *(gmm->fg), lambda);
		graph->maxFlow(mask, maxCount);
	}
}

void GrabcutCuda::setParams(int iiterCount, float fgamma, float flambda, int imaxCount) {
	iterCount = iiterCount;
	gamma = fgamma;
	lambda = flambda;
	maxCount = imaxCount;
}


