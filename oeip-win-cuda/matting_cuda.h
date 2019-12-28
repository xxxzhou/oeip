#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_help.h"
#include "helper_math.h"
#include <iostream>


using namespace std;
//using namespace cv;
using namespace cv::cuda;

#define BLOCK_X 32
#define BLOCK_Y 8
#define BLOCK_XGMM 32
#define BLOCK_YGMM 2
#define CUDA_GRABCUT_K 5   //CUDA_GRABCUT_K
#define CUDA_GRABCUT_K2 10 //CUDA_GRABCUT_K2
//CPU中设置为相应值然后传给GPU
struct kmeansI
{
	float4 kcenters[CUDA_GRABCUT_K] = {};
	float4 cluster[CUDA_GRABCUT_K] = {};
	int length[CUDA_GRABCUT_K] = {};
};

struct gmmI
{
	float3 sums[CUDA_GRABCUT_K] = {};
	float3 prods[CUDA_GRABCUT_K][3] = {};
	int lenght[CUDA_GRABCUT_K] = {};
	int totalCount = 0;
	//每种分类所占比例,加起来为1
	float coefs[CUDA_GRABCUT_K] = {};
	//每种分类的均值
	float3 means[CUDA_GRABCUT_K] = {};
	//每种分类的线性关系解
	float3 inverseCovs[CUDA_GRABCUT_K][3] = {};
	//每种分类的矩阵对应的行列式
	float covDeterms[CUDA_GRABCUT_K] = {};
};

template<typename T>
void writerMat(GpuMat gpuMat)
{
	int width = gpuMat.cols;
	int height = gpuMat.rows;
	cv::Mat cpuresult;
	gpuMat.download(cpuresult);
	FILE *fp = fopen("result_sponge/a1.pgm", "w");
	fprintf(fp, "%c", 'P');
	fprintf(fp, "%c", '2');
	fprintf(fp, "%c", '\n');
	fprintf(fp, "%d %c %d %c ", width, ' ', height, '\n');
	fprintf(fp, "%d %c", 255, '\n');

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int value = (int)cpuresult.at<T>(i, j);//chEdge,cvEdge
			fprintf(fp, "%d\n", value);//
		}
	}

	fclose(fp);
	//exit(1);
}

inline void showMat(GpuMat gpuMat)
{
	cv::Mat cpuresult;
	gpuMat.download(cpuresult);
}

inline void showMat(GpuMat gpuMat, GpuMat gpuMat1, GpuMat gpuMat2)
{
	cv::Mat cpuresult, cpuresult1, cpuresult2;
	gpuMat.download(cpuresult);
	gpuMat1.download(cpuresult1);
	gpuMat2.download(cpuresult2);
}