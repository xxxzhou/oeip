#pragma once

#include "matting_cuda.h"

//GC_BGD    = 0,  //!< an obvious background pixels
//GC_FGD    = 1,  //!< an obvious foreground (object) pixel
//GC_PR_BGD = 2,  //!< a possible background pixel
//GC_PR_FGD = 3   //!< a possible foreground pixel
//const int CUDA_GRABCUT_K = 5;

//是否前景
inline __host__ __device__ bool checkFg(uchar mask)
{
	if (mask == 0 || mask == 2)
		return false;
	return true;
}

//是否是确定的值,如确定前景与背景
inline __host__ __device__ bool checkObvious(uchar mask)
{
	if (mask == 0 || mask == 1)
		return true;
	return false;
}

inline __global__ void setMask(PtrStepSz<uchar> mask, cv::Rect rect)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < mask.cols && idy < mask.rows)
	{
		int mmask = 3;
		//前景为255
		if (idx < rect.x || idx > rect.x + rect.width || idy < rect.y || idy > rect.y + rect.height)
		{
			mmask = 0;
		}
		mask(idy, idx) = mmask;
	}
}

inline __global__ void setMask(PtrStepSz<uchar> source, PtrStepSz<uchar> mask, int radius, cv::Rect rect)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows)
	{
		int mmask = 0;
		//前景为255
		if (idx > rect.x&& idx < rect.x + rect.width && idy > rect.y&& idy < rect.y + rect.height)
		{
			mmask = mask(idy, idx);
			int smask = source(idy, idx);
			if (smask == 3)
				mmask = 1;
		}
		mask(idy, idx) = mmask;
	}
}

inline __global__ void setMask(PtrStepSz<uchar> mask, int x, int y, int radius, int vmask)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx == x && idy == y)
	{
		for (int j = max(0, idy - radius); j < min(mask.rows, idy + radius); j += 1)
			for (int i = max(0, idx - radius); i < min(mask.cols, idx + radius); i += 1)
			{
				mask(j, i) = vmask;
			}
	}
}

inline __global__ void showSeedMask(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, PtrStepSz<uchar> mask, float fx, float fy)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows)
	{
		int vmask = mask(idy, idx);
		float4 color = rgbauchar42float4(source(idy, idx));
		color.w = 0.2f;
		if (vmask == 0)
		{
			color = color / 2.f;
			color.w = 0.8f;
		}
		else if (vmask == 1)
		{
			color = color * 2.f;
			color.w = 1.f;
		}
		dest(idy, idx) = rgbafloat42uchar4(color);
	}
}

inline __global__ void showMask(PtrStepSz<uchar4> source, PtrStepSz<uchar> mask)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows)
	{
		int vmask = mask(idy, idx);
		if (!checkFg(vmask))
		{
			source(idy, idx) = make_uchar4(0, 0, 0, 255);
		}
	}
}

//initKmeans 用于初始化CUDA_GRABCUT_K值的几个cluster
//findNearestCluster 更新clusterIndex对应每点的索引
//updateCluster 更新kcenters的值
//setNewCluster 重新设置cluster的值
inline __global__ void initKmeans(kmeansI& meansbg, kmeansI& meansfg)
{
	const int idx = threadIdx.x;

	meansbg.kcenters[idx] = make_float4(0.f);
	meansbg.length[idx] = 0;
	meansfg.kcenters[idx] = make_float4(0.f);
	meansfg.length[idx] = 0;

	float4 colorArray[] = {
		make_float4(1.f),
		make_float4(1.f,0.f,0.f,1.f),
		make_float4(0.f,1.f,0.f,1.f),
		make_float4(0.f,0.f,1.f,1.f),
		make_float4(0.f,0.f,0.f,1.f) };

	meansbg.cluster[idx] = colorArray[idx];
	meansfg.cluster[idx] = colorArray[idx];
}

template<bool bSeed>
inline __global__ void findNearestCluster(PtrStepSz<uchar4> source, PtrStepSz<uchar> clusterIndex, PtrStepSz<uchar> mask,
	kmeansI& meansbg, kmeansI& meansfg)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows)
	{
		float4 color = rgbauchar42float4(source(idy, idx));
		uchar umask = mask(idy, idx);
		if (!bSeed || (bSeed && checkObvious(umask)))
		{
			bool bFg = checkFg(umask);
			//背景块,使用kcentersbg,否则使用kcentersfg
			kmeansI& kmeans = bFg ? meansfg : meansbg;
			float min_dist = 10000000.f;
			//找到最近的那个点索引
			uchar index = 0;
			for (int i = 0; i < CUDA_GRABCUT_K; i++)
			{
				float4 distance = kmeans.cluster[i] - color;
				float dist = dot(distance, distance);
				if (dist < min_dist)
				{
					min_dist = dist;
					index = i;
				}
			}
			clusterIndex(idy, idx) = index;
		}
	}
}

//把source所有收集到一块gridDim.x*gridDim.y块数据上。
template<int blockx, int blocky, bool bSeed>
__global__ void updateCluster(PtrStepSz<uchar4> source, PtrStepSz<uchar> clusterIndex, PtrStepSz<uchar> mask, float4* kencter, int* kindexs)
{
	__shared__ float3 centers[blockx * blocky][CUDA_GRABCUT_K2];
	__shared__ int indexs[blockx * blocky][CUDA_GRABCUT_K2];
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	const int threadId = threadIdx.x + threadIdx.y * blockDim.x;
#pragma unroll CUDA_GRABCUT_K2
	for (int i = 0; i < CUDA_GRABCUT_K2; i++)
	{
		centers[threadId][i] = make_float3(0.f);
		indexs[threadId][i] = 0;
	}
	__syncthreads();
	if (idx < source.cols && idy < source.rows)
	{
		//所有值都放入共享centers
		int index = clusterIndex(idy, idx);
		uchar umask = mask(idy, idx);
		if (!bSeed || (bSeed && checkObvious(umask)))
		{
			bool bFg = checkFg(umask);
			int kindex = bFg ? index : (index + CUDA_GRABCUT_K);
			float4 color = rgbauchar42float4(source(idy, idx));
			centers[threadId][kindex] = make_float3(color);
			indexs[threadId][kindex] = 1;
		}
		__syncthreads();
		//每个线程块进行二分聚合,每次操作都保存到前一半数组里，直到最后保存在线程块里第一个线程上(这块比较费时,0.1ms)
		for (uint stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1)
		{
			//int tid = (threadId&(stride - 1));
			if (threadId < stride)//stride 2^n
			{
#pragma unroll CUDA_GRABCUT_K2
				for (int i = 0; i < CUDA_GRABCUT_K2; i++)
				{
					centers[threadId][i] += centers[threadId + stride][i];
					indexs[threadId][i] += indexs[threadId + stride][i];
				}
			}
			//if (stride > 32)
			__syncthreads();
		}
		//每块的第一个线程集合，把共享centers存入临时显存块上kencter
		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			int blockId = blockIdx.x + blockIdx.y * gridDim.x;
#pragma unroll CUDA_GRABCUT_K2
			for (int i = 0; i < CUDA_GRABCUT_K2; i++)
			{
				int id = blockId * 2 * CUDA_GRABCUT_K + i;
				kencter[id] = make_float4(centers[0][i], 0.f);
				kindexs[id] = indexs[0][i];
			}
		}
	}
}

//block 1*1,threads(暂时选32*4),对如上gridDim.x*gridDim.y的数据用blockx*blocky个线程来处理
template<int blockx, int blocky>
__global__ void updateCluster(float4* kencter, int* kindexs, kmeansI& meansbg, kmeansI& meansfg, int& delta, int gridcount)
{
	__shared__ float3 centers[blockx * blocky][CUDA_GRABCUT_K2];
	__shared__ int indexs[blockx * blocky][CUDA_GRABCUT_K2];
	const int threadId = threadIdx.x + threadIdx.y * blockDim.x;
#pragma unroll CUDA_GRABCUT_K2
	for (int i = 0; i < CUDA_GRABCUT_K2; i++)
	{
		centers[threadId][i] = make_float3(0.f);
		indexs[threadId][i] = 0;
	}
	__syncthreads();
	//int gridcount = gridDim.x*gridDim.y;
	int blockcount = blockDim.x * blockDim.y;
	int count = gridcount / blockcount + 1;
	//每块共享变量，每个线程操纵自己对应那块地址，不需要同步,用线程块操作一个大内存
	for (int i = 0; i < count; i++)
	{
		int id = i * blockcount + threadId;
		if (id < gridcount)
		{
#pragma unroll CUDA_GRABCUT_K2
			for (int j = 0; j < CUDA_GRABCUT_K2; j++)
			{
				int iid = id * CUDA_GRABCUT_K2 + j;
				centers[threadId][j] += make_float3(kencter[iid]);
				indexs[threadId][j] += kindexs[iid];
			}
		}
	}
	__syncthreads();
	for (uint stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1)
	{
		if (threadId < stride)
		{
#pragma unroll CUDA_GRABCUT_K2
			for (int i = 0; i < CUDA_GRABCUT_K2; i++)
			{
				centers[threadId][i] += centers[threadId + stride][i];
				indexs[threadId][i] += indexs[threadId + stride][i];
			}
		}
		//if (stride > 32)
		__syncthreads();
	}
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		int count = 0;
		//收集所有信息，并重新更新cluster，记录变更的大小
		for (int i = 0; i < CUDA_GRABCUT_K; i++)
		{
			meansfg.kcenters[i] = make_float4(centers[0][i], 0.f);
			if (indexs[0][i] != 0)
				meansfg.cluster[i] = meansfg.kcenters[i] / indexs[0][i];
			count += abs(indexs[0][i] - meansfg.length[i]);
			meansfg.length[i] = indexs[0][i];
		}
		for (int i = CUDA_GRABCUT_K; i < CUDA_GRABCUT_K2; i++)
		{
			meansbg.kcenters[i - CUDA_GRABCUT_K] = make_float4(centers[0][i], 0.f);
			if (indexs[0][i] != 0)
				meansbg.cluster[i - CUDA_GRABCUT_K] = meansbg.kcenters[i - CUDA_GRABCUT_K] / indexs[0][i];
			count += abs(indexs[0][i] - meansbg.length[i - CUDA_GRABCUT_K]);
			meansbg.length[i - CUDA_GRABCUT_K] = indexs[0][i];
		}
		delta = count;
	}
}

//block 1,threads k
inline __global__ void setNewCluster(kmeansI& meansbg, kmeansI& meansfg)
{
	int idx = threadIdx.x;
	if (meansbg.length[idx] != 0)
		meansbg.cluster[idx] = meansbg.kcenters[idx] / meansbg.length[idx];
	if (meansfg.length[idx] != 0)
		meansfg.cluster[idx] = meansfg.kcenters[idx] / meansfg.length[idx];
	__syncthreads();
	meansbg.kcenters[idx] = make_float4(0.f);
	meansbg.length[idx] = 0;
	meansfg.kcenters[idx] = make_float4(0.f);
	meansfg.length[idx] = 0;
}

inline __global__ void showKmeans(PtrStepSz<uchar4> source, PtrStepSz<uchar> clusterIndex)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;

	float4 colorArray[] = {
		make_float4(1.f),
		make_float4(1.f,0.f,0.f,1.f),
		make_float4(0.f,1.f,0.f,1.f),
		make_float4(0.f,0.f,1.f,1.f),
		make_float4(0.f,0.f,0.f,1.f) };

	if (idx < source.cols && idy < source.rows)
	{
		int index = clusterIndex(idy, idx);
		source(idy, idx) = rgbafloat42uchar4(colorArray[index]);
	}
}

inline __host__ __device__ void inverseMat3x3(float& covDeterm, float3& col0, float3& col1, float3& col2,
	float3& invCol0, float3& invCol1, float3& invCol2)
{
	float det = col0.x * (col1.y * col2.z - col2.y * col1.z)
		- col0.y * (col1.x * col2.z - col1.z * col2.x)
		+ col0.z * (col1.x * col2.y - col1.y * col2.x);
	float singularFix = 0.01f;
	if (det <= 1e-6)
	{
		col0.x += singularFix;
		col1.y += singularFix;
		col2.z += singularFix;
		det = col0.x * (col1.y * col2.z - col2.y * col1.z)
			- col0.y * (col1.x * col2.z - col1.z * col2.x)
			+ col0.z * (col1.x * col2.y - col1.y * col2.x);
	}
	covDeterm = det;
	if (det > 1e-6)
	{
		float invdet = 1.0f / det;
		invCol0.x = (col1.y * col2.z - col2.y * col1.z) * invdet;
		invCol0.y = (col0.z * col2.y - col0.y * col2.z) * invdet;
		invCol0.z = (col0.y * col1.z - col0.z * col1.y) * invdet;
		invCol1.x = (col1.z * col2.x - col1.x * col2.z) * invdet;
		invCol1.y = (col0.x * col2.z - col0.z * col2.x) * invdet;
		invCol1.z = (col1.x * col0.z - col0.x * col1.z) * invdet;
		invCol2.x = (col1.x * col2.y - col2.x * col1.y) * invdet;
		invCol2.y = (col2.x * col0.y - col0.x * col2.y) * invdet;
		invCol2.z = (col0.x * col1.y - col1.x * col0.y) * invdet;
	}
}

inline __host__ __device__ float colorGmm(gmmI& gmm, int ci, float3& color)
{
	float res = 0.f;
	if (gmm.coefs[ci] > 0)
	{
		float3 diff = color - gmm.means[ci];
		float3 dmul = mulMat(diff, gmm.inverseCovs[ci][0], gmm.inverseCovs[ci][1], gmm.inverseCovs[ci][2]);
		float mult = dot(diff, dmul);
		res = 1.f / sqrtf(gmm.covDeterms[ci]) * exp(-0.5f * mult);
	}
	return res;
}

inline __host__ __device__ float colorGmm(gmmI& gmm, float3& color)
{
	float res = 0.f;
	for (int i = 0; i < CUDA_GRABCUT_K; i++)
	{
		res += gmm.coefs[i] * colorGmm(gmm, i, color);
	}
	return res;
}

inline __host__ __device__ int whichComponent(gmmI& gmm, float3& color)
{
	int k = 0;
	float max = 0;

	for (int i = 0; i < CUDA_GRABCUT_K; i++)
	{
		float p = colorGmm(gmm, i, color);
		if (p > max)
		{
			k = i;
			max = p;
		}
	}
	return k;
}

inline __global__ void learningGMM(gmmI& gmmbg, gmmI& gmmfg)
{
	const int idx = threadIdx.x;
	if (idx < CUDA_GRABCUT_K2)
	{
		gmmI& gmm = idx < CUDA_GRABCUT_K ? gmmfg : gmmbg;
		int id = idx < CUDA_GRABCUT_K ? (idx) : (idx - CUDA_GRABCUT_K);
		int n = gmm.lenght[id];
		if (n != 0)
		{
			float in_v = 1.0f / n;
			gmm.coefs[id] = (float)n / gmm.totalCount;
			gmm.means[id] = gmm.sums[id] * in_v;
			float3 col1 = gmm.prods[id][0] * in_v - gmm.means[id].x * gmm.means[id];
			float3 col2 = gmm.prods[id][1] * in_v - gmm.means[id].y * gmm.means[id];
			float3 col3 = gmm.prods[id][2] * in_v - gmm.means[id].z * gmm.means[id];
			inverseMat3x3(gmm.covDeterms[id], col1, col2, col3, gmm.inverseCovs[id][0], gmm.inverseCovs[id][1], gmm.inverseCovs[id][2]);
		}
	}
}

inline __global__ void assignGMM(PtrStepSz<uchar4> source, PtrStepSz<uchar> clusterIndex, PtrStepSz<uchar> mask, gmmI& gmmbg, gmmI& gmmfg)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows)
	{
		float3 color = make_float3(source(idy, idx));

		bool bFg = checkFg(mask(idy, idx));
		//背景块,使用kcentersbg,否则使用kcentersfg
		gmmI& gmm = bFg ? gmmfg : gmmbg;
		int index = whichComponent(gmm, color);
		clusterIndex(idy, idx) = index;
	}
}

template<int blockx, int blocky>
inline __global__ void calcBeta(PtrStepSz<uchar4> source, float* tempDiffs)
{
	__shared__ float diffs[blockx * blocky];

	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	const int threadId = threadIdx.x + threadIdx.y * blockDim.x;

	diffs[threadId] = 0.f;
	__syncthreads();
	if (idx < source.cols - 1 && idy < source.rows - 1)
	{
		float4 color = make_float4(source(idy, idx));
		float4 hcolor = make_float4(source(idy, idx + 1)) - color;
		float4 vcolor = make_float4(source(idy + 1, idx)) - color;

		diffs[threadId] = dot(hcolor, hcolor) + dot(vcolor, vcolor);
	}
	__syncthreads();
	for (uint stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1)
	{
		if (threadId < stride)
		{
			diffs[threadId] += diffs[threadId + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		int blockId = blockIdx.x + blockIdx.y * gridDim.x;
		tempDiffs[blockId] = diffs[0];
	}
}

//block 1*1
template<int blockx, int blocky>
inline __global__ void calcBeta(float* tempDiffs, int arrayCount, int edgeCount, float& beta)
{
	__shared__ float diffs[blockx * blocky];
	const int threadId = threadIdx.x + threadIdx.y * blockDim.x;
	const int blockcount = blockDim.x * blockDim.y;
	diffs[threadId] = 0.f;
	__syncthreads();
	int count = arrayCount / blockcount + 1;
	for (int i = 0; i < count; i++)
	{
		int id = i * blockcount + threadId;
		if (id < arrayCount)
		{
			diffs[threadId] += tempDiffs[id];
		}
	}
	__syncthreads();
	for (uint stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1)
	{
		if (threadId < stride)
		{
			diffs[threadId] += diffs[threadId + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		float tbeta = diffs[0];
		if (tbeta < 0.000001f)
			beta = 0.f;
		else
			beta = (float)edgeCount / (2.f * tbeta);//bate越小,边的权重越大
	}
}

inline __global__ void addEdges(PtrStepSz<uchar4> source,
	PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge, float* beta, float gamma)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols - 1 && idy < source.rows - 1)
	{
		float hweight = 0.f;
		float vweight = 0.f;
		float4 color = make_float4(source(idy, idx));
		float4 hcolor = make_float4(source(idy, idx + 1)) - color;
		float4 vcolor = make_float4(source(idy + 1, idx)) - color;

		hweight = gamma * exp(-(*beta) * dot(hcolor, hcolor));//exp e^x x(-无穷-0) exp(0,1)
		vweight = gamma * exp(-(*beta) * dot(vcolor, vcolor));//

		rightEdge(idy, idx) = hweight;
		leftEdge(idy, idx + 1) = hweight;
		downEdge(idy, idx) = vweight;
		upEdge(idy + 1, idx) = vweight;
	}
}

//组成graph每个点的weight
inline __global__ void addTermWeights(PtrStepSz<uchar4> source, PtrStepSz<uchar> mask, PtrStepSz<float> push, PtrStepSz<float> sink,
	gmmI& gmmbg, gmmI& gmmfg, float lambda)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows)
	{
		float3 color = make_float3(source(idy, idx));
		uchar bfmask = mask(idy, idx);
		float fromSource = 0.f;
		float toSink = 0.f;
		if (bfmask == 0)//背景
		{
			toSink = lambda;
		}
		else if (bfmask == 2 || bfmask == 3)
		{
			fromSource = -log(colorGmm(gmmbg, color));
			toSink = -log(colorGmm(gmmfg, color));
		}
		else if (bfmask == 1)//前景
		{
			fromSource = lambda;
		}
		float dw = (fromSource - toSink);
		if (dw > 0)
			push(idy, idx) += dw;
		else
			sink(idy, idx) -= dw;

		//push(idy, idx) = fromSource;
		//sink(idy, idx) = toSink;
	}
}

inline __global__ void addTermWeights(PtrStepSz<uchar4> source, PtrStepSz<float> push, PtrStepSz<float> sink,
	gmmI& gmmbg, gmmI& gmmfg, float weightOffset) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows)
	{
		float3 color = make_float3(source(idy, idx));

		float fromSource = 0;
		float toSink = 0;
		float dw = -log(colorGmm(gmmbg, color)) + log(colorGmm(gmmfg, color)) - weightOffset;
		if (dw > 0)
			push(idy, idx) += dw;
		else
			sink(idy, idx) -= dw;

		//push(idy, idx) = fromSource;
		//sink(idy, idx) = toSink;
	}
}

//edge是自己向四边的值，pull是四边向自己的值
//每次迭代，分三段。1-产生从push流向各边。2-加上各边push来的值。3-重新设置height值
//重标记高度，如果当前点push>=sink,选择能流向的四周最小高度,根据这个高度加1,下次可以从当前节点流向这边
//前景区域 push大sink小 背景相反
//经测试,如果显存块又写又读,会有问题
template<int blockx, int blocky>
__global__ void push_relabe(PtrStepSz<float> push, PtrStepSz<float> sink, PtrStepSz<int> graphHeight,
	PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	PtrStepSz<float> rightPull, PtrStepSz<float> leftPull, PtrStepSz<float> upPull, PtrStepSz<float> downPull)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	float leftFlow = 0.f;
	float rightFlow = 0.f;
	float downFlow = 0.f;
	float upFlow = 0.f;
	if (idx > 0 && idx < push.cols - 1 && idy > 0 && idy < push.rows - 1)
	{
		__shared__ int heights[(blocky + 2) * (blockx + 2)];
		int hIndex = (threadIdx.y + 1) * (blockx + 2) + threadIdx.x + 1;
		heights[hIndex] = graphHeight(idy, idx);
		if (threadIdx.x == 0 && idx > 0)
			heights[hIndex - 1] = graphHeight(idy, idx - 1);
		if (threadIdx.x == blockx - 1 && idx < push.cols - 1)
			heights[hIndex + 1] = graphHeight(idy, idx + 1);
		if (threadIdx.y == 0 && idy > 0)
			heights[hIndex - (blockx + 2)] = graphHeight(idy - 1, idx);
		if (threadIdx.y == blocky - 1 && idy < push.rows - 1)
			heights[hIndex + (blockx + 2)] = graphHeight(idy + 1, idx);

		float flowPush = push(idy, idx);
		float sinkPush = sink(idy, idx);
		float leftedge = leftEdge(idy, idx);
		float rightedge = rightEdge(idy, idx);
		float downedge = downEdge(idy, idx);
		float upedge = upEdge(idy, idx);

		__syncthreads();
		int height = heights[hIndex];// graphHeight(idy, idx);
		int leftHeight = heights[hIndex - 1];// graphHeight(idy, idx - 1);// heights[hIndex - 1];
		int rightHeight = heights[hIndex + 1];// graphHeight(idy, idx + 1);// heights[hIndex + 1];
		int downHeight = heights[hIndex + (blockx + 2)]; // graphHeight(idy + 1, idx);// heights[hIndex + (blockx + 2)];
		int upHeight = heights[hIndex - (blockx + 2)];// graphHeight(idy - 1, idx); //heights[hIndex - (blockx + 2)];

		//Push 计算推出去的流
		//toward sink		
		if (flowPush > 0.f && sinkPush > 0.f && height == 1) {
			float minFlow = min(flowPush, sinkPush);
			flowPush -= minFlow;
			sinkPush -= minFlow;
		}
		//toward left		
		if (flowPush > 0.f && leftedge > 0.f && height == leftHeight + 1) {
			float minFlow = min(flowPush, leftedge);
			flowPush -= minFlow;
			leftedge -= minFlow;
			leftFlow = minFlow;
		}
		//toward right
		if (flowPush > 0.f && rightedge > 0.f && height == rightHeight + 1) {
			float minFlow = min(flowPush, rightedge);
			flowPush -= minFlow;
			rightedge -= minFlow;
			rightFlow = minFlow;
		}
		//toward down
		if (flowPush > 0.f && downedge > 0.f && height == downHeight + 1) {
			float minFlow = min(flowPush, downedge);
			flowPush -= minFlow;
			downedge -= minFlow;
			downFlow = minFlow;
		}
		//toward up
		if (flowPush > 0.f && upedge > 0.f && height == upHeight + 1) {
			float minFlow = min(flowPush, upedge);
			flowPush -= minFlow;
			upedge -= minFlow;
			upFlow = minFlow;
		}

		leftPull(idy, idx - 1) = leftFlow;
		rightPull(idy, idx + 1) = rightFlow;
		downPull(idy + 1, idx) = downFlow;
		upPull(idy - 1, idx) = upFlow;

		__syncthreads();

		//Push 计算拉入的值
		leftFlow = rightPull(idy, idx);
		rightFlow = leftPull(idy, idx);
		downFlow = upPull(idy, idx);
		upFlow = downPull(idy, idx);

		if (leftFlow > 0.f) {
			leftedge += leftFlow;
			flowPush += leftFlow;
		}
		if (rightFlow > 0.f) {
			rightedge += rightFlow;
			flowPush += rightFlow;
		}
		if (downFlow > 0.f) {
			downedge += downFlow;
			flowPush += downFlow;
		}
		if (upFlow > 0.f) {
			upedge += upFlow;
			flowPush += upFlow;
		}
		//relabel 重新标记各个单元对应高度
		int gheight = 1;
		if (sinkPush <= 0.f) {
			int minHeight = push.cols * push.rows;
			if (leftedge > 0.f && minHeight > leftHeight) {
				minHeight = leftHeight;
			}
			if (rightedge > 0.f && minHeight > rightHeight) {
				minHeight = rightHeight;
			}
			if (downedge > 0.f && minHeight > downHeight) {
				minHeight = downHeight;
			}
			if (upedge > 0.f && minHeight > upHeight) {
				minHeight = upHeight;
			}
			gheight = minHeight + 1;
		}
		//记录对应各个数据的新值
		push(idy, idx) = flowPush;
		sink(idy, idx) = sinkPush;
		leftEdge(idy, idx) = leftedge;
		rightEdge(idy, idx) = rightedge;
		downEdge(idy, idx) = downedge;
		upEdge(idy, idx) = upedge;

		graphHeight(idy, idx) = gheight;
	}
}

template<int blockx, int blocky>
__global__ void push_relabel(PtrStepSz<float> push, PtrStepSz<float> sink, PtrStepSz<int> graphHeight,
	PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	PtrStepSz<float> rightPull, PtrStepSz<float> leftPull, PtrStepSz<float> upPull, PtrStepSz<float> downPull)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	float leftFlow = 0.f;
	float rightFlow = 0.f;
	float downFlow = 0.f;
	float upFlow = 0.f;
	if (idx > 0 && idx < push.cols - 1 && idy > 0 && idy < push.rows - 1) {
		__shared__ int heights[(blocky + 2) * (blockx + 2)];
		int hIndex = (threadIdx.y + 1) * (blockx + 2) + threadIdx.x + 1;
		heights[hIndex] = graphHeight(idy, idx);
		if (threadIdx.x == 0 && idx > 0)
			heights[hIndex - 1] = graphHeight(idy, idx - 1);
		if (threadIdx.x == blockx - 1 && idx < push.cols - 1)
			heights[hIndex + 1] = graphHeight(idy, idx + 1);
		if (threadIdx.y == 0 && idy > 0)
			heights[hIndex - (blockx + 2)] = graphHeight(idy - 1, idx);
		if (threadIdx.y == blocky - 1 && idy < push.rows - 1)
			heights[hIndex + (blockx + 2)] = graphHeight(idy + 1, idx);

		float flowPush = push(idy, idx);
		float sinkPush = sink(idy, idx);
		float leftedge = leftEdge(idy, idx);
		float rightedge = rightEdge(idy, idx);
		float downedge = downEdge(idy, idx);
		float upedge = upEdge(idy, idx);

		__syncthreads();
		int height = heights[hIndex];// graphHeight(idy, idx);
		int leftHeight = heights[hIndex - 1];// graphHeight(idy, idx - 1);// heights[hIndex - 1];
		int rightHeight = heights[hIndex + 1];// graphHeight(idy, idx + 1);// heights[hIndex + 1];
		int downHeight = heights[hIndex + (blockx + 2)]; // graphHeight(idy + 1, idx);// heights[hIndex + (blockx + 2)];
		int upHeight = heights[hIndex - (blockx + 2)];// graphHeight(idy - 1, idx); //heights[hIndex - (blockx + 2)];

		//Push 计算推出去的流
		//toward sink		
		if (flowPush > 0.f && sinkPush > 0.f && height == 1) {
			float minFlow = min(flowPush, sinkPush);
			flowPush -= minFlow;
			sinkPush -= minFlow;
		}
		//toward left		
		if (flowPush > 0.f && leftedge > 0.f && height == leftHeight + 1) {
			float minFlow = min(flowPush, leftedge);
			flowPush -= minFlow;
			leftedge -= minFlow;
			leftFlow = minFlow;
		}
		//toward right
		if (flowPush > 0.f && rightedge > 0.f && height == rightHeight + 1) {
			float minFlow = min(flowPush, rightedge);
			flowPush -= minFlow;
			rightedge -= minFlow;
			rightFlow = minFlow;
		}
		//toward down
		if (flowPush > 0.f && downedge > 0.f && height == downHeight + 1) {
			float minFlow = min(flowPush, downedge);
			flowPush -= minFlow;
			downedge -= minFlow;
			downFlow = minFlow;
		}
		//toward up
		if (flowPush > 0.f && upedge > 0.f && height == upHeight + 1) {
			float minFlow = min(flowPush, upedge);
			flowPush -= minFlow;
			upedge -= minFlow;
			upFlow = minFlow;
		}

		leftPull(idy, idx - 1) = leftFlow;
		rightPull(idy, idx + 1) = rightFlow;
		downPull(idy + 1, idx) = downFlow;
		upPull(idy - 1, idx) = upFlow;
	}
}

template<int blockx, int blocky>
__global__ void push_relabel2(PtrStepSz<float> push, PtrStepSz<float> sink, PtrStepSz<int> graphHeight,
	PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	PtrStepSz<float> rightPull, PtrStepSz<float> leftPull, PtrStepSz<float> upPull, PtrStepSz<float> downPull)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	float leftFlow = 0.f;
	float rightFlow = 0.f;
	float downFlow = 0.f;
	float upFlow = 0.f;
	if (idx > 0 && idx < push.cols - 1 && idy > 0 && idy < push.rows - 1) {
		__shared__ int heights[(blocky + 2) * (blockx + 2)];
		int hIndex = (threadIdx.y + 1) * (blockx + 2) + threadIdx.x + 1;
		heights[hIndex] = graphHeight(idy, idx);
		if (threadIdx.x == 0 && idx > 0)
			heights[hIndex - 1] = graphHeight(idy, idx - 1);
		if (threadIdx.x == blockx - 1 && idx < push.cols - 1)
			heights[hIndex + 1] = graphHeight(idy, idx + 1);
		if (threadIdx.y == 0 && idy > 0)
			heights[hIndex - (blockx + 2)] = graphHeight(idy - 1, idx);
		if (threadIdx.y == blocky - 1 && idy < push.rows - 1)
			heights[hIndex + (blockx + 2)] = graphHeight(idy + 1, idx);

		float flowPush = push(idy, idx);
		float sinkPush = sink(idy, idx);
		float leftedge = leftEdge(idy, idx);
		float rightedge = rightEdge(idy, idx);
		float downedge = downEdge(idy, idx);
		float upedge = upEdge(idy, idx);

		__syncthreads();
		int height = heights[hIndex];// graphHeight(idy, idx);
		int leftHeight = heights[hIndex - 1];// graphHeight(idy, idx - 1);// heights[hIndex - 1];
		int rightHeight = heights[hIndex + 1];// graphHeight(idy, idx + 1);// heights[hIndex + 1];
		int downHeight = heights[hIndex + (blockx + 2)]; // graphHeight(idy + 1, idx);// heights[hIndex + (blockx + 2)];
		int upHeight = heights[hIndex - (blockx + 2)];// graphHeight(idy - 1, idx); //heights[hIndex - (blockx + 2)];
		//Push 计算拉入的值
		leftFlow = rightPull(idy, idx);
		rightFlow = leftPull(idy, idx);
		downFlow = upPull(idy, idx);
		upFlow = downPull(idy, idx);

		if (leftFlow > 0.f) {
			leftedge += leftFlow;
			flowPush += leftFlow;
		}
		if (rightFlow > 0.f) {
			rightedge += rightFlow;
			flowPush += rightFlow;
		}
		if (downFlow > 0.f) {
			downedge += downFlow;
			flowPush += downFlow;
		}
		if (upFlow > 0.f) {
			upedge += upFlow;
			flowPush += upFlow;
		}
		//relabel 重新标记各个单元对应高度
		int gheight = 1;
		if (sinkPush <= 0.f) {
			int minHeight = push.cols * push.rows;
			if (leftedge > 0.f && minHeight > leftHeight) {
				minHeight = leftHeight;
			}
			if (rightedge > 0.f && minHeight > rightHeight) {
				minHeight = rightHeight;
			}
			if (downedge > 0.f && minHeight > downHeight) {
				minHeight = downHeight;
			}
			if (upedge > 0.f && minHeight > upHeight) {
				minHeight = upHeight;
			}
			gheight = minHeight + 1;
		}
		//记录对应各个数据的新值
		push(idy, idx) = flowPush;
		sink(idy, idx) = sinkPush;
		leftEdge(idy, idx) = leftedge;
		rightEdge(idy, idx) = rightedge;
		downEdge(idy, idx) = downedge;
		upEdge(idy, idx) = upedge;

		graphHeight(idy, idx) = gheight;
	}
}

inline __global__ void bfsInit(PtrStepSz<uchar> mask, PtrStepSz<float> push, PtrStepSz<float> sink, PtrStepSz<int> graphHeight)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < mask.cols && idy < mask.rows)
	{
		int height = 0;
		int mmask = 1;
		if (push(idy, idx) > 0)
		{
			mmask = 0;
			height = 1;
		}
		else if (sink(idy, idx) > 0)
		{
			mmask = 0;
			height = -1;
		}
		mask(idy, idx) = mmask;
		graphHeight(idy, idx) = height;
	}
}

inline __global__ void bfs(PtrStepSz<uchar> mask, PtrStepSz<int> graphHeight,
	PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	int count, int& over)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx > 0 && idx < mask.cols - 1 && idy > 0 && idy < mask.rows - 1)
	{
		if (mask(idy, idx) != 0)
		{
			int leftHeight = graphHeight(idy, idx - 1);
			int rightHeight = graphHeight(idy, idx + 1);
			int downHeight = graphHeight(idy + 1, idx);
			int upHeight = graphHeight(idy - 1, idx);

			if ((leftHeight == count && rightEdge(idy, idx - 1) > 0) ||
				(rightHeight == count && leftEdge(idy, idx + 1) > 0) ||
				(downHeight == count && upEdge(idy + 1, idx) > 0) ||
				(upHeight == count && downEdge(idy - 1, idx) > 0))
			{
				graphHeight(idy, idx) = count + 1;
				mask(idy, idx) = 0;
				over = 1;
			}
		}
	}
}

inline __global__ void maxflowMask(PtrStepSz<uchar> mask, PtrStepSz<int> graphHeight)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx > 0 && idx < mask.cols - 1 && idy > 0 && idy < mask.rows - 1)
	{
		uchar bfmask = mask(idy, idx);
		if (bfmask == 2 || bfmask == 3)
		{
			if (graphHeight(idy, idx) > 0)
				bfmask = 3;
			else
				bfmask = 2;
		}
		mask(idy, idx) = bfmask;
	}
}

inline __global__ void combinGrabcutMask(PtrStepSz<uchar4> source, PtrStepSz<float4> dest, PtrStepSz<uchar> mask, float fx, float fy)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows)
	{
		uchar vmask = mask(idy, idx);
		float4 color = rgbauchar42float4(source(idy, idx));
		color.w = checkFg(vmask) ? 1.f : 0.f;
		dest(idy, idx) = color;
	}
}

template<int blockx, int blocky, bool bSeed>
__global__ void updateGMM(PtrStepSz<uchar4> source, PtrStepSz<uchar> mask, PtrStepSz<uchar> clusterIndex, float3* ksumprods, int* kindexs)
{
	__shared__ float3 sums[blockx * blocky][CUDA_GRABCUT_K2];
	__shared__ float3 prods[blockx * blocky][CUDA_GRABCUT_K2][3];
	__shared__ int indexs[blockx * blocky][CUDA_GRABCUT_K2];

	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	const int threadId = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll CUDA_GRABCUT_K2
	for (int i = 0; i < CUDA_GRABCUT_K2; i++)
	{
		sums[threadId][i] = make_float3(0.f);
		prods[threadId][i][0] = make_float3(0.f);
		prods[threadId][i][1] = make_float3(0.f);
		prods[threadId][i][2] = make_float3(0.f);
		indexs[threadId][i] = 0;
	}
	__syncthreads();
	if (idx < source.cols && idy < source.rows)
	{
		uchar umask = mask(idy, idx);
		if (!bSeed || (bSeed && checkObvious(umask)))
		{
			int index = clusterIndex(idy, idx);
			bool bFg = checkFg(umask);
			int kindex = bFg ? index : (index + CUDA_GRABCUT_K);
			float3 color = make_float3(source(idy, idx));
			sums[threadId][kindex] = color;
			prods[threadId][kindex][0] = color.x * color;
			prods[threadId][kindex][1] = color.y * color;
			prods[threadId][kindex][2] = color.z * color;
			indexs[threadId][kindex] = 1;
		}
		__syncthreads();
		for (uint stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1)
		{
			if (threadId < stride)//stride 2^n
			{
#pragma unroll CUDA_GRABCUT_K2
				for (int i = 0; i < CUDA_GRABCUT_K2; i++)
				{
					sums[threadId][i] += sums[threadId + stride][i];
					prods[threadId][i][0] += prods[threadId + stride][i][0];
					prods[threadId][i][1] += prods[threadId + stride][i][1];
					prods[threadId][i][2] += prods[threadId + stride][i][2];
					indexs[threadId][i] += indexs[threadId + stride][i];
				}
			}
			__syncthreads();
			if (threadIdx.x == 0 && threadIdx.y == 0)
			{
				int blockId = blockIdx.x + blockIdx.y * gridDim.x;
#pragma unroll CUDA_GRABCUT_K2
				for (int i = 0; i < CUDA_GRABCUT_K2; i++)
				{
					int id = blockId * CUDA_GRABCUT_K2 + i;
					ksumprods[id * 4] = sums[0][i];
					ksumprods[id * 4 + 1] = prods[0][i][0];
					ksumprods[id * 4 + 2] = prods[0][i][1];
					ksumprods[id * 4 + 3] = prods[0][i][2];
					kindexs[id] = indexs[0][i];
				}
			}
		}
	}
}

//block 1 * 1, threads
template<int blockx, int blocky>
__global__ void updateGMM(float3* ksumprods, int* kindexs, gmmI& gmmbg, gmmI& gmmfg, int gridcount)
{
	__shared__ float3 sums[blockx * blocky][CUDA_GRABCUT_K2];
	__shared__ float3 prods[blockx * blocky][CUDA_GRABCUT_K2][3];
	__shared__ int indexs[blockx * blocky][CUDA_GRABCUT_K2];

	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	const int threadId = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll CUDA_GRABCUT_K2
	for (int i = 0; i < 2 * CUDA_GRABCUT_K; i++)
	{
		sums[threadId][i] = make_float3(0.f);
		prods[threadId][i][0] = make_float3(0.f);
		prods[threadId][i][1] = make_float3(0.f);
		prods[threadId][i][2] = make_float3(0.f);
		indexs[threadId][i] = 0;
	}
	__syncthreads();
	int blockcount = blockDim.x * blockDim.y;
	int count = gridcount / blockcount + 1;
	//每块共享变量，每个线程操纵自己对应那块地址，不需要同步,用线程块操作一个大内存
	for (int i = 0; i < count; i++)
	{
		int id = i * blockcount + threadId;
		if (id < gridcount)
		{
#pragma unroll CUDA_GRABCUT_K2
			for (int j = 0; j < 2 * CUDA_GRABCUT_K; j++)
			{
				int iid = id * 2 * CUDA_GRABCUT_K + j;
				sums[threadId][j] += ksumprods[4 * iid];
				prods[threadId][j][0] += ksumprods[4 * iid + 1];
				prods[threadId][j][1] += ksumprods[4 * iid + 2];
				prods[threadId][j][2] += ksumprods[4 * iid + 3];
				indexs[threadId][j] += kindexs[iid];
			}
		}
	}
	__syncthreads();
	for (uint stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1)
	{
		if (threadId < stride)//stride 2^n
		{
#pragma unroll CUDA_GRABCUT_K2
			for (int i = 0; i < 2 * CUDA_GRABCUT_K; i++)
			{
				sums[threadId][i] += sums[threadId + stride][i];
				prods[threadId][i][0] += prods[threadId + stride][i][0];
				prods[threadId][i][1] += prods[threadId + stride][i][1];
				prods[threadId][i][2] += prods[threadId + stride][i][2];
				indexs[threadId][i] += indexs[threadId + stride][i];
			}
		}
		__syncthreads();
	}
	//聚合统计计算
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		gmmfg.totalCount = 0;
		for (int i = 0; i < CUDA_GRABCUT_K; i++)
		{
			gmmfg.sums[i] = sums[0][i];
			gmmfg.prods[i][0] = prods[0][i][0];
			gmmfg.prods[i][1] = prods[0][i][1];
			gmmfg.prods[i][2] = prods[0][i][2];
			gmmfg.lenght[i] = indexs[0][i];
			gmmfg.totalCount += indexs[0][i];
		}
		gmmbg.totalCount = 0;
		for (int i = CUDA_GRABCUT_K; i < CUDA_GRABCUT_K2; i++)
		{
			gmmbg.sums[i - CUDA_GRABCUT_K] = sums[0][i];
			gmmbg.prods[i - CUDA_GRABCUT_K][0] = prods[0][i][0];
			gmmbg.prods[i - CUDA_GRABCUT_K][1] = prods[0][i][1];
			gmmbg.prods[i - CUDA_GRABCUT_K][2] = prods[0][i][2];
			gmmbg.lenght[i - CUDA_GRABCUT_K] = indexs[0][i];
			gmmbg.totalCount += indexs[0][i];
		}
	}
}

inline __global__ void image2netData(PtrStepSz<uchar4> source, float* outData, int size)
{
	//rgbargbargba->bbbgggrrr
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows)
	{
		float4 color = rgbauchar42float4(source(idy, idx));
		int nindex = idy * source.cols + idx;
		outData[nindex] = color.x;
		outData[size + nindex] = color.y;
		outData[2 * size + nindex] = color.z;
	}
}

inline __global__ void drawRect(PtrStepSz<uchar4> source, int xmin, int xmax, int ymin, int ymax, int radius, uchar4 drawColor)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows)
	{
		int4 xx = make_int4(idx, xmax, idy, ymax);
		int4 yy = make_int4(xmin, idx, ymin, idy);

		int4 xy = abs(xx - yy);
		//只要有一个条件满足就行（分别代表左边，右边，上边，下边）
		int sum = (xy.x < radius) + (xy.y < radius) + (xy.z < radius) + (xy.w < radius);
		float2 lr = make_float2(xy.x + xy.y, xy.z + xy.w);
		float2 rl = make_float2(xmax - xmin, ymax - ymin);
		if (sum > 0 && length(lr - rl) < radius) {
			source(idy, idx) = drawColor;
		}
	}
}

inline __global__ void test_pp1(PtrStepSz<float> push, PtrStepSz<float> sink, PtrStepSz<int> graphHeight,
	PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	PtrStepSz<float> rightPull, PtrStepSz<float> leftPull, PtrStepSz<float> upPull, PtrStepSz<float> downPull)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx > 0 && idx < rightPull.cols - 1 && idy > 0 && idy < rightPull.rows - 1)
	{
		leftPull(idy, idx - 1) = 0.1f;
		rightPull(idy, idx + 1) = 0.25f;
		downPull(idy + 1, idx) = 0.5f;
		upPull(idy - 1, idx) = 1.f;
	}
}

inline __global__ void test_pp2(PtrStepSz<float> push, PtrStepSz<float> sink, PtrStepSz<int> graphHeight,
	PtrStepSz<float> rightEdge, PtrStepSz<float> leftEdge, PtrStepSz<float> upEdge, PtrStepSz<float> downEdge,
	PtrStepSz<float> rightPull, PtrStepSz<float> leftPull, PtrStepSz<float> upPull, PtrStepSz<float> downPull)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx > 0 && idx < rightPull.cols - 1 && idy > 0 && idy < rightPull.rows - 1)
	{
		float leftFlow = rightPull(idy, idx);
		float rightFlow = leftPull(idy, idx);
		float downFlow = upPull(idy, idx);
		float upFlow = downPull(idy, idx);

		leftEdge(idy, idx) = leftFlow;
		rightEdge(idy, idx) = rightFlow;
		downEdge(idy, idx) = downFlow;
		upEdge(idy, idx) = upFlow;
	}
}


