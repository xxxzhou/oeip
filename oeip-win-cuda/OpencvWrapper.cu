#include "opencvcuda.h"

//opncv resize需要引入 opencv_cudawarping，包含几乎各种类型的resize模板，生成出来太大,在这不需要，单独拉出来
using namespace cv;
using namespace cv::cuda;
using namespace cv::cuda::device;

#define BLOCK_X 32
#define BLOCK_Y 8

const dim3 block = dim3(BLOCK_X, BLOCK_Y);

template <typename T>
void resize_gpu(PtrStepSz<T> source, PtrStepSz<T> dest, bool bLinear, cudaStream_t stream) {
	float fx = static_cast<float>(source.cols) / dest.cols;
	float fy = static_cast<float>(source.rows) / dest.rows;
	dim3 grid(divUp(dest.cols, block.x), divUp(dest.rows, block.y));
	if (bLinear) {
		resize_linear<T> << <grid, block, 0, stream >> > (source, dest, fx, fy);
	}
	else {
		resize_nearest<T> << <grid, block, 0, stream >> > (source, dest, fx, fy);
	}
}

//实例化几个
template void resize_gpu<uchar4>(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, bool bLinear, cudaStream_t stream);
template void resize_gpu<uchar>(PtrStepSz<uchar> source, PtrStepSz<uchar> dest, bool bLinear, cudaStream_t stream);
template void resize_gpu<float4>(PtrStepSz<float4> source, PtrStepSz<float4> dest, bool bLinear, cudaStream_t stream);

//void resize_gpuf(PtrStepSz<float4> source, PtrStepSz<float4> dest, bool bLinear, cudaStream_t stream) {
//	float fx = static_cast<float>(source.cols) / dest.cols;
//	float fy = static_cast<float>(source.rows) / dest.rows;
//	dim3 grid(divUp(dest.cols, block.x), divUp(dest.rows, block.y));
//	if (bLinear) {
//		resize_linear<float4> << <grid, block, 0, stream >> > (source, dest, fx, fy);
//	}
//	else {
//		resize_nearest<float4> << <grid, block, 0, stream >> > (source, dest, fx, fy);
//	}
//}
