#include "opencvcuda.h"

//opncv resize需要引入 opencv_cudawarping，包含几乎各种类型的resize模板，生成出来太大,在这不需要，单独拉出来
using namespace cv;
using namespace cv::cuda;
using namespace cv::cuda::device;

#define BLOCK_X 32
#define BLOCK_Y 8

const dim3 block = dim3(BLOCK_X, BLOCK_Y);

void resize_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, bool bLinear, cudaStream_t stream) {
	auto fx = static_cast<float>(source.cols) / dest.cols;
	auto fy = static_cast<float>(source.rows) / dest.rows;
	dim3 grid(divUp(dest.cols, block.x), divUp(dest.rows, block.y));
	if (bLinear) {
		resize_linear<uchar4> << <grid, block, 0, stream >> > (source, dest, fx, fy);
	}
	else {
		resize_nearest<uchar4> << <grid, block, 0, stream >> > (source, dest, fx, fy);
	}
}

void resize_gpuf(PtrStepSz<float4> source, PtrStepSz<float4> dest, bool bLinear, cudaStream_t stream) {
	auto fx = static_cast<float>(source.cols) / dest.cols;
	auto fy = static_cast<float>(source.rows) / dest.rows;
	dim3 grid(divUp(dest.cols, block.x), divUp(dest.rows, block.y));
	if (bLinear) {
		resize_linear<float4> << <grid, block, 0, stream >> > (source, dest, fx, fy);
	}
	else {
		resize_nearest<float4> << <grid, block, 0, stream >> > (source, dest, fx, fy);
	}
}
