#pragma once

//vec_math与helper_math 都有相应的重载，故相应功能分头文件实现
#include <opencv2/core/cuda.hpp> 
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

using namespace cv;
using namespace cv::cuda;
using namespace cv::cuda::device;

//copy opencv resize实现，因为引入opencv_cudawarping太大
template <typename T> __global__
void resize_nearest(const PtrStep<T> src, PtrStepSz<T> dst, const float fx, const float fy) {
	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

	if (dst_x < dst.cols && dst_y < dst.rows) {
		const float src_x = dst_x * fx;
		const float src_y = dst_y * fy;

		dst(dst_y, dst_x) = src(__float2int_rz(src_y), __float2int_rz(src_x));
	}
}

template <typename T> __global__
void resize_linear(const PtrStepSz<T> src, PtrStepSz<T> dst, const float fx, const float fy) {
	typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type work_type;

	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

	if (dst_x < dst.cols && dst_y < dst.rows) {
		const float src_x = dst_x * fx;
		const float src_y = dst_y * fy;

		work_type out = VecTraits<work_type>::all(0);

		const int x1 = __float2int_rd(src_x);
		const int y1 = __float2int_rd(src_y);
		const int x2 = x1 + 1;
		const int y2 = y1 + 1;
		const int x2_read = ::min(x2, src.cols - 1);
		const int y2_read = ::min(y2, src.rows - 1);

		T src_reg = src(y1, x1);
		out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

		src_reg = src(y1, x2_read);
		out = out + src_reg * ((src_x - x1) * (y2 - src_y));

		src_reg = src(y2_read, x1);
		out = out + src_reg * ((x2 - src_x) * (src_y - y1));

		src_reg = src(y2_read, x2_read);
		out = out + src_reg * ((src_x - x1) * (src_y - y1));

		dst(dst_y, dst_x) = saturate_cast<T>(out);
	}
}
