#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include "cuda_help.h"
#include "../oeip/Oeip.h"

//长宽用目标的
template <int32_t yuvpType>
__global__ void yuv2rgb(PtrStepSz<uchar> source, PtrStepSz<uchar4> dest) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < dest.cols && idy < dest.rows) {
		uchar y = source(idy, idx);
		uchar u = 0;
		uchar v = 0;
		float3 yuv = make_float3(0.f, 0.f, 0.f);
		//编译时确定分支
		if (yuvpType == 1) {
			int halfidx = idx >> 1;
			int halfidy = idy >> 1;
			u = source(halfidy + dest.rows, halfidx * 2);
			v = source(halfidy + dest.rows, halfidx * 2 + 1);
		}
		if (yuvpType == 5) {
			u = source(idy / 2 + dest.rows, idx);
			v = source(idy / 2 + dest.rows * 3 / 2, idx);
		}
		if (yuvpType == 6) {
			u = source(idy / 4 + dest.rows, idx);
			v = source(idy / 4 + dest.rows * 5 / 4, idx);
		}
		yuv = rgbauchar32float3(make_uchar3(y, u, v));
		dest(idy, idx) = rgbafloat42uchar4(make_float4(yuv2Rgb(yuv), 1.f));
	}
}

//长宽用源的packed yuyv bitx/yoffset 0/0,yvyu 2/0,uyvy 0/1 source 一点分二点
inline __global__ void yuv2rgb(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, int bitx, int yoffset) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows) {
		uchar4 yuyv = source(idy, idx);
		uchar ayuyv[4]{ yuyv.x,yuyv.y,yuyv.z,yuyv.w };
		uchar y1 = ayuyv[yoffset];
		uchar u = ayuyv[bitx + (1 - yoffset)];
		uchar y2 = ayuyv[yoffset + 2];
		uchar v = ayuyv[(2 - bitx) + (1 - yoffset)];
		float3 yuv = rgbauchar32float3(make_uchar3(y1, u, v));
		dest(idy, idx * 2) = rgbafloat42uchar4(make_float4(yuv2Rgb(yuv), 1.f));
		yuv = rgbauchar32float3(make_uchar3(y2, u, v));
		dest(idy, idx * 2 + 1) = rgbafloat42uchar4(make_float4(yuv2Rgb(yuv), 1.f));
	}
}

template <int32_t yuvpType>
__global__ void rgb2yuv(PtrStepSz<uchar4> source, PtrStepSz<uchar> dest) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows) {
		float3 rgb = make_float3(rgbauchar42float4(source(idy, idx)));
		float3 yuv = rgb2Yuv(rgb);
		dest(idy, idx) = rgbafloat2ucha1(yuv.x);
		if (yuvpType == 1) {
			int halfidx = idx >> 1;
			int halfidy = idy >> 1;
			dest(halfidy + source.rows, halfidx * 2) = rgbafloat2ucha1(yuv.y);
			dest(halfidy + source.rows, halfidx * 2 + 1) = rgbafloat2ucha1(yuv.z);
		}
		if (yuvpType == 5) {
			dest(idy / 2 + source.rows, idx) = rgbafloat2ucha1(yuv.y);
			dest(idy / 2 + source.rows * 3 / 2, idx) = rgbafloat2ucha1(yuv.z);
		}
		if (yuvpType == 6) {
			dest(idy / 4 + source.rows, idx) = rgbafloat2ucha1(yuv.y);
			dest(idy / 4 + source.rows * 5 / 4, idx) = rgbafloat2ucha1(yuv.z);
		}
	}
}

//dest 一点分二点
inline __global__ void rgb2yuv(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, int bitx, int yoffset) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < dest.cols && idy < dest.rows) {
		float4 rgba1 = rgbauchar42float4(source(idy, idx * 2));
		float4 rgba2 = rgbauchar42float4(source(idy, idx * 2 + 1));
		float3 yuv1 = rgb2Yuv(make_float3(rgba1));
		float3 yuv2 = rgb2Yuv(make_float3(rgba2));
		float4 ryuyv = make_float4(yuv1.x, (yuv1.y + yuv2.y) / 2.f, yuv2.x, (yuv1.z + yuv2.z) / 2.f);
		float yuyv[4] = { ryuyv.x,ryuyv.y,ryuyv.z,ryuyv.w };
		float4 syuyv = make_float4(yuyv[yoffset], yuyv[bitx + (1 - yoffset)], yuyv[yoffset + 2], yuyv[(2 - bitx) + (1 - yoffset)]);
		dest(idy, idx) = rgbafloat42uchar4(syuyv);
	}
}

inline __global__ void rgb2rgba(PtrStepSz<uchar3> source, PtrStepSz<uchar4> dest) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows) {
		uchar3 rgb = source(idy, idx);
		dest(idy, idx) = make_uchar4(rgb.x, rgb.y, rgb.z, 255);
	}
}

inline __global__ void rgba2bgr(PtrStepSz<uchar4> source, PtrStepSz<uchar3> dest)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows) {
		uchar4 rgba = source(idy, idx);
		dest(idy, idx) = make_uchar3(rgba.z, rgba.y, rgba.x);
	}
}

inline __global__ void argb2rgba(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < source.cols && idy < source.rows) {
		uchar4 rgba = source(idy, idx);
		dest(idy, idx) = make_uchar4(rgba.y, rgba.z, rgba.w, rgba.x);
	}
}

inline __global__ void textureMap(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, MapChannelParamet paramt) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < dest.cols && idy < dest.rows) {
		uchar4 rgba = source(idy, idx);
		uchar color[4] = { rgba.x,rgba.y,rgba.z,rgba.w };
		dest(idy, idx) = make_uchar4(color[paramt.red], color[paramt.green], color[paramt.blue], color[paramt.alpha]);
	}
}

inline __global__ void blend(PtrStepSz<uchar4> source, PtrStepSz<uchar4> blendTex, PtrStepSz<uchar4> dest, int32_t left, int32_t top, float opacity) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < dest.cols && idy < dest.rows) {
		float4 rgba = rgbauchar42float4(source(idy, idx));
		if (idx >= left && idx < left + blendTex.cols&& idy >= top && idy < top + blendTex.rows) {
			float4 rgba2 = rgbauchar42float4(blendTex(idy - top, idx - left));
			rgba = rgba2 * (1.f - opacity) + rgba * opacity;
		}
		dest(idy, idx) = rgbafloat42uchar4(rgba);
	}
}

inline __global__ void operate(PtrStepSz<uchar4> source, PtrStepSz<uchar4> dest, OperateParamet paramt) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < dest.cols && idy < dest.rows) {
		int ix = idx;
		int iy = idy;
		if (paramt.bFlipX) {
			ix = source.cols - idx;
		}
		if (paramt.bFlipY) {
			iy = source.rows - idy;
		}
		float4 rgba = rgbauchar42float4(source(iy, ix));
		float4 grgba = make_float4(powf(rgba.x, paramt.gamma), powf(rgba.y, paramt.gamma), powf(rgba.z, paramt.gamma),rgba.w);
		dest(idy, idx) = rgbafloat42uchar4(grgba);
	}
}


