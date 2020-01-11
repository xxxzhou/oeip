#pragma once
#include <npp.h>
#include <d3d11.h>
#include <cuda_d3d11_interop.h>
#include <stdint.h> 
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>  
#include "../oeip-win/Dx11Resource.h"
#include <memory>

struct Dx11CudaResource
{
	//cuda资源
	cudaGraphicsResource* cudaResource = nullptr;
	//cuda具体资源
	cudaArray* cuArray = nullptr;
	//对应DX11纹理
	ID3D11Texture2D* texture = nullptr;
	//对应CPU数据
	uint8_t* cpuData = nullptr;
	//取消绑定
	void unBind() {
		if (cudaResource != nullptr && texture != nullptr) {
			cudaGraphicsUnregisterResource(cudaResource);
			cudaResource = nullptr;
		}
	}
};

inline void setNppStream(cudaStream_t& stream) {
	auto oldStream = nppGetStream();
	if (oldStream != stream) {
		cudaDeviceSynchronize();
		nppSetStream(stream);
	}
};

//把CUDA资源复制给DX11纹理(这里有个算是我遇到最奇怪的BUG之一,有行中文注释会导致这函数不能运行?)
inline void gpuMat2D3dTexture(cv::cuda::GpuMat frame, Dx11CudaResource& cudaResource, cudaStream_t stream) {
	if (cudaResource.texture != nullptr) {
		//cuda map dx11,资源数组间map
		cudaError_t cerror = cudaGraphicsMapResources(1, &cudaResource.cudaResource, stream);
		//map单个资源 cuda->(dx11 bind cuda resource)
		cerror = cudaGraphicsSubResourceGetMappedArray(&cudaResource.cuArray, cudaResource.cudaResource, 0, 0);
		cerror = cudaMemcpy2DToArray(cudaResource.cuArray, 0, 0, frame.ptr(), frame.step, frame.cols * sizeof(int32_t), frame.rows, cudaMemcpyDeviceToDevice);
		//cuda unmap dx11
		cerror = cudaGraphicsUnmapResources(1, &cudaResource.cudaResource, stream);
	}
};

//把与DX11资源绑定显存复制出来
inline void d3dTexture2GpuMat(cv::cuda::GpuMat frame, Dx11CudaResource& cudaResource, cudaStream_t stream) {
	if (cudaResource.texture != nullptr) {
		//cuda map dx11,资源数组间map
		cudaGraphicsMapResources(1, &cudaResource.cudaResource, stream);
		//map单个资源 (dx11 bind cuda resource)->cuda
		cudaGraphicsSubResourceGetMappedArray(&cudaResource.cuArray, cudaResource.cudaResource, 0, 0);
		cudaMemcpy2DFromArray(frame.ptr(), frame.step, cudaResource.cuArray, 0, 0, frame.cols * sizeof(int32_t), frame.rows, cudaMemcpyDeviceToDevice);
		//cuda unmap dx11
		cudaGraphicsUnmapResources(1, &cudaResource.cudaResource, stream);
	}
};

//绑定一个DX11共享资源与CUDA资源,分别在DX11与CUDA都有相关引用,二边分别可读可写
inline bool registerCudaResource(Dx11CudaResource& cudaDx11, std::shared_ptr<Dx11SharedTex>& sharedResource, ID3D11Device* device, int32_t width, int32_t height) {

	cudaDx11.unBind();
	bool bInit = sharedResource->restart(device, width, height);
	if (bInit) {
		cudaDx11.texture = sharedResource->texture->texture;
		cudaError_t result = cudaGraphicsD3D11RegisterResource(&cudaDx11.cudaResource, cudaDx11.texture, cudaGraphicsRegisterFlagsNone);
		if (result != cudaSuccess) {
			logMessage(OEIP_INFO, "cudaGraphicsD3D11RegisterResource fails.");
		}
	}
	return bInit;
}

inline void reCudaAllocCpu(void** data, int32_t length) {
	if (*data != nullptr) {
		cudaFreeHost(*data);
		*data = nullptr;
	}
	cudaHostAlloc(data, length, cudaHostAllocDefault);
}

inline void reCudaAllocGpu(void** data, int32_t length) {
	if (*data != nullptr) {
		cudaFree(*data);
		*data = nullptr;
	}
	cudaMalloc(data, length);
}

#ifndef SAFE_GPUDATA
#define SAFE_GPUDATA(p)  { if (p) { cudaFree(p); p = nullptr; } } 
#endif

