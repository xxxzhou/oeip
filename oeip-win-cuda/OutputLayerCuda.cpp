#include "OutputLayerCuda.h"

OutputLayerCuda::OutputLayerCuda() {
	shardTexs.resize(inCount);
	cudaResoures.resize(inCount);
	for (int32_t i = 0; i < inCount; i++) {
		shardTexs[i] = std::make_shared< Dx11SharedTex>();
	}
	cpudatas.resize(inCount);
}

OutputLayerCuda::~OutputLayerCuda() {
	for (int32_t i = 0; i < inCount; i++) {
	}
}

void OutputLayerCuda::onParametChange(OutputParamet oldT) {
	ipCuda->resetLayers();
}

bool OutputLayerCuda::onInitBuffer() {
	if (layerParamet.bGpu) {
		device.Release();
		ctx.Release();
		createDevice11(&device, &ctx);
	}
	for (int32_t i = 0; i < inCount; i++) {
		if (layerParamet.bCpu) {
			//reCudaAllocCpu((void**)&cudaResoures[i].cpuData, selfConnects[i].width*selfConnects[i].height*OEIP_CV_ELEM_SIZE(selfConnects[i].dataType));
			cpudatas[i].resize(selfConnects[i].width*selfConnects[i].height*OEIP_CV_ELEM_SIZE(selfConnects[i].dataType));
		}
		//暂时只支持RGBA
		if (layerParamet.bGpu) {
			DXGI_FORMAT dxFormat = getDxFormat(selfConnects[i].dataType);
			registerCudaResource(cudaResoures[i], shardTexs[i], device, selfConnects[i].width, selfConnects[i].height);
		}
	}
	return LayerCuda::onInitBuffer();
}

void OutputLayerCuda::onRunLayer() {
	for (int32_t i = 0; i < outCount; i++) {
		if (layerParamet.bCpu) {
			int32_t byteWidth = selfConnects[i].width * OEIP_CV_ELEM_SIZE(selfConnects[i].dataType);
			cudaMemcpy2DAsync(cpudatas[i].data(), byteWidth, inMats[i].ptr(), inMats[i].step, byteWidth, inMats[i].rows, cudaMemcpyDeviceToHost, ipCuda->cudaStream);
			cudaStreamSynchronize(ipCuda->cudaStream);
#if _DEBUG 
			//用于image watch查看
			auto frame = cv::Mat(selfConnects[i].height, selfConnects[i].width, selfConnects[i].dataType, cpudatas[i].data());
#endif
			ipCuda->outputData(layerIndex, cpudatas[i].data(), selfConnects[i].width, selfConnects[i].height, i);
		}
		if (layerParamet.bGpu) {
			if (shardTexs[i]->texture == nullptr)
				return;
			cudaStreamSynchronize(ipCuda->cudaStream);
			CComPtr<IDXGIKeyedMutex> pDX11Mutex = nullptr;
			HRESULT hResult = shardTexs[i]->texture->texture->QueryInterface(__uuidof(IDXGIKeyedMutex), (LPVOID*)&pDX11Mutex);
			DWORD result = pDX11Mutex->AcquireSync(0, 0);
			if (result == WAIT_OBJECT_0) {
				gpuMat2D3dTexture(inMats[i], cudaResoures[i], ipCuda->cudaStream);
				shardTexs[i]->bGpuUpdate = true;
			}
			result = pDX11Mutex->ReleaseSync(1);
		}
	}
}

void OutputLayerCuda::outputGpuTex(void * device, void * texture, int32_t outputIndex) {
	ID3D11Device* dxdevice = (ID3D11Device*)device;
	ID3D11Texture2D* dxtexture = (ID3D11Texture2D*)texture;
	if (!layerParamet.bGpu || dxdevice == nullptr || dxtexture == nullptr)
		return;
	if (shardTexs[outputIndex] && shardTexs[outputIndex]->bGpuUpdate) {
		//把当前的GPU纹理复制到另一DX上下文的纹理中
		copySharedToTexture(dxdevice, shardTexs[outputIndex]->sharedHandle, dxtexture);
		shardTexs[outputIndex]->bGpuUpdate = false;
	}
}
